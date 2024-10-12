import os
import re
import io
import uuid
import json
import boto3
import logging
import cohere
import requests
from datetime import datetime
from botocore.config import Config
import pdfplumber
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSearch_Manager:
    def __init__(self, prefix='aws_'):
        self.client = self._init_opensearch()
        self.prefix = prefix 
        self.index_list = self._get_indices()
        self.content_field = 'content'

    def _init_opensearch(self):
        try:
            host = os.getenv('OPENSEARCH_HOST')
            user = os.getenv('OPENSEARCH_USER')
            password = os.getenv('OPENSEARCH_PASSWORD')
            region = os.getenv('OPENSEARCH_REGION')
            client = OpenSearch(
                hosts = [{'host': host.replace("https://", ""), 'port': 443}],
                http_auth = (user, password),
                use_ssl = True,
                verify_certs = True,
                connection_class = RequestsHttpConnection
            )
            return client
        except Exception as e:
            logger.error(f"Error initializing OpenSearch: {e}")
            return None

    def set_content_field(self, index_name):
        self.content_field = 'contextual_content' if 'contextual_' in index_name else 'content'

    def _get_mappings(self):
        mapping = {
            "settings": {
                "index.knn": True,
                "index.knn.algo_param.ef_search": 512
            },
            "mappings": {
                "properties": {
                    "metadata": {
                        "properties": {
                            "source": {
                                "type": "keyword"
                            },
                            "doc_id": {
                                "type": "keyword"
                            },
                            "timestamp": {
                                "type": "date"
                            }
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "contextual_content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "content_embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "engine": "faiss",
                            "name": "hnsw",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 16
                            },
                            "space_type": "l2"
                        }
                    }
                }
            }
        }
        return mapping

    def _get_indices(self):
        try:
            index_pattern = f"{self.prefix}*" if self.prefix else "*"
            indices = self.client.cat.indices(index=index_pattern, format="json")
            return [index['index'][len(self.prefix):] for index in indices]
        except Exception as e:
            logger.error(f"Error fetching indices: {e}")
            return []

    def refresh_index_list(self):
        self.index_list = self._get_indices()

    def create_index(self, index_name, index_action):
        full_index_name = f"{self.prefix}{index_name}"
        try:
            if index_action == "Overwrite existing index":
                if self.client.indices.exists(index=full_index_name):
                    self.client.indices.delete(index=full_index_name)
                    logger.info(f"Existing index '{full_index_name}' deleted.")

                mapping = self._get_mappings()
                self.client.indices.create(index=full_index_name, body=mapping)
                logger.info(f"Index '{full_index_name}' created successfully.")

            elif index_action == "Append to existing index":
                if not self.client.indices.exists(index=full_index_name):
                    mapping = self._get_mappings()
                    self.client.indices.create(index=full_index_name, body=mapping)
                    logger.info(f"Index '{full_index_name}' did not exist. Created new index.")
                else:
                    logger.info(f"Index '{full_index_name}' already exists. Ready to append data.")

            else:
                logger.error(f"Invalid index_action: {index_action}")
                return False

            self.refresh_index_list()
            return True

        except Exception as e:
            logger.error(f"Error performing {index_action} action on index '{full_index_name}': {e}")
            return False

    def _search(self, query, index_name, top_n=80):
        try:
            response = self.client.search(index=index_name, body=query)
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "content": hit['_source'][self.content_field],
                    "score": hit['_score'],
                    "metadata": hit['_source']['metadata'],
                    "search_method": query['query'].get('knn', 'bm25')
                }
                results.append(result)
            return results
        except Exception as e:
            logger.error(f"An error occurred during search: {e}")
            return []

    def search_by_knn(self, vector, index_name, top_n=80):
        query = {
            "size": top_n,
            "_source": [self.content_field, "metadata"],
            "query": {
                "knn": {
                    "content_embedding": {
                        "vector": vector,
                        "k": top_n
                    }
                }
            }
        }
        results = self._search(query, index_name, top_n)
        for result in results:
            result['search_method'] = 'knn'
        return results

    def search_by_bm25(self, query_text, index_name, top_n=80):
        query = {
            "size": top_n,
            "_source": [self.content_field, "metadata"],
            "query": {
                "match": {
                    self.content_field: {
                        "query": query_text,
                        "operator": "or"
                    }
                }
            }
        }
        return self._search(query, index_name, top_n)

    def _rerank_documents(self, question, documents, top_k=20):
        rerank_api_url = os.getenv('RERANK_API_URL')
        payload = {
            "documents": documents,
            "query": question,
            "rank_fields": ["content"],
            "top_n": top_k
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(rerank_api_url, json=payload, headers=headers)

            if response.status_code == 200:
                result = response.json()
                return result 
            else:
                logger.error(f"Error: API failed (status code: {response.status_code})")
                logger.error(f"response: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error in _rerank_documents: {str(e)}")
            return None

    def search_by_rank_fusion(self, query_text, vector, index_name, initial_search_results=160, hybrid_score_filter=40, final_reranked_results=20, knn_weight=0.6):
        half_initial = initial_search_results // 2
        knn_results = self.search_by_knn(vector, index_name, half_initial)
        bm25_results = self.search_by_bm25(query_text, index_name, half_initial)

        bm25_weight = 1 - knn_weight

        def _normalize_and_weight_score(results, weight):
            if not results:
                return results
            min_score = min(r['score'] for r in results)
            max_score = max(r['score'] for r in results)
            score_range = max_score - min_score
            if score_range == 0:
                return results
            for r in results:
                r['normalized_score'] = ((r['score'] - min_score) / score_range) * weight
            return results

        knn_results = _normalize_and_weight_score(knn_results, knn_weight)
        bm25_results = _normalize_and_weight_score(bm25_results, bm25_weight)

        # Combine results and calculate hybrid score
        combined_results = {}
        for result in knn_results + bm25_results:
            chunk_id = result['metadata'].get('chunk_id', result['content']) 
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result.copy()
                combined_results[chunk_id]['hybrid_score'] = result.get('normalized_score', 0)
                combined_results[chunk_id]['search_methods'] = [result['search_method']]
            else:
                combined_results[chunk_id]['hybrid_score'] += result.get('normalized_score', 0)
                if result['search_method'] not in combined_results[chunk_id]['search_methods']:
                    combined_results[chunk_id]['search_methods'].append(result['search_method'])

        # Convert back to list and sort by hybrid score
        results_list = list(combined_results.values())
        results_list.sort(key=lambda x: x['hybrid_score'], reverse=True)
        hybrid_results = results_list[:hybrid_score_filter]

        # Prepare documents for reranking
        documents_for_rerank = [
            {"content": doc['content'], "metadata": doc['metadata']} for doc in hybrid_results
        ]

        # Rerank the documents -> return ranked indices
        reranked_results = self._rerank_documents(query_text, documents_for_rerank, final_reranked_results)

        # Prepare final results
        if reranked_results and isinstance(reranked_results, dict) and 'results' in reranked_results:
            final_results = []
            for reranked_doc in reranked_results['results']:
                if isinstance(reranked_doc, dict) and 'index' in reranked_doc and 'relevance_score' in reranked_doc:
                    index = reranked_doc['index']
                    if 0 <= index < len(hybrid_results):
                        original_doc = hybrid_results[index]
                        final_doc = {
                            "content": original_doc["content"],
                            'metadata': original_doc['metadata'],
                            'score': reranked_doc['relevance_score'], 
                            'hybrid_score': original_doc['hybrid_score'],
                            'search_methods': original_doc['search_methods']
                        }   
                        final_results.append(final_doc)
                else:
                    logger.warning(f"Unexpected reranked document format: {reranked_doc}")

            final_results.sort(key=lambda x: x['score'], reverse=True)

        else:
            logger.warning("Reranking failed or returned unexpected format. Using hybrid results.")
            final_results = [{
                "content": doc["content"],
                'metadata': doc['metadata'],
                'score': doc['hybrid_score'],
                'hybrid_score': doc['hybrid_score'],
                'search_methods': doc['search_methods']
            } for doc in hybrid_results[:final_reranked_results]]
        
        return final_results

class Context_Processor:
    def __init__(self, os_manager, embed_model, bedrock_region, index_name, chunk_size, overlap, use_context_retrieval, context_model=None, max_document_len=None):
        self.os_manager = os_manager
        self.embed_model = embed_model
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.use_context_retrieval = use_context_retrieval 
        self.overlap = overlap
        if use_context_retrieval == True:
            self.overlap = 0 
        self.context_model = context_model 
        self.max_document_len = max_document_len
        self.bedrock_client = self._init_bedrock_client(bedrock_region)

    def _init_bedrock_client(self, bedrock_region):
        retry_config = Config(
            region_name=bedrock_region,
            retries={"max_attempts": 10, "mode": "standard"}
        )
        return boto3.client("bedrock-runtime", config=retry_config)

    def _split_into_chunks(self, text, chunk_size, overlap):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break

            next_newline = text.find('\n', end)
            next_sentence = text.find('. ', end)
            next_word = text.find(' ', end)

            separators = [s for s in [next_newline, next_sentence, next_word] if s != -1]
            if separators:
                end = min(separators)
                if next_newline == end:
                    end += 1
                elif next_sentence == end:
                    end += 2
                else:
                    end += 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = max(end - overlap, start + 1)

        return chunks

    def _load_and_split(self, file, start_page, end_page):
        documents = []
        full_text = ""
        with pdfplumber.open(io.BytesIO(file.getvalue())) as pdf:
            total_pages = len(pdf.pages)
            end_page = min(end_page or total_pages, total_pages)

            for page_num in range(start_page - 1, end_page):
                text = pdf.pages[page_num].extract_text()
                text = re.sub(r'\s+', ' ', text).strip()
                full_text += text + " "

        if self.use_context_retrieval and self.max_document_len:
            doc_chunks = self._split_into_chunks(full_text, self.max_document_len, 0)
        else:
            doc_chunks = [full_text]

        for doc_index, doc_chunk in enumerate(doc_chunks):
            doc_id = f"doc_{doc_index+1}"
            chunks = self._split_into_chunks(doc_chunk, self.chunk_size, self.overlap)

            document_chunks = [
                {
                    "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                    "original_index": chunk_index,
                    "content": chunk
                } for chunk_index, chunk in enumerate(chunks)
            ]

            documents.append({
                "doc_id": doc_id,
                "original_uuid": str(uuid.uuid4()),
                "content": doc_chunk,
                "chunks": document_chunks
            })

        return documents

    def _save_documents_to_json(self, documents, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

    def _situate_document(self, documents):

        sys_prompt = [{"text": "You are a helpful assistant that provides concise context for document chunks."}]
        for document in documents:
            doc_content = document['content']
            for chunk in document['chunks']:
                
                doc_content=doc_content
                document_context_prompt = f"""
                <document>
                {doc_content}
                </document>
                """

                chunk_content=chunk['content']
                chunk_context_prompt = f"""
                Here is the chunk we want to situate within the whole document
                <chunk>
                {chunk_content}
                </chunk>

                Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
                Answer only with the succinct context and nothing else.
                """
                usr_prompt = [{
                        "role": "user", 
                        "content": [
                            {"text": document_context_prompt},
                            {"text": document_context_prompt}
                        ]
                    }]

                temperature = 0.0
                top_p = 0.1
                inference_config = {"temperature": temperature, "topP": top_p}

                try:
                    response = self.bedrock_client.converse(
                        modelId=self.context_model,
                        messages=usr_prompt, 
                        system=sys_prompt,
                        inferenceConfig=inference_config,
                    )
                    situated_context = response['output']['message']['content'][0]['text']
                    chunk['contextual_content'] = situated_context.strip()
                except Exception as e:
                    logger.error(f"Error generating context for chunk: {e}")
                    chunk['contextual_content'] = ""
        return documents
    

    def _embed_document(self, text):
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.embed_model,
                body=json.dumps({"inputText": text})
            )
            return json.loads(response['body'].read())['embedding']
        except Exception as e:
            logger.error(f"Error embedding document: {e}")
            return None


    def _embed_and_store(self, source_file_name):
        try:
            with open(f"{self.index_name}_chunks.json", 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            embedded_documents = []

            for document in documents:
                doc_id = document['doc_id']
                embedded_chunks = []

                for chunk in document['chunks']:
                    context = chunk['contextual_content'] if self.use_context_retrieval else chunk['content']
                    chunk_embedding = self._embed_document(context)
                    if chunk_embedding:
                        chunk_id = chunk['chunk_id']
                        _id = f"{doc_id}_{chunk_id}"
                        embedded_chunk = {
                            "metadata": {
                                "source": source_file_name, 
                                "doc_id": doc_id,
                                "chunk_id": chunk_id,
                                "timestamp": datetime.now().isoformat()
                            },
                            "content": chunk['content'],
                            "content_embedding": chunk_embedding
                        }
                        if self.use_context_retrieval:
                            embedded_chunk["contextual_content"] = chunk['contextual_content']
                        embedded_chunks.append(embedded_chunk)

                        self.os_manager.client.index(
                            index=f"aws_{self.index_name}",
                            body=embedded_chunk
                        )
                        
                    embedded_documents.append({
                        "_id": _id,
                        "embedded_chunks": embedded_chunks
                    })
                    
            with open(f"{self.index_name}_embedded_chunks.json", 'w', encoding='utf-8') as f:
                json.dump(embedded_documents, f, ensure_ascii=False, indent=2)

            print(f"Successfully embedded and stored documents in index 'aws_{self.index_name}'")
        except Exception as e:
            print(f"Error embedding and storing documents: {e}")


    def process_file(self, file, index_action, start_page=1, end_page=None):
        documents = self._load_and_split(file, start_page, end_page)
        if self.use_context_retrieval:
            documents = self._situate_document(documents)
        self._save_documents_to_json(documents, f"{self.index_name}_chunks.json")
        self.os_manager.create_index(self.index_name, index_action)
        self._embed_and_store(file.name)
        

