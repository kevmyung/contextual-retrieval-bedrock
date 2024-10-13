import io
import re
import os
import yaml
import json
import streamlit as st
import pdfplumber
import logging
from dotenv import load_dotenv
from libs.rag_utils import Context_Processor, OpenSearch_Manager

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INIT_MESSAGE = "Hello! I am the Bedrock AI Chatbot. How can I help you today?"
DEFAULT_TOP_K = 20
DEFAULT_INITIAL_SEARCH_RESULTS = 160
DEFAULT_HYBRID_SCORE_FILTER = 40
DEFAULT_FINAL_RERANKED_RESULTS = 20
DEFAULT_KNN_WEIGHT = 0.6
DEFAULT_BM25_WEIGHT = 0.4
BEDROCK_REGIONS = ['us-west-2', 'us-east-1', 'ap-northeast-1', 'ap-northeast-2']

def new_chat():
    st.session_state.messages = [{"role": "assistant", "content": INIT_MESSAGE}]

def initialize_session_state():
    default_values = {
        "messages": [{"role": "assistant", "content": INIT_MESSAGE}],
        "os_manager": OpenSearch_Manager(),
        "rank_fusion": False,
        "top_k": DEFAULT_TOP_K,
        "initial_search_results": DEFAULT_INITIAL_SEARCH_RESULTS,
        "hybrid_score_filter": DEFAULT_HYBRID_SCORE_FILTER,
        "final_reranked_results": DEFAULT_FINAL_RERANKED_RESULTS,
        "knn_weight": DEFAULT_KNN_WEIGHT,
        "bm25_weight": DEFAULT_BM25_WEIGHT,
        "search_target": None
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.search_target is None and st.session_state.os_manager.index_list:
        st.session_state.search_target = st.session_state.os_manager.index_list[0]

def load_model_config():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(file_dir, "config.yml")

    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def parse_stream(stream):
    for chunk in stream:
        if 'contentBlockDelta' in chunk:
            delta = chunk['contentBlockDelta']['delta']
            if 'text' in delta:
                yield delta['text']
        elif 'messageStop' in chunk:
            return


def get_model_settings(model_config):
    with st.popover("Bedrock Settings", use_container_width=True):
        model_name = st.selectbox('Chat Model 💬', list(model_config['models'].keys()), key='model_name')
        model_info = model_config['models'][model_name]
        model_info["region_name"] = st.selectbox("Bedrock Region 🤖", BEDROCK_REGIONS, key='bedrock_region')

        embedding_model = st.selectbox(
            'Embedding Model', 
            list(model_config['embedding_models'].keys()),
            key='embedding_model'
        )
        embed_model_id = model_config['embedding_models'][embedding_model]['model_id']

        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                'Temperature', 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key="temperature_input",
                on_change=update_state,
                args=("temperature",)
            )
        with col2:
            top_p = st.slider(
                'Top P', 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.01,
                key="top_p_input",
                on_change=update_state,
                args=("top_p",)
            )

        return model_info, embed_model_id, {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 4096,
            "system_prompt": "You are a helpful AI assistant. Your task is to provide accurate and relevant answers based on the given context.\n",
        }


def update_state(key):
    if key in st.session_state:
        st.session_state[key] = st.session_state[f"{key}_input"]  


def search_toolbar():
    with st.popover("Advanced Search", use_container_width=True):
        index_list = st.session_state.os_manager.index_list
        if index_list:
            selected_index = st.selectbox(
                "Search Target (Index Name)",
                options=index_list,
                index=0 if "search_target" not in st.session_state or st.session_state.search_target not in index_list else index_list.index(st.session_state.search_target),
                key="search_target_input"
            )
            if selected_index != st.session_state.search_target:
                st.session_state.search_target = selected_index
        else:
            st.warning("No search indices available. Please upload and process a file first.")
            st.session_state.search_target = None

        if st.button("Refresh", key="refresh_index_list"):
            with st.spinner("Refreshing index list..."):
                st.session_state.os_manager.index_list = st.session_state.os_manager._get_indices()
            st.success("Index list refreshed!")
            st.session_state.pop('search_target_input', None)

        rerank_api_url = os.getenv('RERANK_API_URL')
        if not rerank_api_url:
            st.warning("RERANK_API_URL is not set in '.env'. \n\nRank Fusion is disabled.")
            st.session_state.rank_fusion = False

        rank_fusion_disabled = not rerank_api_url

        st.checkbox(
            "Use Rank Fusion",
            value=st.session_state.rank_fusion,
            key="rank_fusion_input",
            on_change=update_state,
            args=("rank_fusion",),
            disabled=rank_fusion_disabled
        )

        with st.expander("Advanced Settings"):
            if st.session_state.rank_fusion and not rank_fusion_disabled:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.number_input(
                        "Initial Retrieval",
                        min_value=20,
                        max_value=500,
                        value=st.session_state.initial_search_results,
                        step=20,
                        key="initial_search_results_input",
                        help="Number of documents to retrieve in the initial search",
                        on_change=update_state,
                        args=("initial_search_results",)
                    )

                with col2:
                    st.number_input(
                        "Hybrid Retrieval",
                        min_value=10,
                        max_value=st.session_state.initial_search_results,
                        value=min(st.session_state.hybrid_score_filter, st.session_state.initial_search_results),
                        step=10,
                        key="hybrid_score_filter_input",
                        help="Number of documents to keep after hybrid score filtering",
                        on_change=update_state,
                        args=("hybrid_score_filter",)
                    )

                with col3:
                    st.number_input(
                        "Final Ranked Results",
                        min_value=1,
                        max_value=st.session_state.hybrid_score_filter,
                        value=min(st.session_state.final_reranked_results, st.session_state.hybrid_score_filter),
                        step=1,
                        key="final_reranked_results_input",
                        help="Number of documents to return after final reranking",
                        on_change=update_state,
                        args=("final_reranked_results",)
                    )

                st.slider(
                    "KNN Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.knn_weight,
                    step=0.1,
                    key="knn_weight_input",
                    help="Weight for KNN score in hybrid scoring",
                    on_change=update_state,
                    args=("knn_weight",)
                )
                st.session_state.bm25_weight = 1 - st.session_state.knn_weight

            else:
                st.number_input(
                    "Top-K Results",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.top_k,
                    key="top_k_input",
                    on_change=update_state,
                    args=("top_k",)
                )

def upload_toolbar(model_config, embed_model_id):
    with st.popover("Upload & Process File", use_container_width=True):
        use_context_retrieval = st.checkbox(
            "Use Context Retrieval", 
            value=False,
            help="Enables contextual retrieval for more accurate results."
        )

        default_index_name = "contextual_test" if use_context_retrieval else "test"
        base_index_name = st.text_input("Index Name", value=default_index_name)
        index_name = f"contextual_{base_index_name}" if use_context_retrieval and not base_index_name.startswith("contextual_") else base_index_name

        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if use_context_retrieval:
            context_model = st.selectbox(
                'Context Processing Model 💬', 
                list(model_config['models'].keys()), 
                index=1 if len(list(model_config['models'].keys())) > 1 else 0
            )
            context_model_id = model_config['models'][context_model]['model_id']

            max_document_len = st.number_input(
                "Max Document Length", 
                min_value=1000, 
                max_value=100000, 
                value=50000, 
                step=1000,
                key="max_document_len_input",
                help="If a document exceeds this length, it will be split into multiple documents before processing",
                on_change=update_state,
                args=("max_document_len",)
            )

        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk Size", 
                min_value=100, 
                max_value=1000, 
                value=500, 
                step=50,
                key="chunk_size_input",
                on_change=update_state,
                args=("chunk_size",)
            )        
        with col2:
            overlap = st.number_input(
                "Overlap", 
                min_value=0, 
                max_value=200, 
                value=50, 
                step=10,
                key="overlap_input",
                on_change=update_state,
                args=("overlap",)
            )

        # File processing options outside the popover
        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name, "FileSize": uploaded_file.size}
            with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                total_pages = len(pdf.pages)

            st.write("Select page range for processing:")
            col1, col2 = st.columns(2)
            with col1:
                start_page = st.number_input("Start Page", min_value=1, max_value=total_pages, value=1)
            with col2:
                end_page = st.number_input("End Page", min_value=start_page, max_value=total_pages, value=total_pages)

            index_action = st.radio(
                "If index already exists:",
                ("Overwrite existing index", "Append to existing index")
            )

            if st.button("Process File"):
                if re.match(r'^[a-z0-9_]+$', index_name):
                    context_processor = Context_Processor(
                        st.session_state.os_manager,
                        embed_model_id,
                        st.session_state.bedrock_region,
                        index_name, 
                        chunk_size, 
                        overlap, 
                        use_context_retrieval, 
                        context_model=(context_model_id if use_context_retrieval else None),
                        max_document_len=(max_document_len if use_context_retrieval else None)
                    )
                    with st.spinner("Processing file..."):
                        context_processor.process_file(uploaded_file, index_action, start_page=start_page, end_page=end_page)
                        st.success(f"File processed successfully! - {index_name}")
                else:
                    st.error("Index name should contain only English letters (a-z), numbers, and '_'.")

def evaluation_toolbar():
    with st.popover("Evaluation Setup", use_container_width=True):
        st.write("To be implemented...")


def create_toolbar():
    with st.sidebar:
        st.button("New Chat", on_click=new_chat, type="secondary")

        model_config = load_model_config()
        model_info, embed_model_id, model_kwargs = get_model_settings(model_config)

        upload_toolbar(model_config, embed_model_id)

        search_toolbar()

        evaluation_toolbar()

        return model_info, embed_model_id, model_kwargs


def build_valid_message_history(messages, max_length):
    history = []
    idx = len(messages) - 1

    while idx >= 0 and messages[idx]['role'] != 'user':
        idx -= 1

    while idx >= 0 and len(history) < max_length:
        if messages[idx]['role'] == 'user':
            user_msg = messages[idx]
            idx -= 1
            history.insert(0, user_msg)
            if idx >= 0 and messages[idx]['role'] == 'assistant':
                assistant_msg = messages[idx]
                idx -= 1
                history.insert(0, assistant_msg)
            else:
                break
        else:
            idx -= 1

    while history and history[0]['role'] != 'user':
        history = history[1:]

    while history and history[-1]['role'] != 'user':
        history = history[:-1]

    if not history or history[0]['role'] != 'user' or history[-1]['role'] != 'user':
        raise ValueError("Conversation must start and end with a user message.")

    return history[-max_length:]


def invoke_model(bedrock_client, model_id, message_history, model_kwargs, history_length, search_result):
    valid_history = build_valid_message_history(message_history, history_length)

    # RAG context parsing
    additional_info = ""
    if isinstance(search_result, list):
        for item in search_result:
            if isinstance(item, dict) and 'content' in item:
                additional_info += f"- {item['content']}\n\n"
    elif isinstance(search_result, str):
        try:
            search_result_list = json.loads(search_result.replace("'", '"'))
            for item in search_result_list:
                if 'content' in item:
                    additional_info += f"- {item['content']}\n\n"
        except json.JSONDecodeError:
            additional_info = "Error parsing search result."
    else:
        additional_info = "Unexpected search result format."

    temp_last_message = None
    if valid_history and valid_history[-1]['role'] == 'user':
        last_user_message = valid_history[-1]['content']
        temp_last_message = {
            'role': 'user',
            'content': f"{last_user_message}\n\nAdditional Information:\n{additional_info}"
        }

    bedrock_messages = [{'role': msg['role'], 'content': [{'text': msg['content']}]} for msg in valid_history[:-1] if 'content' in msg]
    if temp_last_message:
        bedrock_messages.append({'role': 'user', 'content': [{'text': temp_last_message['content']}]})

    # Model Invocation
    response = bedrock_client.converse_stream(
        modelId=model_id,
        messages=bedrock_messages,
        system=[{'text': model_kwargs['system_prompt']}],
        inferenceConfig={
            'maxTokens': model_kwargs['max_tokens'],
            'temperature': model_kwargs['temperature'],
            'topP': model_kwargs['top_p']
        }
    )
    return parse_stream(response['stream'])


def retrieve_search_results(question, bedrock_client, embed_model_id):
    if not st.session_state.search_target:
        st.warning("No search target (index) selected. Please upload and process a file, then select a search target.")
        return []

    response = bedrock_client.invoke_model(
        modelId=embed_model_id,
        body=json.dumps({"inputText": question})
    )
    embedding = json.loads(response['body'].read())['embedding']

    index_name = f"aws_{st.session_state.search_target}"

    if st.session_state.rank_fusion:
        search_result = st.session_state.os_manager.search_by_rank_fusion(
            query_text=question,
            vector=embedding,
            index_name=index_name,
            initial_search_results=st.session_state.initial_search_results,
            hybrid_score_filter=st.session_state.hybrid_score_filter,
            final_reranked_results=st.session_state.final_reranked_results,
            knn_weight=st.session_state.knn_weight
        )
    else:  
        search_result = st.session_state.os_manager.search_by_knn(
            vector=embedding,
            index_name=index_name,
            top_n=st.session_state.top_k
        )

    return search_result

def display_search_results(search_result):
    st.subheader("Search Results")
    for i, result in enumerate(search_result, 1):
        score = result.get('score', 'N/A')
        score_type = "Score"

        with st.expander(f"Result {i} ({score_type}: {score})", expanded=False):
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**Content:** {result.get('content', 'No content available')}")

            metadata = result.get('metadata', {})
            col2.markdown("**Metadata:**")
            col2.markdown(f"**Source:** {metadata.get('source', 'N/A')}")
            col2.markdown(f"**Doc ID:** {metadata.get('doc_id', 'N/A')}")
            col2.markdown(f"**Timestamp:** {metadata.get('timestamp', 'N/A')}")
            col2.markdown(f"**{score_type}:** {score}")

            if 'hybrid_score' in result:
                col2.markdown(f"**Hybrid Score:** {result.get('hybrid_score', 'N/A')}")
            if 'search_methods' in result:
                col2.markdown(f"**Search Methods:** {', '.join(result.get('search_methods', ['N/A']))}")

    st.markdown(f"**Total Results:** {len(search_result)}")


def handle_ai_response(question, bedrock_client, model_id, model_kwargs, embed_model_id, history_length):
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            ai_answer = ""
            message_history = st.session_state.messages.copy()

            with st.spinner('Searching...'):
                search_result = retrieve_search_results(question, bedrock_client, embed_model_id)
                if search_result:
                    display_search_results(search_result)
            
            try:
                response_stream = invoke_model(bedrock_client, model_id, message_history, model_kwargs, history_length, search_result)
                for text_chunk in response_stream:
                    ai_answer += text_chunk
                    message_placeholder.markdown(ai_answer + "▌")
                message_placeholder.markdown(ai_answer)
                st.session_state.messages.append({"role": "assistant", "content": ai_answer})
            except ValueError as e:
                st.error(str(e))