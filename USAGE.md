# Detailed Usage Guide

## Vector Store and Reranker Model Setup

1. Deploy Amazon OpenSearch Service:
   - Use the `manifest/setup-opensearch.yaml` file in CloudFormation to deploy a public OpenSearch domain.

2. Deploy Cohere Rerank Nimble 3 model:
   - Subscribe to the model in the [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-ea3rcr6y56jp2?sr=0-2&ref_=beagle&applicationId=AWS-Marketplace-Console).
   - Deploy the model to automatically create a SageMaker Endpoint.

3. Connect the rerank model to API Gateway:
   - Create a Lambda function to handle Cohere Rerank model calls and responses.
   - Use `manifest/lambda_handler.py` as a reference for the Lambda function configuration.
   - Ensure the Lambda execution role has permissions for the SageMaker Endpoint.
   - Create a REST API in API Gateway with a `/predict` resource and connect it to the Lambda function as a POST API.


## Environment Setup

```
Create a `.env` file with the following format:
OPENSEARCH_HOST=https://...es.amazonaws.com 
OPENSEARCH_USER=raguser 
OPENSEARCH_PASSWORD=MarsEarth1! 
OPENSEARCH_REGION=us-east-1 
RERANK_API_URL=https://....execute-api.us-east-1.amazonaws.com/v1/predict
```

Note: The provided CloudFormation template sets default credentials for OpenSearch. Modify these as needed.

## Bedrock Configuration

1. In the browser, click on 'Bedrock Settings' in the Model Settings tab.
2. Select the Bedrock region.
3. Choose the chat model and embedding model.
4. Adjust Temperature and Top-P to control answer patterns.

## Document Preprocessing

1. Click 'Upload and Process File' in the Pre-processing tab.
2. Upload a PDF document (start with shorter documents, <10 pages).
3. Specify an index name for storing processed data.
4. Select preprocessing options:
   - Set chunk size
   - Choose whether to use contextual retrieval
   - If using contextual retrieval, select an LLM for creating chunk-specific context
5. Wait for processing to complete (contextual retrieval takes longer).
    - The following prompt is used for generating chunk-specific context:

    ```
    <document> {{WHOLE_DOCUMENT}} </document>

    Here is the chunk we want to situate within the whole document
    <chunk> {{CHUNK_CONTENT}} </chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
    ```
   - To differentiate the context of each chunk, we employed a specialized system prompt.
   ```
   You're an expert at providing a succinct context, targeted for specific text chunks.

    <instruction>
    - Offer 1-5 short sentences that explain what specific information this chunk provides within the document.
    - Focus on the unique content of this chunk, avoiding general statements about the overall document.
    - Clarify how this chunk's content relates to other parts of the document and its role in the document.
    - If there's essential information in the document that backs up this chunk's key points, mention the details.
    </instruction>  
    ``` 


## Search Settings

1. Click 'Search Settings' in the Search Settings tab.
2. Specify the index name to use for searching.
3. Set the Top-K option for the context window.
4. Decide whether to use Rank Fusion:
   - If selected, adjust the number of documents to filter at each stage.


## Performing Searches

After completing the above steps, you're ready to use the chatbot for contextual retrieval-enhanced searches.


