import json
import boto3

def lambda_handler(event, context):
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

    try:
        payload = json.loads(event['body'])
        documents = payload['documents']
        query = payload['query']
        rank_fields = payload.get('rank_fields', ["Title", "Content"])
        top_n = payload.get('top_n', 2)
    except:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid JSON in request body')
        }

    input_data = {
        "documents": documents,
        "query": query,
        "rank_fields": rank_fields,
        "top_n": top_n
    }

    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='Endpoint-Cohere-Rerank-3-Nimble-Model-Multi-1', 
            ContentType='application/json',
            Body=json.dumps(input_data)
        )

        result = json.loads(response['Body'].read().decode())

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps(str(e))
        }