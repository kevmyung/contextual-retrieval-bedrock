import streamlit as st
import boto3
from libs.common_utils import initialize_session_state, create_toolbar, handle_ai_response

MAX_MESSAGE_HISTORY = 10
st.set_page_config(page_title='Bedrock AI Chatbot', page_icon="🤖", layout="wide")
st.title("🤖 Bedrock AI Chatbot")

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            content = message.get("user_prompt_only", message['content']) if message["role"] == "user" else message['content']
            st.markdown(content)

def main():
    initialize_session_state()
    model_info, embed_model_id, model_kwargs = create_toolbar()

    display_chat_messages()

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt, "user_prompt_only": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    bedrock_client = boto3.client('bedrock-runtime', region_name=model_info['region_name'])
    model_id = model_info['model_id']

    handle_ai_response(prompt, bedrock_client, model_id, model_kwargs, embed_model_id, history_length=MAX_MESSAGE_HISTORY)

if __name__ == "__main__":
    main()