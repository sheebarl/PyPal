from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
import streamlit as st
import os
import json

class StreamingStreamlitCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, query_area, response_area, active_query):
        super().__init__()
        self.response_area = response_area
        self.query_area = query_area
        self.active_query = active_query
        self.active_response = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if isinstance(token, str):
            with self.query_area.chat_message("user"):
                st.write(self.active_query)
            with self.response_area.chat_message("assistant"):
                self.active_response += token
                st.write(self.active_response, unsafe_allow_html=True)

def init_page(config):

    st.set_page_config(
        page_title=config['app_name'], 
        page_icon=config['app_icon'],
        initial_sidebar_state="collapsed"
    )
    st.header(config['app_name'])
    #st.markdown(f"<small><i>Powered by {model_name} at {llm_supplier.capitalize()}</i></small>", unsafe_allow_html=True)
    st.write(config['app_descr'])
    
    if "response_area" not in st.session_state:
        st.session_state.response_area = st.empty()
    if "query_area" not in st.session_state:
        st.session_state.query_area = st.empty()


def init_messages(config):

    if "system_prompt" not in st.session_state:
        try:
            with open(config['system_prompt_filename'], "r", encoding="utf-8") as f:
                st.session_state.system_prompt = f.read()
        except FileNotFoundError:
            st.warning("Please create a system_message.txt file in the root directory of the project!")
            st.stop()
    system_prompt = st.session_state.system_prompt

    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content=system_prompt)
        ]

def init_llm(model_name, api_type, temperature, api_config_file):

    # Get API config from json file
    with open(api_config_file, "r") as f:
        apis = json.load(f)

    try:
        api_config = next((item for item in apis if item["model"] == model_name), None)
    except KeyError:
        raise KeyError(f"Please make sure the model {model_name} is defined in azure_api_config.json!")

    # Get key, version, and endpoint from environment variable
    try:
        key = api_config["key"]
        version = api_config["version"]
        endpoint = api_config["endpoint"]
        
    except KeyError:
        raise KeyError(f"Error reading API config file: {api_config_file}")
    
    # Initialize 
    llm = AzureChatOpenAI(
        streaming=True,
        temperature=temperature,
        deployment_name="",
        openai_api_type=api_type,
        openai_api_version=version,
        openai_api_base=endpoint,
        openai_api_key=key,
    )
   
    return llm


def get_answer(llm, messages):
    user_message = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
    
    callback_handler = StreamingStreamlitCallbackHandler(
        query_area = st.empty(),
        response_area = st.empty(),
        active_query=user_message
    )

    answer = llm(messages, callbacks=[callback_handler])

    return answer.content

def read_config(config_file):

    # Import the JSON file 'config.json' as a dictionary
    if "config" not in st.session_state:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        st.session_state.config = config

    return st.session_state.config

def display_chat_history(messages):

    # Loop through messages and display them
    print("Messages: ", messages)
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)        
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

def main():

    # Initialize conversation_started flag
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False

    # Read config file
    config = read_config("./config.json")

    # Initialize page
    init_page(config)

    # Initialize LLM
    st.sidebar.title("Options")
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("GPT4", "gpt-35-turbo", "GPT-4-Turbo"))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)

    llm = init_llm(
        model_name = model_name,
        api_type = "azure",
        temperature = temperature,
        api_config_file = config['api_config_file'],
    )

    # Initialize messages
    init_messages(config)

    # Display system prompt
    st.sidebar.title("System Prompt")
    st.sidebar.write(st.session_state.system_prompt)

    # If conversation has not started, add a system message to start it
    if not st.session_state.conversation_started:
        # Add the bot's reply to the conversation
        st.session_state.messages.append(AIMessage(content=config["start_message"]))
        # Set the conversation_started flag to True
        st.session_state.conversation_started = True

    # Display chat history
    display_chat_history(st.session_state.messages)

    # Supervise user input
    max_words = config['max_words_per_query']
    user_input = st.chat_input(f"Enter your input (max {max_words} words):")
    if user_input:

        # Truncate the input to 200 words if necessary
        words = user_input.split()
        if len(words) > max_words:
            user_input = " ".join(words[:max_words])
            status_message = st.empty()
            status_message.warning(f"Input truncated to {max_words} words.")

        # Add the user's input to the conversation
        st.session_state.messages.append(HumanMessage(content=user_input))
        print("User input: ", user_input)
        answer = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
   
if __name__ == "__main__":
    main()