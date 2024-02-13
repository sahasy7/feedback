from llama_index.llms import ChatMessage, MessageRole
from llama_index.prompts import ChatPromptTemplate
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from pathlib import Path
from llama_index import download_loader
import openai
import os
import streamlit as st
from trubrics.integrations.streamlit import FeedbackCollector

st.set_page_config(
    page_title="Chat with the Chat Bot",
    page_icon="🤖",  # Changed page icon to a robot emoji
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None)

# Set OpenAI API key
openai.api_key = st.secrets.openai_key

st.title("Welcome To GSM infoBot")

if "messages" not in st.session_state.keys():
    # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Need Info? Ask Me Questions about GSM Mall's Features"
        }
    ]


@st.cache_resource(show_spinner=False)
def load_data():
  with st.spinner(
      text=
      "Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."
  ):

    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    documents = reader.load_data()

    # Create an index using a chat model, so that we can use the chat prompts!
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1))

    index = VectorStoreIndex.from_documents(documents,
                                            service_context=service_context)
    return index


index = load_data()

# Text QA Prompt
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=
        ("**Welcome to GSM Mall in Hyderabad! 🛍️**\n\n"
         "Your friendly assistant is here to help! Remember, always provide clear, concise, and friendly responses within 10-15 words. value User time and aim to provide clear and concise responses. Maintain a positive and professional tone. Encourage users to visit the store subtly, without being pushy. Dont hallucinate. Let's make every interaction a delightful experience! 😊"
         ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=("Context information is below.\n"
                 "---------------------\n"
                 "{context_str}\n"
                 "---------------------\n"
                 "Given the context information and not prior knowledge, "
                 "answer the question: {query_str}\n"),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

# Refine Prompt
chat_refine_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=
        ("**Welcome to GSM Mall in Hyderabad! 🛍️**\n\n"
         "Your friendly assistant is here to help! Remember, always provide clear, concise, and friendly responses within 10-15 words. value User time and aim to provide clear and concise responses. Maintain a positive and professional tone. Encourage users to visit the store subtly, without being pushy. Dont Halluicante. Let's make every interaction a delightful experience! 😊"
         ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "We have the opportunity to refine the original answer "
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{context_msg}\n"
            "------------\n"
            "Given the new context, refine the original answer to better "
            "answer the question: {query_str}. "
            "If the context isn't useful, output the original answer again.\n"
            "Original Answer: {existing_answer}"),
    ),
]
refine_template = ChatPromptTemplate(chat_refine_msgs)

if "logged_prompt" not in st.session_state:
    st.session_state.logged_prompt = None

if "chat_engine" not in st.session_state.keys():
  # Initialize the chat engine
  st.session_state.chat_engine = index.as_query_engine(
      text_qa_template=text_qa_template, refine_template=refine_template)

if prompt := st.chat_input("Your question"):
  # Prompt for user input and save to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
  # Display the prior chat messages
  with st.chat_message(message["role"]):
    st.write(message["content"])


collector = FeedbackCollector(
    email=st.secrets.TRUBRICS_EMAIL,
    password=st.secrets.TRUBRICS_PASSWORD,
    project="default"
)


# If the last message is not from the assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
  with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
      response = st.session_state.chat_engine.query(prompt)
      st.write(response.response)
      message = {"role": "assistant", "content": response.response}
      st.session_state.messages.append(
          message)  # Add response to message history
if len(message)> 0:
    st.session_state.logged_prompt = collector.log_prompt(
        config_model={"model": "gpt-3.5-turbo"},
        prompt=prompt,
        generation=response,
    )
if st.session_state.logged_prompt:
    st.write("A model generation...")
    user_feedback = collector.st_feedback(
        component="default",
        feedback_type="thumbs",
        open_feedback_label="[Optional] Provide additional feedback",
        model=st.session_state.logged_prompt.config_model.model,
        prompt_id=st.session_state.logged_prompt.id,
        align="flex-start"
    )


