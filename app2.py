import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader

# Personal Input
cv_path = "PATH_TO_CV"
model = "MODEL_NAME"

# Load CV
@st.cache_resource
def load_cv_text(cv_path):
    loader = PyMuPDFLoader(cv_path)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)

cv_text = load_cv_text(cv_path)

# Load model
llm = ChatOllama(model=model)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# System prompt with CV included
system_message = SystemMessage(
    content=f'''
    You are a helpful assistant who writes a cover letter based on a given job description.

    Tailor it based on the user's CV on relevant experience to the job description.

    Here is the user's CV:\n{cv_text}
''')


# Streamlit Interface
st.title("💬 Cover Letter Chatbot (with CV memory)")
st.markdown("Chat with your assistant to build and refine your cover letter. Start by pasting a job description!")

user_input = st.chat_input("Say something (e.g., paste a job description or ask for changes)...")

if user_input:
    # Append user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Build full conversation
    full_chat = [system_message] + st.session_state.chat_history

    # Get model response
    with st.spinner("Thinking..."):
        response = llm.invoke(full_chat)

    # Append model response
    st.session_state.chat_history.append(response)

# Display chat history
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").markdown(msg.content)
    else:
        st.chat_message("assistant").markdown(msg.content)