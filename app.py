import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

# Load LangSmith key
os.environ["LANGSMITH_TRACING"] = "true"
with open("api.txt") as f:
    os.environ["LANGSMITH_API_KEY"] = f.read().strip()

# Get Google API Key
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = st.text_input("Enter Google API Key:", type="password")

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Title
st.title("Smart Chatbot (Gemini-2.0)")
st.write("Developed by Arshad Anwar")

# Set up memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Set up chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input from user
user_input = st.text_input("You:", key="input")

# Chat handling
if user_input:
    # Reconstruct memory
    memory = st.session_state.memory
    history = memory.chat_memory.messages

    # Add user's message
    history.append(HumanMessage(content=user_input))

    # Get model response
    response = model.invoke(history)

    # Add AI's message to memory
    history.append(AIMessage(content=response.content))

    # Save updated memory and chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("AI", response.content))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"**🧑 {speaker}:** {msg}")
    else:
        st.markdown(f"**🤖 {speaker}:** {msg}")
