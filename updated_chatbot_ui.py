import streamlit as st
import httpx

# Ensure you are using Streamlit v1.25 or higher otherwise it would cause a fail due to version issue

# Set the API endpoint
API_URL = "http://127.0.0.1:8000/chat"

# Title
st.title("Lachesis Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history using st.chat_message()
for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

# Input container at the bottom
if user_query := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    try:
        with st.spinner("Thinking..."):
            response = httpx.post(API_URL, json={"user_query": user_query}, timeout=10)

        if response.status_code == 200:
            chatbot_response = response.json().get("response", "No response received.")
        else:
            chatbot_response = f"Error: Status code {response.status_code}"

    except httpx.ConnectError:
        chatbot_response = "Connection error: Unable to connect to the chatbot API."
    except httpx.TimeoutException:
        chatbot_response = "Timeout: The chatbot server took too long to respond."
    except Exception as e:
        chatbot_response = f"Unexpected error: {e}"

    with st.chat_message("assistant"):
        st.markdown(chatbot_response)

    # Store messages
    st.session_state.chat_history.append(("You", user_query))
    st.session_state.chat_history.append(("Chatbot", chatbot_response))

# Add a button in sidebar to clear chat
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")
