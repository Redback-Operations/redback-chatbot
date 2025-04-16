import streamlit as st
import httpx

# Set the API endpoint
API_URL = "http://127.0.0.1:8000/chat"

# Title
st.title("Lachesis Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input
user_query = st.text_input("Enter your query:")

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")

# Button to send query
if st.button("Get Response"):
    if user_query.strip():
        try:
            with st.spinner("Waiting for response..."):
                response = httpx.post(API_URL, json={"user_query": user_query}, timeout=10)

            if response.status_code == 200:
                data = response.json()
                chatbot_response = data.get("response", "No response received.")

                # Save to chat history
                st.session_state.chat_history.append(("You", user_query))
                st.session_state.chat_history.append(("Chatbot", chatbot_response))
            else:
                st.error(f"Server returned status code {response.status_code}")

        except httpx.ConnectError:
            st.error("Connection error: Unable to connect to the chatbot API.")
        except httpx.TimeoutException:
            st.error("Timeout: The chatbot server took too long to respond.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.warning("Please enter a valid query.")

# Display chat history
for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}**: {msg}")
