import streamlit as st
import httpx
import pyttsx3

# 🔊 Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume: 0.0 to 1.0

# 🌐 API endpoint
API_URL = "http://127.0.0.1:8000/chat"

# 🧠 Streamlit UI
st.title("Lachesis Health Assistant (v2)")

# User query input
user_query = st.text_input("Ask your question:")

# Optional voice toggle
use_voice = st.checkbox("Read response aloud", value=True)

# Submit button
if st.button("Ask"):
    if user_query.strip():
        try:
            # Send query to FastAPI backend
            response = httpx.post(API_URL, json={"user_query": user_query})
            if response.status_code == 200:
                chatbot_response = response.json().get("response", "")
                st.markdown(f"**Chatbot:** {chatbot_response}")

                # Speak the response aloud
                if use_voice and chatbot_response:
                    engine.say(chatbot_response)
                    engine.runAndWait()
            else:
                st.error("Chatbot failed to respond.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question before submitting.")