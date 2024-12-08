import streamlit as st
import requests

# Set the API endpoint to localhost (local only)
API_URL = "http://127.0.0.1:8000/chat"

# Streamlit application
st.title("Lachesis Chatbot")

# Input for user query
user_query = st.text_input("Enter your query:")

# Button to submit the query
if st.button("Get Response"):
    if user_query:
        try:
            # Send the query to the API
            response = requests.post(API_URL, json={"user_query": user_query})
            
            if response.status_code == 200:
                # Display the response from the API
                chatbot_response = response.json().get("response")
                st.write(f"Chatbot: {chatbot_response}")
            else:
                st.write("Error: Unable to get response from the chatbot.")
        except Exception as e:
            st.write(f"Error: {e}")
    else:
        st.write("Please enter a query.")