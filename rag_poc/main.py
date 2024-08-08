####### INTEGRATION TEST WITH EXISTING LLM APIs #######

import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Give me an 3 best example of how I can use Redback Senior; a wearable technology for the elderly",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)