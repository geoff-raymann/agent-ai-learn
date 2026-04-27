import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

question = input("Ask something: ")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI teacher."
        },
        {
            "role": "user",
            "content": question
        }
    ],
    temperature=0.7,
    max_tokens=300
)

print("\nAnswer:\n")
print(response.choices[0].message.content)