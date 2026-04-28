from dotenv import load_dotenv
load_dotenv('.env')
import os
from google import genai

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
print("Available models:")
for m in client.models.list():
    print(" ", m.name)
