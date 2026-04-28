from dotenv import load_dotenv
load_dotenv('.env')
import os
from google import genai

key = os.getenv('GEMINI_API_KEY', '')
print(f"Key present: {bool(key)}, length: {len(key)}")

client = genai.Client(api_key=key)
models = ['gemini-2.0-flash-lite', 'gemini-1.5-pro', 'gemini-1.5-flash-8b', 'gemini-2.5-flash-preview-05-20']

for model in models:
    try:
        r = client.models.generate_content(model=model, contents='Say OK')
        print(f"OK: {model} -> {r.text[:40]}")
        break
    except Exception as e:
        print(f"FAIL: {model} -> {str(e)[:150]}")
