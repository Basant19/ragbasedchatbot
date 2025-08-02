import langchain

print(langchain.__version__)

from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
print("Token loaded:", hf_token)  # Debug print

# Check if token is None or empty
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found. Check your .env file.")
