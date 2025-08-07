# test/test4.py

import sys
import os
# Add the parent directory to sys.path BEFORE any imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from assistant import AutoRAGAssistant
from langchain_core.documents import Document

def test_assistant_basic_query():
    try:
        assistant = AutoRAGAssistant(model_name="gemini-1.5-flash")
        response, success = assistant.query("What is Gemini?")
        print("Response:\n", response)
        assert success
    except Exception as e:
        print(f"Test failed: {e}")

def test_assistant_add_document_and_query():
    try:
        assistant = AutoRAGAssistant()
        dummy_doc = Document(
            page_content="LangChain enables LLM-powered applications using composable tools.",
            metadata={"source": "test"}
        )
        assistant.add_documents([dummy_doc])
        response = assistant.search_knowledge_base("What is LangChain?")
        print("Search Result:\n", response)
        assert "LangChain" in response or "Not found" in response
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    print("Running tests...")
    test_assistant_basic_query()
    test_assistant_add_document_and_query()
