# 🤖 AutoRAG Assistant – Retrieval-Augmented Chatbot Powered by Gemini 

AutoRAG Assistant is a **Streamlit-based intelligent chatbot** that leverages **Retrieval-Augmented Generation (RAG)** with Google’s **Gemini models** to provide **context-aware, document-informed** answers. It allows users to upload documents or add URLs, enabling the chatbot to learn dynamically and provide more relevant responses in real time.

---

## 📌 Key Features

- ✅ **Gemini Integration**: Supports `gemini-1.5-flash`, `gemini-1.5-pro`, and `gemini-pro` models.
- 📄 **Document Upload**: Supports `.pdf`, `.txt`, `.docx` file formats for contextual learning.
- 🌐 **Web Scraping**: Add knowledge directly from web pages using their URLs.
- 🔎 **Knowledge Base Search**: Built-in search engine over uploaded content.
- 🧠 **Agent-Driven Reasoning**: Uses LangChain ReAct agent for multi-step reasoning.
- 💬 **Memory Handling**: Keeps conversation context for smoother user experience.
- 🛡️ **Robust Error Logging & Exception Handling**
- ⚡ **Rate-limited Internet Search (DuckDuckGo)** as fallback for out-of-scope queries.

---
## 🚀 How It Works

1. **Assistant Initialization**:
   - Select a Gemini model from the sidebar and click "Initialize Assistant".
   
2. **Add Knowledge Base**:
   - Upload PDF, TXT, or DOCX files
   - OR add a URL to fetch and ingest content from the web

3. **Ask Questions**:
   - Use the chat box to ask questions based on the uploaded data.
   - Assistant uses retrieval + generative modeling for accurate responses.

4. **Search Mode**:
   - Use “Search Knowledge Base” to query documents directly without agent reasoning.

---
👨‍💻 Author
Made by Basant Tirkey
📫 Reach out on LinkedIn:https://www.linkedin.com/in/basant-tirkey-28767620a/