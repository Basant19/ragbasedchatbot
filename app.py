
import streamlit as st
from assistant import AutoRAGAssistant
from utils import save_uploaded_file, load_pdf, load_url, cleanup_file
from langchain_community.document_loaders import UnstructuredFileLoader

def init_session_state():
    if "assistant" not in st.session_state:
        st.session_state.assistant = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False

def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    st.set_page_config(
        page_title="AutoRAG Assistant",
        layout="wide"
    )
    
    init_session_state()

    
    with st.sidebar:
        st.header("Settings")
        model_option = st.selectbox(
            "Select Gemini Model",
            ("gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"),
            key="model_select"
        )

        if st.button("Initialize Assistant"):
            try:
                st.session_state.assistant = AutoRAGAssistant(model_name=model_option)
                st.session_state.model_initialized = True
                st.success("Assistant initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize assistant: {str(e)}")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            if st.session_state.assistant:
                st.session_state.assistant.memory.clear()
            st.rerun()

    
    st.title("AutoRAG Assistant with Gemini")

    if not st.session_state.model_initialized:
        st.warning("Please initialize the assistant from the sidebar settings")
        return

    
    with st.expander("Add Documents to Knowledge Base", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Upload PDF, TXT, or DOCX")
            uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"], key="file_upload")

            if uploaded_file and st.button("Process Uploaded File"):
                try:
                    file_path = save_uploaded_file(uploaded_file)
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        if uploaded_file.name.endswith(".pdf"):
                            documents = load_pdf(file_path)
                        else:
                            loader = UnstructuredFileLoader(file_path)
                            documents = loader.load()

                        st.session_state.assistant.add_documents(documents)
                        st.session_state.uploaded_files.append(uploaded_file.name)
                        st.success(f"Added {uploaded_file.name} to knowledge base")
                    cleanup_file(file_path)
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        with col2:
            st.subheader("Add Web Content")
            url = st.text_input("Enter URL to scrape", key="url_input")

            if st.button("Add URL Content"):
                if url:
                    try:
                        with st.spinner(f"Processing {url}..."):
                            documents = load_url(url)
                            st.session_state.assistant.add_documents(documents)
                            st.session_state.uploaded_files.append(url)
                            st.success("Added URL content to knowledge base")
                    except Exception as e:
                        st.error(f"Error loading URL: {str(e)}")
                else:
                    st.warning("Please enter a valid URL")

    
    if st.session_state.uploaded_files:
        with st.expander("Uploaded Documents", expanded=False):
            for file in st.session_state.uploaded_files:
                st.write(f"- {file}")

    
    st.header(" Chat with AutoRAG")
    display_chat_history()

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, success = st.session_state.assistant.query(prompt)
                    if success:
                        st.markdown(response)
                    else:
                        st.error(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

   
    with st.expander("Search Knowledge Base", expanded=False):
        search_query = st.text_input("Enter search query", key="kb_search")
        if st.button("Search Knowledge Base"):
            if search_query:
                with st.spinner("Searching knowledge base..."):
                    results = st.session_state.assistant.search_knowledge_base(search_query)
                    st.markdown("### Search Results")
                    st.markdown(results)
            else:
                st.warning("Please enter a search query")

if __name__ == "__main__":
    main()
