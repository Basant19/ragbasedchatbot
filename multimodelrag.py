import streamlit as st
import logging
import os
import uuid
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, UnstructuredFileLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
from ratelimit import limits, sleep_and_retry
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AutoRAGAssistant:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3,
                max_output_tokens=2048
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.vectorstore = None
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
                is_separator_regex=False
            )
            self.setup_tools()
            self.setup_agent()
        except Exception as e:
            logger.error(f"Error initializing assistant: {str(e)}")
            raise

    def setup_tools(self):
        # Knowledge Base Search Tool with enhanced retrieval
        knowledge_base_tool = Tool(
            name="knowledge_base",
            func=self.search_knowledge_base,
            description="Useful for searching information in the knowledge base. Always check here first."
        )
        
        # Improved Internet Search Tool with rate limiting
        internet_search_tool = Tool(
            name="internet_search",
            func=self.safe_search,
            description="Useful for searching the internet when you don't find the answer in the knowledge base."
        )
        
        self.tools = [knowledge_base_tool, internet_search_tool]
    
    @sleep_and_retry
    @limits(calls=3, period=60)  # 3 calls per minute
    def safe_search(self, query: str) -> str:
        """Rate-limited search with response trimming"""
        try:
            search = DuckDuckGoSearchRun(max_results=3)
            result = search.run(query)
            return result[:500]  # Limit response length
        except Exception as e:
            logger.warning(f"Search failed: {str(e)}")
            return "Search unavailable. Try again later."

    def setup_agent(self):
        prompt = hub.pull("hwchase17/react-chat")
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            return_intermediate_steps=True
        )
    
    def add_documents(self, documents: List[Document]):
        if not documents:
            raise ValueError("No documents provided")
        
        # Add metadata for better filtering
        for doc in documents:
            doc.metadata.update({
                "timestamp": datetime.now().isoformat(),
                "source_type": "pdf" if "pdf" in doc.metadata.get("source","").lower() else "web"
            })
        
        split_docs = self.text_splitter.split_documents(documents)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        else:
            self.vectorstore.add_documents(split_docs)
        
        logger.info(f"Added {len(split_docs)} document chunks to knowledge base")
    
    def search_knowledge_base(self, query: str) -> str:
        if self.vectorstore is None:
            return "No documents in knowledge base"
        
        try:
            # Use MMR for better diversity in results
            docs = self.vectorstore.max_marginal_relevance_search(
                query, 
                k=3,
                fetch_k=10,
                lambda_mult=0.5
            )
            
            # Extract precise answers using LLM
            return self._extract_precise_answer(query, docs)
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return f"Search error: {str(e)}"

    def _extract_precise_answer(self, query: str, docs: List[Document]) -> str:
        """Use LLM to extract only relevant information"""
        template = """Extract ONLY the specific information requested from these documents:
        Question: {query}
        Documents: {docs}
        
        Return:
        - "Not found" if the information isn't present
        - ONLY the exact information if found
        - Never make up answers"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        return chain.invoke({
            "query": query,
            "docs": "\n---\n".join(d.page_content for d in docs)
        }).content

    def query(self, question: str) -> Tuple[str, bool]:
        try:
            response = self.agent_executor.invoke({"input": question})
            return response["output"], True
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return f"Error: {str(e)}", False

def load_pdf(file_path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise

def load_url(url: str) -> List[Document]:
    try:
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading URL: {str(e)}")
        raise

def save_uploaded_file(uploaded_file) -> str:
    try:
        os.makedirs("temp_files", exist_ok=True)
        file_ext = os.path.splitext(uploaded_file.name)[1]
        file_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join("temp_files", file_name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise

def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up file: {str(e)}")

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
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    
    # Sidebar for settings
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
        
        st.markdown("---")
        st.markdown("**Note:** Make sure you have set your GOOGLE_API_KEY in the .env file")
    
    # Main content area
    st.title("AutoRAG Assistant with Gemini")
    
    # Check if assistant is initialized
    if not st.session_state.get("model_initialized", False):
        st.warning("Please initialize the assistant from the sidebar settings")
        return
    
    # Document upload section
    with st.expander("📁 Add Documents to Knowledge Base", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload PDF or Text File")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "txt", "docx"],
                key="file_uploader"
            )
            
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
                        st.success(f"Added {uploaded_file.name} to knowledge base!")
                    cleanup_file(file_path)
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        with col2:
            st.subheader("Add Website Content")
            url = st.text_input("Enter URL to scrape", key="url_input")
            if st.button("Add URL Content"):
                if url:
                    try:
                        with st.spinner(f"Processing {url}..."):
                            documents = load_url(url)
                            st.session_state.assistant.add_documents(documents)
                            st.session_state.uploaded_files.append(url)
                            st.success(f"Added URL content to knowledge base!")
                    except Exception as e:
                        st.error(f"Error processing URL: {str(e)}")
                else:
                    st.warning("Please enter a valid URL")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        with st.expander("📂 Uploaded Documents", expanded=False):
            for file in st.session_state.uploaded_files:
                st.write(f"- {file}")
    
    # Chat interface
    st.header("💬 Chat with AutoRAG")
    display_chat_history()
    
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, success = st.session_state.assistant.query(prompt)
                    if success:
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    else:
                        st.error(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Knowledge base search
    with st.expander("🔍 Search Knowledge Base", expanded=False):
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