# assistant.py
import os
import sys
import logging
from datetime import datetime
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain.memory import ConversationBufferMemory
from ratelimit import limits, sleep_and_retry

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AutoRAGAssistant:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
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
            logger.error(f"Initialization error: {e}")
            raise

    def setup_tools(self):
        knowledge_tool = Tool(
            name="knowledge_base",
            func=self.search_knowledge_base,
            description="Search knowledge base documents."
        )
        internet_tool = Tool(
            name="internet_search",
            func=self.safe_search,
            description="Fallback tool for internet search."
        )
        self.tools = [knowledge_tool, internet_tool]

    @sleep_and_retry
    @limits(calls=3, period=60)
    def safe_search(self, query: str) -> str:
        try:
            search = DuckDuckGoSearchRun(max_results=3)
            result = search.run(query)
            return result[:500]
        except Exception as e:
            logger.warning(f"DuckDuckGo error: {e}")
            return "Search unavailable."

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
        for doc in documents:
            doc.metadata.update({
                "timestamp": datetime.now().isoformat(),
                "source_type": "pdf" if "pdf" in doc.metadata.get("source", "").lower() else "web"
            })
        split_docs = self.text_splitter.split_documents(documents)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        else:
            self.vectorstore.add_documents(split_docs)
        logger.info(f"Added {len(split_docs)} document chunks")

    def search_knowledge_base(self, query: str) -> str:
        if self.vectorstore is None:
            return "No documents in knowledge base"
        try:
            docs = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=3,
                fetch_k=10,
                lambda_mult=0.5
            )
            return self._extract_precise_answer(query, docs)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"Search failed: {e}"

    def _extract_precise_answer(self, query: str, docs: List[Document]) -> str:
        template = """Extract ONLY the specific information requested from these documents:
        Question: {query}
        Documents: {docs}

        Return:
        - \"Not found\" if the information isn't present
        - ONLY the exact information if found
        - Never make up answers"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        return chain.invoke({
            "query": query,
            "docs": "\n---\n".join([d.page_content for d in docs])
        }).content

    def query(self, question: str) -> Tuple[str, bool]:
        try:
            response = self.agent_executor.invoke({"input": question})
            return response["output"], True
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"Error: {e}", False
