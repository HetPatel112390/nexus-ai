# ==============================================================================
# rag_engine.py - The Core Artificial Intelligence Engine (LangChain)
# ==============================================================================
# WHAT IS RAG? (Retrieval-Augmented Generation)
# Standard LLMs (like ChatGPT) don't know your company's private data. 
# If you ask ChatGPT "What is our company's PTO policy?", it will hallucinate.
# RAG solves this by:
#   1. Reading your private documents.
#   2. "Chunking" them into smaller paragraphs.
#   3. Converting them into math vectors (Embeddings) and saving them in a DB.
#   4. When a user asks a question, finding the most relevant paragraph.
#   5. Sending that paragraph to the LLM and saying "Answer the question USING ONLY THIS TEXT."
# ==============================================================================

import os
from dotenv import load_dotenv

# LangChain components for loading text, chunking, embedding, and LLM communication
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# New Super-Fast Free Models
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Load environment variables (API keys) from our .env file
load_dotenv()

# We need an API key to talk to Groq's supercomputers. If it's missing, crash early to warn the developer.
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("CRITICAL ERROR: GROQ_API_KEY not found in .env file. Please add it.")

class EnterpriseRAG:
    def __init__(self, data_path="./data/company_policy.txt"):
        """
        Constructor: Initializes the AI models and builds the Vector Database.
        This runs once when the server starts.
        """
        print("[RAG ENGINE] Initializing Enterprise LLM System...")
        
        # 1. INITIALIZE MODELS
        # We use Groq's lightning fast Llama-3 model
        groq_api_key = os.environ.get("GROQ_API_KEY")
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=groq_api_key)

        # 2. LOAD & CHUNK DATA
        # Read the text file containing our private company data.
        loader = TextLoader(data_path)
        raw_documents = loader.load()

        # LLMs have a "context window" limit. We split our document into chunks of 500 characters. 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(raw_documents)
        print(f"[RAG ENGINE] Split document into {len(chunks)} chunks.")

        # 3. CREATE VECTOR DATABASE (ChromaDB)
        # We use FastEmbed because it runs locally for free without needing any API keys
        print("[RAG ENGINE] Embedding data with ultra-fast local FastEmbed...")
        self.embeddings = FastEmbedEmbeddings()
        self.vector_db = Chroma.from_documents(documents=chunks, embedding=self.embeddings)

        # A "Retriever" is a tool that takes a user query, turns it into a vector, 
        # and searches the Vector DB for the 'k' most mathematically similar chunks.
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 2})

        # 4. BUILD THE PROMPT TEMPLATE
        # This is the strict instruction we send to the LLM. 
        # We force it to act as an HR assistant and ONLY use the provided context.
        system_prompt = (
            "You are Nexus AI, a highly intelligent and helpful AI assistant. "
            "You have access to the following context to help you answer questions accurately if they relate to it. "
            "If the question is conversational or general, just chat naturally and helpfully as you normally would.\n\n"
            "Context: {context}\n"
        )
        prompt = PromptTemplate.from_template(system_prompt + "Question: {input}\nAnswer:")

        # 5. SAVE PROMPT
        self.prompt = prompt
        print("[RAG ENGINE] System Ready.")

    async def ask_question(self, user_question: str) -> dict:
        """
        Public method to query the RAG system asynchronously.
        """
        # 1. Ask the Vector Database to find the most relevant chunks of text (using async)
        retrieved_docs = await self.retriever.ainvoke(user_question)
        
        # 2. Extract just the text from those chunks and join them together
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # 3. Format the prompt with the question and the context
        formatted_prompt = self.prompt.format(context=context_text, input=user_question)
        
        # 4. Send the formatted prompt to the LLM via asynchronous network call
        response = await self.llm.ainvoke(formatted_prompt)
        
        # Return the final answer and how many chunks it read
        return {
            "answer": response.content,
            "sources_used": len(retrieved_docs)
        }

    async def astream_question(self, user_question: str):
        """
        Streams the answer token by token for an instant 'flash' UI experience.
        """
        retrieved_docs = await self.retriever.ainvoke(user_question)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        formatted_prompt = self.prompt.format(context=context_text, input=user_question)
        
        async for chunk in self.llm.astream(formatted_prompt):
            if chunk.content:
                yield chunk.content
