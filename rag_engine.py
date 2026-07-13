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

from langchain_core.prompts import PromptTemplate

# New Super-Fast Free Models
from langchain_groq import ChatGroq

# Load environment variables (API keys) from our .env file
load_dotenv()

# We need an API key to talk to Groq's supercomputers. If it's missing, crash early to warn the developer.
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("CRITICAL ERROR: GROQ_API_KEY not found in .env file. Please add it.")

class EnterpriseRAG:
    def __init__(self, data_path=None):
        """
        Constructor: Initializes the AI model.
        This runs once when the server starts.
        """
        print("[RAG ENGINE] Initializing General AI System...")
        
        # 1. INITIALIZE MODELS
        # We use Groq's lightning fast Llama-3 model
        groq_api_key = os.environ.get("GROQ_API_KEY")
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=groq_api_key)

        # 2. BUILD THE PROMPT TEMPLATE
        # This is the strict instruction we send to the LLM. 
        system_prompt = (
            "You are Nexus AI, a highly intelligent and helpful AI assistant. "
            "You provide clear, accurate, and concise answers to any questions the user has.\n\n"
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
        # 1. Format the prompt with just the question (General AI mode)
        formatted_prompt = self.prompt.format(context="", input=user_question)
        
        # 2. Send the formatted prompt to the LLM via asynchronous network call
        response = await self.llm.ainvoke(formatted_prompt)
        
        # Return the final answer
        return {
            "answer": response.content,
            "sources_used": 0
        }

    async def astream_question(self, user_question: str):
        """
        Streams the answer token by token for an instant 'flash' UI experience.
        """
        formatted_prompt = self.prompt.format(context="", input=user_question)
        
        async for chunk in self.llm.astream(formatted_prompt):
            if chunk.content:
                yield chunk.content
