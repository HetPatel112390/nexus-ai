import os
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# New Super-Fast Free Models
from langchain_groq import ChatGroq

# Load environment variables (API keys) from our .env file
load_dotenv()

# We need an API key to talk to Groq's supercomputers.
if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("CRITICAL ERROR: GROQ_API_KEY not found in .env file. Please add it.")

class EnterpriseRAG:
    def __init__(self, data_path=None):
        """
        Constructor: Initializes the AI model and memory.
        """
        print("[AI ENGINE] Initializing General AI System with Memory...")
        
        # We use Groq's insanely smart Llama-3.3 70B model (Free & incredibly capable)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, groq_api_key=groq_api_key)

        # 2. BUILD THE SYSTEM PROMPT & MEMORY
        # We set extremely strict rules for conciseness and typo correction.
        system_prompt = (
            "You are Nexus AI, a highly intelligent and helpful AI assistant. "
            "CRITICAL INSTRUCTION 1: You MUST be concise. Answer the user's question directly without unnecessary fluff, long introductions, or over-explaining. "
            "CRITICAL INSTRUCTION 2: Be extremely forgiving with typos, spelling mistakes, or bad grammar. If the user misspells a college (like 'gutam' instead of 'gitam'), a city, or any topic, intelligently figure out what they mean and answer the question without complaining that you don't recognize the typo. "
            "Remember previous messages in the conversation to provide accurate context."
        )
        self.chat_history = [SystemMessage(content=system_prompt)]
        
        print("[AI ENGINE] System Ready.")

    async def ask_question(self, user_question: str) -> dict:
        """
        Public method to query the system asynchronously (Non-streaming).
        """
        self.chat_history.append(HumanMessage(content=user_question))
        
        response = await self.llm.ainvoke(self.chat_history)
        
        self.chat_history.append(AIMessage(content=response.content))
        
        # Return the final answer
        return {
            "answer": response.content,
            "sources_used": 0
        }

    async def astream_question(self, user_question: str):
        """
        Streams the answer token by token and remembers the conversation.
        """
        self.chat_history.append(HumanMessage(content=user_question))
        
        full_response = ""
        async for chunk in self.llm.astream(self.chat_history):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
                
        self.chat_history.append(AIMessage(content=full_response))
        
        # Prevent memory from growing infinitely (keep last 20 messages + system prompt)
        if len(self.chat_history) > 21:
            self.chat_history = [self.chat_history[0]] + self.chat_history[-20:]
