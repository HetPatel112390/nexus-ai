import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()

if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY not found.")

SYSTEM_PROMPT = (
    "You are Nexus AI, a highly intelligent and helpful AI assistant — like Gemini or ChatGPT. "
    "RULE 1 — CONCISENESS: Answer directly. No fluff, no long preambles. Short question = short answer. "
    "RULE 2 — TYPO TOLERANCE: If the user misspells something (e.g. 'gutam' instead of 'gitam', 'hydrabad' instead of 'hyderabad'), "
    "intelligently infer the correct meaning and answer without complaining about the typo. "
    "RULE 3 — CONTEXT MEMORY: Remember everything from this conversation. If the user says 'what about CSE?' after asking about a college, "
    "you know exactly which college they mean. "
    "RULE 4 — FILES: If the user shares text extracted from a PDF or file, read it carefully and answer questions about it accurately. "
    "RULE 5 — FORMAT: Use markdown formatting (bold, lists, tables) when it improves clarity."
)


class EnterpriseRAG:
    def __init__(self):
        groq_api_key = os.environ.get("GROQ_API_KEY")
        # llama-3.3-70b-versatile: Massive 70B model, free on Groq, near GPT-4 quality
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=groq_api_key,
            max_tokens=2048
        )
        self.reset()

    def reset(self):
        """Clear conversation memory (called on New Chat)."""
        self.chat_history = [SystemMessage(content=SYSTEM_PROMPT)]

    async def ask_question(self, user_question: str) -> dict:
        """Non-streaming question answering."""
        self.chat_history.append(HumanMessage(content=user_question))
        response = await self.llm.ainvoke(self.chat_history)
        self.chat_history.append(AIMessage(content=response.content))
        self._trim_history()
        return {"answer": response.content, "sources_used": 0}

    async def astream_question(self, user_question: str):
        """Streaming question — yields token chunks one by one."""
        self.chat_history.append(HumanMessage(content=user_question))
        full_response = ""
        async for chunk in self.llm.astream(self.chat_history):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
        self.chat_history.append(AIMessage(content=full_response))
        self._trim_history()

    def _trim_history(self):
        """Keep last 30 messages + system prompt to prevent context overflow."""
        if len(self.chat_history) > 31:
            self.chat_history = [self.chat_history[0]] + self.chat_history[-30:]
