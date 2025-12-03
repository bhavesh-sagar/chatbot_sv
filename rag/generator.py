import os
from langchain_groq import ChatGroq

class Generator:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2
        )

    def answer(self, question, context):
        prompt = f"Context: {context}\n\nQuestion: {question}"
        res = self.llm.invoke(prompt)
        return res.content
