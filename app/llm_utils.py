import os
import json
import requests
from .vectorstore import create_vector_store
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")  # Use .env or set manually

def call_llm(prompt):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "BylawAssistant"
            },
            data=json.dumps({
                "model": "deepseek/deepseek-prover-v2:free",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant trained on university academic bylaws."},
                    {"role": "user", "content": prompt}
                ]
            })
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ LLM Error: {str(e)}"

def process_query(user_query):
    try:
        vector_store = create_vector_store(r"E:\chatbot\chatbot\bylaw")
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        docs = retriever.get_relevant_documents(f"query: {user_query}")
        if not docs:
            return "⚠️ No relevant documents found."
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"
        return call_llm(prompt)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
