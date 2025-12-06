import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from chat_memory import PostgresChatMessageHistory
from retriever import retrieve_similar_docs  # <-- optional if using RAG

load_dotenv()

# --- MODEL ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    convert_system_message_to_human=True
)

# --- SESSION HISTORY LOADER ---
def get_session_history(session_id: str):
    return PostgresChatMessageHistory(session_id=session_id)

# --- PROMPT WITH MEMORY ---
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support agent. "
            "Answer ONLY based on policy and previous chat context."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

# Attach memory to chain
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)

# --- CHAT LOOP ---
def chat_loop(session_id="default_user"):
    print("Customer Support Chatbot (type 'exit' to quit)")
    print("Session:", session_id)

    config = {"configurable": {"session_id": session_id}}

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # OPTIONAL: retrieve RAG context
        rag_docs = retrieve_similar_docs(user_input, top_k=3)
        rag_context = "\n\n".join([d["content"] for d in rag_docs])

        # Inject RAG + user message into conversation
        messages = [
            HumanMessage(
                content=f"Context:\n{rag_context}\n\nUser Query: {user_input}"
            )
        ]

        response = memory_chain.invoke(
            {"messages": messages},
            config=config
        )

        print(f"Bot: {response.content}\n")

# --- RUN ---
if __name__ == "__main__":
    chat_loop("session123")
