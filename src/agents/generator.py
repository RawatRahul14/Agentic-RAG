from src.agents.classifier import AgentState
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

def generate_llm():
    llm = ChatOpenAI(model = "gpt-4o-mini")

    template = """
    You are a financial assistant. Use the following context and chat history to answer the user's question accurately.

    - Always prioritize the **most recent** part of the chat history.
    - Do **not** use bold, italics, underlining, or any other formatting styles.
    - Use bullet points only when necessary for clarity (e.g., lists of values, steps, or comparisons).
    - If the answer cannot be determined from the given context, say so clearly.

    Chathistory:
    {history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = prompt | llm

    return rag_chain

def generate_answer(state: AgentState):

    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer.")
    
    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    rag_chain = generate_llm()

    response = rag_chain.invoke(
        {"history": history,
         "context": documents,
         "question": rephrased_question}
    )

    generation = response.content.strip()

    state["messages"].append(AIMessage(content = generation))
    print(f"generate_answer: Generated response: {generation}")

    return state