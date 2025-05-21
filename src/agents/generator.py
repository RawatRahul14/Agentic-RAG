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
    Answer the question based on the following context and the Chathistory. Especially take the latest into consideration:

    Chathistory: {history}

    Context: {context}

    Question: {question}
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