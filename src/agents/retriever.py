from src.state.agent_state import AgentState

def retrieve(state: AgentState) -> AgentState:
    retriever = state["retriever"]
    query = state["rephrased_question"]
    
    # Correct method for retriever
    documents = retriever.invoke(query)

    state["documents"] = documents
    return state