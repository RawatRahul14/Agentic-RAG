from src.state.agent_state import AgentState
from langchain_core.messages import AIMessage

def cannot_answer(state: AgentState):
    print("Entering cannot_answer")
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    state["messages"].append(
        AIMessage(
            content = "I'm sorry, but I cannot find the information you're looking for."
        )
    )

    return state

def off_topic_response(state: AgentState):
    print("Entering off_topic_response")
    
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    state["messages"].append(
        AIMessage(
            content = "I'm sorry! I cannot answer this question!"
        )
    )

    return state