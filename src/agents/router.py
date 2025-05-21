from src.state.agent_state import AgentState

def on_topic_router(state: AgentState):
    on_topic = state.get("on_topic", "").strip().lower()

    if on_topic == "yes":
        return "retrieve"
    else:
        return "off_topic_reponse"
    
def proceed_router(state: AgentState):
    rephrase_count = state.get("rephrase_count", "")

    if state.get("proceed_to_generate", False):
        return "generate_answer"
    
    elif rephrase_count >= 2:
        return "cannot_answer"
    
    else:
        return "refine_question"