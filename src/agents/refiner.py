from src.state.agent_state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

def refine_question(state: AgentState):

    rephrase_count = state.get("rephrase_count", 0)

    if rephrase_count >= 2:
        return state
    
    question_to_refine = state["rephrased_question"]

    system_message = SystemMessage(
        content = """
        You are a helpful assistant that slightly refines the user's question to improve the retrieval process.
        Provide a slightly adjusted version of the question to improve the retrieval process.
        """
    )

    human_message = HumanMessage(
        content = f"Original question: {question_to_refine} \n\nProvide a slightly refined question."
    )

    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    llm = ChatOpenAI(model = "gpt-4o-mini")

    prompt = refine_prompt.format()

    response = llm.invoke(prompt)

    refined_question = response.content.strip()

    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1

    return state