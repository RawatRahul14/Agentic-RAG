from src.state.agent_state import AgentState
from langchain_openai import ChatOpenAI
from src.schema.schemas import GradeDocument
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

def retrieval_grader(state: AgentState):

    system_message = """
    You are a grader assessing the relevance of a retrieved document to a user question.
    Only answer with 'Yes' or 'No'

    If the document contains the information relevant to the user's question, respond with 'Yes'.
    Otherwise, respond with 'No'.
    """

    llm = ChatOpenAI(model = "gpt-4o-mini")
    structured_output = llm.with_structured_output(GradeDocument)

    relevant_docs = []

    for doc in state["documents"]:
        human_message = HumanMessage(
            content = f"User question: {state['rephrased_question']} \n\nRetrieved document: \n{doc.page_content}"
        )

        grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        grader_llm = grade_prompt | structured_output

        result = grader_llm.invoke({})

        if result.score.strip().lower() == "yes":
            relevant_docs.append(doc)

    state["documents"] = relevant_docs
    state["proceed_to_generate"] = len(relevant_docs) > 0

    return state