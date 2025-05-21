from src.state.agent_state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.schema.schemas import GradeQuestion

def question_classifier(state: AgentState):
    
    system_message = SystemMessage(
        content = """    
        You are a finance domain question classifier.

        Determine whether the user's question is related to **finance**. This includes topics such as:

        - Stock Market
        - Investment (Mututal funds, ETFs, SIPs, etc.)
        - Company financials or reports
        - Trading (intraday, long-term)
        - Financial Metrics (like PE ratio, ROI, EPS)
        - Cryptocurrency and blockchain finance
        - Budgeting, personal finance, or financial planning
        - Financial news or trends

        If the user's question is clearly about any finance-related topic, respond with 'Yes'.
        Otherwise, reponsd with 'No'.
        """
    )

    human_message = HumanMessage(
        content = f"User question: {state['rephrased_question']}"
    )

    prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = ChatOpenAI(model = "gpt-4o-mini")

    structure_llm = llm.with_structured_output(GradeQuestion)

    pipeline = prompt | structure_llm

    result = pipeline.invoke({})

    state["on_topic"] = result.score.strip()

    return state