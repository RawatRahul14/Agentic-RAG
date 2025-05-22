from src.state.agent_state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.schema.schemas import GradeQuestion

def question_classifier(state: AgentState):
    
    system_message = SystemMessage(
    content = """    
    You are a finance domain question classifier.

    Your job is to determine whether a user's question is related to **finance**. This includes direct or indirect references to:

    - Stock Market
    - Investment (Mutual funds, ETFs, SIPs, etc.)
    - Company financials or reports
    - Trading (intraday, long-term)
    - Financial Metrics (like PE ratio, ROI, EPS)
    - Cryptocurrency and blockchain finance
    - Budgeting, personal finance, or financial planning
    - Financial news or trends
    - Financial years, quarters, or earnings seasons

    ⚠️ Even if the question doesn’t explicitly mention “finance”, infer the financial context when terms like **“quarter”, “report”, “statement”, or “market”** are used and may refer to financial data in a document.

    For example:
    - “What are the quarters mentioned in the document?” → Yes ✅ (assume financial quarters in context of company reports)
    - “What quarters are taught in the curriculum?” → No ❌ (educational context)

    If the user's question is clearly or implicitly about any finance-related topic, respond with 'Yes'.
    Otherwise, respond with 'No'.
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