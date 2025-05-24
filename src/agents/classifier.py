from src.state.agent_state import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.schema.schemas import GradeQuestion

def question_classifier(state: AgentState):
    
    system_message = SystemMessage(
    content = """    
    You are a finance domain question classifier for a fintech chatbot.

    The user uploads financial documents (such as earnings reports, quarterly filings, or investor statements), and then asks questions based on them. Your job is to determine if the user's question is related to **finance** in the context of these uploaded files.

    You should classify the question as 'Yes' if it is related to:

    - Stock market activity
    - Investment instruments (mutual funds, SIPs, ETFs, stocks)
    - Company financials, earnings, or balance sheets
    - Trading (day trading, long-term, options, etc.)
    - Financial metrics (e.g., EPS, PE ratio, ROI, revenue, profit, margins)
    - Budgeting, financial planning, or personal finance
    - Cryptocurrency or blockchain-based finance
    - Financial statements, annual or quarterly reports
    - Financial periods (quarters, fiscal years), even if mentioned indirectly
    - Any references to values or trends found in the uploaded documents

    ⚠️ Important: Since the user's question is always asked **after uploading a financial PDF**, terms like “quarter”, “year”, “report”, or “statement” should be interpreted as **finance-related by default**, unless the context clearly proves otherwise.

    Examples:
    - “What quarters are mentioned in the file?” → Yes ✅ (quarter = financial quarter)
    - “How much profit did the company make?” → Yes ✅
    - “When does the semester start?” → No ❌ (educational context)
    - “What topics are covered in Q3 training?” → No ❌ (training context)

    Return 'Yes' if the question is financially relevant or refers to anything in the uploaded PDF in a financial context. Otherwise, return 'No'.
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