from src.state.agent_state import AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

def question_rewriter(state: AgentState):

    # Reset state variables except "question" and "messages", as with each question new flow will start
    state["documents"] = []
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    # For the first user query
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    # If the user inputs a question
    if len(state["messages"]) > 1:

        # Extract prior messages as conversation history excluding the latest one
        conversation = state["messages"][:-1]

        # Current Question
        current_question = state["question"]

        messages = [SystemMessage(
        content = """
        You are a helpful assistant that rewrites user questions about financial documents into clear, standalone questions optimized for document retrieval.

        Context:
        - The user has uploaded one or more financial documents (e.g., quarterly reports, earnings statements, balance sheets).
        - They may ask follow-up or vague questions like “what about this quarter?” or “how much did they earn?” — your job is to rephrase those into complete, unambiguous questions using prior conversation history.

        Instructions:
        - Give the highest importance to the latest question asked by the user.
        - Make the question standalone by including any missing context (e.g., subject, time period, metric).
        - Assume references to "this quarter", "they", "it", or "the report" relate to the uploaded financial document.
        - Avoid changing the user's intent or adding new information.
        - Be concise, specific, and neutral in tone.

        Examples:
        - User: “What about Q2?” → Rephrased: “What were the financial results for Q2 in the uploaded report?”
        - User: “How much profit did they make?” → Rephrased: “What was the net income reported in the uploaded financial document?”
        - User: “And the margins?” → Rephrased: “What were the operating margins reported in the uploaded financial document?”

        Your output should be a single, well-formed question suitable for querying a document retriever.
        """
        )]

        # Adding the whole lists of Human an AI interactions
        messages.extend(conversation)

        # Adding the latest question
        messages.append(HumanMessage(content = current_question.content))

        llm = ChatOpenAI(model = "gpt-4o-mini")
        response = llm.invoke(messages)
        better_question = response.content.strip()
        state["rephrased_question"] = better_question

    # If the user asks the first question
    else:
        state["rephrased_question"] = state["question"].content

    return state