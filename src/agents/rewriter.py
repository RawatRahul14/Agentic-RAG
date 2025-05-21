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
            content = "You are a helpful assistant that rephrases the user's question to be a"
                      "stand alone question optimized for retrieval."
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