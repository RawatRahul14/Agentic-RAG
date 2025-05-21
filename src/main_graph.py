from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import StateGraph
from langgraph.graph import END

from src.state.agent_state import AgentState
from src.agents.rewriter import question_rewriter
from src.agents.classifier import question_classifier
from src.agents.no_answer import off_topic_response, cannot_answer
from src.agents.retriever import retrieve
from src.agents.doc_grader import retrieval_grader
from src.agents.generator import generate_answer
from src.agents.refiner import refine_question
from src.agents.router import on_topic_router, proceed_router


def build_agentic_rag_graph():
    """
    Builds and returns the compiled Agentic RAG LangGraph.
    """
    # Initialize the checkpointer (in-memory)
    checkpointer = MemorySaver()

    # Initialize the graph with state schema
    workflow = StateGraph(AgentState)

    # Register agent nodes
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("off_topic_response", off_topic_response)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("retrieval_grader", retrieval_grader)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("refine_question", refine_question)
    workflow.add_node("cannot_answer", cannot_answer)

    # Define edges between nodes
    workflow.add_edge("question_rewriter", "question_classifier")

    workflow.add_conditional_edges(
        "question_classifier",
        on_topic_router,
        {
            "retrieve": "retrieve",
            "off_topic_response": "off_topic_response"
        }
    )

    workflow.add_edge("retrieve", "retrieval_grader")

    workflow.add_conditional_edges(
        "retrieval_grader",
        proceed_router,
        {
            "generate_answer": "generate_answer",
            "refine_question": "refine_question",
            "cannot_answer": "cannot_answer"
        }
    )

    workflow.add_edge("refine_question", "retrieve")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("cannot_answer", END)
    workflow.add_edge("off_topic_response", END)

    # Set entry point
    workflow.set_entry_point("question_rewriter")

    # Compile and return the graph
    return workflow.compile(checkpointer = checkpointer)