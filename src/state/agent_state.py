from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    retriever: BaseRetriever
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage
    final_answer: str