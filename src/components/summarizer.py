from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

def get_summary(texts: list, tables: list):
    """
    Generates paragraph-style summaries for financial tables and narrative financial texts.

    Args:
        tables (list): List of tables (each as a string or joined list of rows).
        texts (list): List of plain financial text descriptions.

    Returns:
        tuple: (table_summaries, text_summaries) as lists of paragraph strings
    """

    # === Prompt for Tables ===
    table_prompt_text = """   
    Generate a concise and accurate financial summary of a company's performance in a single paragraph based on the provided table data. \
    Each row of the table should be individually analyzed, ensuring that all relevant values and trends are considered. The summary should report precise values and trends directly from the data, emphasizing critical metrics such as revenue, profit, growth trends, significant changes, and anomalies. \
    It must use clear, professional language without jargon, focusing on key aspects like profitability, cash flow, debt levels, and performance ratios. Assumptions or estimates should not be included. \
    The paragraph should also highlight trends or insights relevant for predictive or comparative queries, such as growth expectations, decline patterns, or industry benchmarks.

    Table Data:
    {element}
    """
    table_prompt = ChatPromptTemplate.from_template(table_prompt_text)

    # === Prompt for Texts ===
    text_prompt_text = """   
    Based on the following financial discussion or text data, generate a concise single-paragraph summary of the company’s performance. \
    Carefully extract and report any key metrics such as revenue, net income, margins, growth rates, and financial ratios mentioned. Focus on trends or major changes in financial health, profitability, or strategic shifts. \
    Use formal and professional tone, and do not add your own interpretations or assumptions.

    Text Data:
    {element}
    """
    text_prompt = ChatPromptTemplate.from_template(text_prompt_text)

    # === Model and Output Parser ===
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    # === Chains ===
    table_chain = {"element": lambda x: x} | table_prompt | model | parser
    text_chain = {"element": lambda x: x} | text_prompt | model | parser

    # === Batch Process with Concurrency ===
    table_summaries = table_chain.batch(tables, config={"max_concurrency": 5})
    text_summaries = text_chain.batch(texts, config={"max_concurrency": 5})

    return text_summaries, table_summaries