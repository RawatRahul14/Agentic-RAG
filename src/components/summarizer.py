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
    You are a financial analyst assistant.

    You will be given a raw paragraph from a financial press release. Your task is to extract and summarize all key financial metrics and performance insights into a clear and concise paragraph, suitable for inclusion in a fintech chatbot’s response.

    Instructions:
    - Accurately extract all numerical values: revenue, operating income, margins, net income, and EPS.
    - Always mention which period the value corresponds to (e.g., Q4 2023, full year 2023).
    - Include percentage changes (e.g., year-over-year growth) if mentioned.
    - Report trends (e.g., increases, decreases, or stability) clearly.
    - Briefly include any management commentary if it provides meaningful context (e.g., focus on cost-cutting or AI).
    - Ignore corporate PR or fluff (e.g., “the best is yet to come”).
    - Do NOT fabricate or assume any missing values.
    - Keep the output as a single, professional paragraph (no bullet points or section headers).
    - The tone should be factual, formal, and precise.

    Now, summarize the following paragraph:
    {element}
    """
    table_prompt = ChatPromptTemplate.from_template(table_prompt_text)

    # === Prompt for Texts ===
    text_prompt_text = """   
    You are a financial analyst assistant.

    You will be given a structured financial data table that includes key metrics such as revenues, operating income, margins, net income, other income/expenses, and earnings per share (EPS) for both a specific quarter and the full fiscal year.

    Your task is to generate a clear, concise, and professional paragraph summarizing the financial performance. Follow these guidelines:

    Instructions:
    - Extract all numerical values accurately and clearly distinguish between **quarterly (e.g., Q4 2023)** and **full-year (e.g., FY 2023)** metrics.
    - Clearly mention the year-over-year percentage changes if they are available.
    - Group and summarize related metrics in order:
    1. **Revenues**
    2. **Operating Income and Margin**
    3. **Other Income/Expenses**
    4. **Net Income and EPS**
    5. **Make sure you are consistent with formatting** if any
    - If a metric shows notable growth, decline, or consistency, describe the trend factually.
    - Do not include any information that is not explicitly present in the table.
    - Avoid overly promotional or speculative language.
    - The output must be a **single, continuous paragraph**, written in a professional and factual tone (no bullet points or section headers).

    Here is the table:
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