import pdfplumber

def extract_from_pdf(uploaded_file):
    texts = []
    tables = []

    # pdfplumber supports file-like byte objects
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            # Extract text
            text = page.extract_text()
            if text:
                texts.append(text)

            # Extract tables
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append(table)  # Or convert to DataFrame if needed

    return texts, tables