import PyPDF2

def get_file(file):
    content = ""

    if file.name.endswith(".txt"):
        # Read text file
        content = file.read().decode("utf-8")

    elif file.name.endswith(".pdf"):
        # Read PDF file using PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            content += page.extract_text()

    else:
        content = "Unsupported file format."

    return content