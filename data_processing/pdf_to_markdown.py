import os
import pathlib
import pymupdf4llm

def convert_pdf_to_text(pdf_path, output_folder):
    """
    Convert a PDF file to a markdown text file.
    """
    import_doc = pymupdf4llm.to_markdown(pdf_path)
    base_name = os.path.basename(pdf_path).replace('.pdf', '').upper()
    output_file = os.path.join(output_folder, f"{base_name}.md")
    pathlib.Path(output_file).write_bytes(import_doc.encode())

def convert_all_pdfs(input_folder, output_folder):
    """
    Convert all PDF files in a folder to markdown text files.
    """
    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith('.pdf'):
            convert_pdf_to_text(os.path.join(input_folder, pdf_file), output_folder)
