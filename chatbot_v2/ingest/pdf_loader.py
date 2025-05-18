import os
import pathlib
import pymupdf4llm

def convert_pdf_to_markdown(pdf_path: str, output_folder: str) -> str:
    """
    Converts a single PDF to markdown and saves the file.
    """
    doc_text = pymupdf4llm.to_markdown(pdf_path)
    base_name = os.path.basename(pdf_path).replace('.pdf', '').upper()
    output_file = os.path.join(output_folder, f"{base_name}.md")
    pathlib.Path(output_file).write_bytes(doc_text.encode())
    return output_file

def convert_all_pdfs(input_folder: str, output_folder: str) -> list:
    """
    Converts all PDFs in a folder to markdown files.
    Returns list of output file paths.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_files = []
    for file in os.listdir(input_folder):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, file)
            output_files.append(convert_pdf_to_markdown(pdf_path, output_folder))
    return output_files
