from pathlib import Path
import pymupdf4llm

def convert_pdf_to_text(pdf_path, output_folder):
    """
    Convert a single PDF file to a markdown (.md) text file.

    Args:
        pdf_path (str or Path): Path to the input PDF file.
        output_folder (str or Path): Path to the output folder where the markdown file will be saved.
    """
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    try:
        markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
        output_file = output_folder / f"{pdf_path.stem.upper()}.md"
        output_file.write_text(markdown_text, encoding='utf-8')
        print(f"Converted: {pdf_path.name} -> {output_file.name}")
    except Exception as e:
        print(f"Failed to convert {pdf_path.name}: {e}")

def convert_all_pdfs(input_folder, output_folder):
    """
    Convert all PDF files in a folder to markdown text files.

    Args:
        input_folder (str or Path): Folder containing input PDF files.
        output_folder (str or Path): Folder to save the converted markdown files.
    """
    input_folder = Path(input_folder)
    for pdf_file in input_folder.glob("*.pdf"):
        convert_pdf_to_text(pdf_file, output_folder)
