import pdfplumber
import re
import logging

logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text from a PDF resume.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
                else:
                    logging.warning(f"Page {page.page_number} in {pdf_path} contains no extractable text.")
    except Exception as e:
        logging.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        return ""

    if not text.strip():
        logging.warning(f"No text extracted from {pdf_path}. It may be an image-based PDF.")

    return text.strip()

def clean_resume_text(text):
    """
    Cleans extracted resume text by removing unnecessary whitespace and special characters.
    """
    original_length = len(text)
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[^\w\s.,]', '', text)  # Remove special characters
    cleaned_length = len(text)

    if cleaned_length < original_length * 0.2:
        logging.warning(f"Text cleaning reduced resume size significantly: {original_length} → {cleaned_length} characters.")

    return text.strip()

def parse_resume(pdf_path):
    """
    Parses a resume PDF and returns structured text.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_resume_text(raw_text)

    if not cleaned_text:
        logging.warning(f"Parsed resume from {pdf_path} is empty after cleaning.")

    return cleaned_text
