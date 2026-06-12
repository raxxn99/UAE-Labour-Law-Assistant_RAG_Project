import PyPDF2
import re
import os


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""

    text = ''

    try:
        with open(pdf_path, 'rb') as file:

            pdf_reader = PyPDF2.PdfReader(file)

            for page in pdf_reader.pages:

                extracted = page.extract_text()

                if extracted:
                    text += extracted + '\n'

    except Exception as e:
        print(f'Error reading {pdf_path}: {e}')

    return text


def clean_text(text):
    """Clean extracted text"""

    # Remove page numbers like - 1 -
    text = re.sub(r'\-\s*\d+\s*\-', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove extra spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove strange unicode artifacts
    text = text.replace('\x00', ' ')

    return text.strip()


def process_pdf_files(input_dir, output_dir):
    """Process all PDF files including subfolders"""

    os.makedirs(output_dir, exist_ok=True)

    # Walk through all folders and subfolders
    for root, dirs, files in os.walk(input_dir):

        for pdf_file in files:

            if pdf_file.endswith('.pdf'):

                pdf_path = os.path.join(root, pdf_file)

                print(f'Processing {pdf_file}...')

                # Extract text
                raw_text = extract_text_from_pdf(pdf_path)

                # Clean text
                cleaned_text = clean_text(raw_text)

                # Output filename
                output_filename = pdf_file.replace('.pdf', '.txt')

                output_file = os.path.join(
                    output_dir,
                    output_filename
                )

                # Save cleaned text
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)

                print(f'Saved to {output_file}')


if __name__ == '__main__':

    process_pdf_files(
        'data/raw_data',
        'data/processed_data'
    )