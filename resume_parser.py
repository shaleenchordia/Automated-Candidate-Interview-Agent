import PyPDF2
import docx
import pdfplumber
import io

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF file"""
    try:
        # First try with PyPDF2
        pdf_file = io.BytesIO(file_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # If PyPDF2 fails to extract meaningful text, try pdfplumber
        if not text.strip():
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")

def parse_resume(file):
    """Parse resume file (PDF or DOCX) and extract text"""
    try:
        file_bytes = file.read()
        file_extension = file.name.lower().split('.')[-1]
        
        print(f"📄 Parsing {file.name} ({file_extension}) - {len(file_bytes)} bytes")
        
        extracted_text = ""
        
        if file_extension == 'pdf':
            extracted_text = extract_text_from_pdf(file_bytes)
        elif file_extension in ['doc', 'docx']:
            extracted_text = extract_text_from_docx(file_bytes)
        elif file_extension == 'txt':
            extracted_text = file_bytes.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        print(f"✅ Extracted {len(extracted_text)} characters from {file.name}")
        
        if not extracted_text.strip():
            print("⚠️ Warning: No text content extracted from file")
            return "No text content could be extracted from this file. Please ensure the file contains readable text or try a different format."
            
        return extracted_text
            
    except Exception as e:
        print(f"❌ Error parsing resume: {str(e)}")
        raise Exception(f"Error parsing resume: {str(e)}")
