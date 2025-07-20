import spacy
import nltk
from typing import Dict

class ResumeAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def analyze(self, resume_text: str) -> Dict:
        """
        Analyze the resume content and extract relevant information
        """
        doc = self.nlp(resume_text)
        
        # Extract key information
        analysis = {
            "skills": self._extract_skills(doc),
            "experience": self._extract_experience(doc),
            "education": self._extract_education(doc),
            "key_terms": self._extract_key_terms(doc)
        }
        
        return analysis
    
    def _extract_skills(self, doc):
        # Implement skill extraction logic
        # This is a simplified version - you would want to enhance this
        skills = []
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "GPE"]:
                skills.append(ent.text)
        return list(set(skills))
    
    def _extract_experience(self, doc):
        # Implement experience extraction logic
        experience = []
        # Add your logic here
        return experience
    
    def _extract_education(self, doc):
        # Implement education extraction logic
        education = []
        # Add your logic here
        return education
    
    def _extract_key_terms(self, doc):
        # Extract important keywords
        key_terms = []
        for token in doc:
            if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "PROPN"]:
                key_terms.append(token.text)
        return list(set(key_terms))
