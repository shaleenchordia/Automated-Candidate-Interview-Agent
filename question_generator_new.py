from typing import Dict, List
import json
from pymongo import MongoClient
import requests
import re
from datetime import datetime
from ..config import MONGODB_URI, DATABASE_NAME

class QuestionGenerator:
    def __init__(self):
        # Connect to MongoDB
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[DATABASE_NAME]
        self.questions_collection = self.db["generated_questions"]
        self.ollama_url = "http://localhost:11434/api/generate"

    def generate(self, job_description: str) -> List[Dict[str, str]]:
        """
        Generate interview questions using Ollama based on job description.
        Returns a list of questions with their categories.
        """
        if not job_description or len(job_description.strip()) < 20:
            raise ValueError("Please provide a detailed job description")

        prompt = f"""You are an expert technical interviewer. Given this job description:
{job_description}

Generate exactly 3 unique and challenging interview questions based on this job description.

Requirements:
1. One technical question to test core skills and knowledge
2. One behavioral question to evaluate past experience and soft skills
3. One problem-solving question based on real-world scenarios

Format each question exactly as:
1. [Technical] your_question
2. [Behavioral] your_question
3. [Problem-solving] your_question

Guidelines:
- Questions should be highly specific to the role and required skills
- Make questions challenging but clear
- Focus on key requirements from the job description
- Ensure questions test both theoretical knowledge and practical experience"""

        try:
            # Generate questions using Ollama
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                }
            )
            response.raise_for_status()
            
            # Parse the response
            generated_text = response.json()["response"]
            
            # Extract questions and their categories
            questions = []
            for line in generated_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                    # Extract category and question
                    category_match = re.search(r'\[(.*?)\]', line)
                    if category_match:
                        category = category_match.group(1)
                        question = re.sub(r'^\d+\.\s*\[.*?\]\s*', '', line).strip()
                        if question:
                            questions.append({
                                "category": category,
                                "question": question
                            })
            
            # Store questions in MongoDB
            if questions:
                self.questions_collection.insert_one({
                    "job_description": job_description,
                    "questions": questions,
                    "timestamp": datetime.utcnow()
                })
            
            return questions
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []
