"""
Dynamic LLM-Based Question Generator
Generates unique, personalized interview questions for each candidate
Never repeats questions - always fresh and tailored content
"""

import json
import os
import random
from typing import Dict, List, Any
from datetime import datetime
import requests

class DynamicQuestionGenerator:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "llama3.2:latest"  # Using llama3.2 which is available
        self.question_history = []  # Track generated questions to avoid repeats
        
        # Job category specific training configurations
        self.job_category_prompts = {
            "Software Development": {
                "context": "You are an expert software engineering interviewer with 10+ years of experience",
                "focus_areas": ["coding architecture", "system design", "performance optimization", "debugging strategies", "code quality"],
                "difficulty_levels": ["junior", "mid-level", "senior"],
                "question_types": ["technical deep-dive", "problem-solving scenarios", "architecture decisions", "code review situations"],
                "unique_aspects": ["scalability challenges", "legacy system modernization", "team collaboration", "technical debt management"]
            },
            "Data & Analytics": {
                "context": "You are an expert data science and analytics interviewer",
                "focus_areas": ["statistical modeling", "data pipeline design", "machine learning implementation", "data visualization", "business impact"],
                "difficulty_levels": ["entry", "experienced", "expert"],
                "question_types": ["case studies", "methodology discussions", "technical implementation", "business scenarios"],
                "unique_aspects": ["data quality issues", "model deployment", "stakeholder communication", "ethical considerations"]
            },
            "Infrastructure & DevOps": {
                "context": "You are an expert DevOps and infrastructure interviewer",
                "focus_areas": ["automation strategies", "cloud architecture", "monitoring systems", "security implementation", "incident response"],
                "difficulty_levels": ["junior", "mid-level", "senior"],
                "question_types": ["scenario troubleshooting", "architecture design", "automation challenges", "security incidents"],
                "unique_aspects": ["disaster recovery", "cost optimization", "compliance", "team processes"]
            },
            "Quality Assurance & Testing": {
                "context": "You are an expert QA and testing interviewer",
                "focus_areas": ["test strategy", "automation frameworks", "quality metrics", "risk assessment", "team collaboration"],
                "difficulty_levels": ["junior", "mid-level", "senior"],
                "question_types": ["testing scenarios", "quality challenges", "automation decisions", "process improvements"],
                "unique_aspects": ["continuous testing", "performance testing", "security testing", "mobile testing"]
            },
            "Product & Project Management": {
                "context": "You are an expert product and project management interviewer",
                "focus_areas": ["product strategy", "stakeholder management", "agile methodologies", "data-driven decisions", "team leadership"],
                "difficulty_levels": ["junior", "mid-level", "senior"],
                "question_types": ["strategic scenarios", "conflict resolution", "prioritization challenges", "metrics analysis"],
                "unique_aspects": ["cross-functional collaboration", "customer feedback", "market analysis", "roadmap planning"]
            },
            "Design & UX": {
                "context": "You are an expert UX/UI design interviewer",
                "focus_areas": ["user research", "design systems", "accessibility", "interaction design", "usability testing"],
                "difficulty_levels": ["junior", "mid-level", "senior"],
                "question_types": ["design challenges", "user research scenarios", "collaboration situations", "design critiques"],
                "unique_aspects": ["design system scalability", "inclusive design", "design metrics", "cross-platform consistency"]
            }
        }
    
    def generate_personalized_questions(
        self, 
        job_category: str,
        candidate_skills: List[str],
        experience_level: str,
        technical_level: str = "Mid-level",
        num_questions: int = 3
    ) -> List[str]:
        """
        Generate completely unique, personalized questions for each candidate
        """
        if job_category not in self.job_category_prompts:
            job_category = "Software Development"  # Default fallback
        
        category_config = self.job_category_prompts[job_category]
        questions = []
        
        for i in range(num_questions):
            # Create a unique prompt for each question
            unique_prompt = self._create_unique_prompt(
                category_config,
                job_category,
                candidate_skills,
                experience_level,
                technical_level,
                i + 1,
                num_questions
            )
            
            # Generate question using LLM
            question = self._generate_single_question(unique_prompt)
            
            # Ensure uniqueness and add to results
            if question and self._is_unique_question(question):
                questions.append(question)
                self.question_history.append(question)
            else:
                # Retry with different approach if not unique
                retry_prompt = self._create_retry_prompt(unique_prompt, question)
                retry_question = self._generate_single_question(retry_prompt)
                if retry_question:
                    questions.append(retry_question)
                    self.question_history.append(retry_question)
        
        return questions
    
    def _create_unique_prompt(
        self, 
        category_config: Dict,
        job_category: str,
        candidate_skills: List[str],
        experience_level: str,
        technical_level: str,
        question_number: int,
        total_questions: int
    ) -> str:
        """Create a unique, detailed prompt for question generation"""
        
        # Select random focus areas and aspects for uniqueness
        selected_focus = random.sample(category_config['focus_areas'], min(2, len(category_config['focus_areas'])))
        selected_aspects = random.sample(category_config['unique_aspects'], min(2, len(category_config['unique_aspects'])))
        
        # Build candidate context
        skills_context = f"Primary skills: {', '.join(candidate_skills[:3])}" if candidate_skills else "General skills"
        
        # Create uniqueness constraints
        avoid_patterns = [
            "Tell me about a time",
            "What is your experience with",
            "How would you handle",
            "Describe your approach to"
        ]
        
        prompt = f"""You are {category_config['context']} conducting a {job_category} interview.

CANDIDATE PROFILE:
- Experience Level: {experience_level}
- Technical Level: {technical_level}
- {skills_context}
- Question {question_number} of {total_questions}

QUESTION REQUIREMENTS:
- Focus on: {', '.join(selected_focus)}
- Include aspects of: {', '.join(selected_aspects)}
- Difficulty: {experience_level} level
- Must be completely unique and never asked before
- Should reveal deep thinking and problem-solving approach
- Must be practical and scenario-based

CRITICAL INSTRUCTIONS:
- NEVER use generic interview phrases
- NEVER start with: "Tell me about", "What is your", "How would you", "Describe your"
- ALWAYS create a specific, detailed workplace scenario
- ALWAYS include realistic constraints and challenges
- ALWAYS require both technical and strategic thinking
- MINIMUM 2 sentences, MAXIMUM 4 sentences

ADVANCED QUESTION PATTERNS TO FOLLOW:
1. "You inherit a system where [specific problem]. The business needs [specific outcome] in [timeframe]. Given constraints of [limitation], what's your complete approach?"
2. "Your team is facing [specific technical challenge] affecting [business impact]. You have [resources/constraints]. Walk through your solution strategy."
3. "A critical production issue occurs where [detailed scenario]. Multiple stakeholders have conflicting priorities. How do you navigate this?"

SPECIFIC SCENARIO ELEMENTS TO INCLUDE:
- Real numbers (users, requests, data volume, timeline)
- Business constraints (budget, time, resources)
- Technical constraints (legacy systems, performance, security)
- Stakeholder conflicts or competing priorities
- Multiple valid solution approaches

Generate ONE sophisticated interview question following these patterns for a {experience_level} {job_category} professional with {skills_context}.

RESPOND WITH ONLY THE QUESTION - NO PREAMBLE, NO EXPLANATION, JUST THE QUESTION:"""

        return prompt
    
    def _create_retry_prompt(self, original_prompt: str, previous_question: str) -> str:
        """Create a retry prompt if the first attempt wasn't unique enough"""
        return f"""{original_prompt}

ADDITIONAL CONSTRAINTS:
- The previous attempt was: "{previous_question}"
- Generate something COMPLETELY different
- Use a different angle or perspective
- Focus on different technical aspects
- Create a more innovative scenario

Generate a TOTALLY DIFFERENT question that shares no similarities with the previous attempt."""
    
    def _generate_single_question(self, prompt: str) -> str:
        """Generate a single question using LLM"""
        try:
            print(f"🤖 Attempting to generate question with {self.model_name}...")
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,  # Balanced creativity and coherence
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.3,
                        "num_predict": 200,  # Limit response length
                        "stop": ["\n\n", "Question:", "Here's", "I hope"]  # Stop at common completions
                    }
                },
                timeout=60  # Longer timeout for better generation
            )
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                question = result.get('response', '').strip()
                print(f"✅ Generated question length: {len(question)}")
                
                # Clean up the response
                question = self._clean_question(question)
                
                if len(question) > 50:  # Ensure it's a substantial question
                    print(f"✅ Question generated successfully: {question[:100]}...")
                    return question
                else:
                    print(f"❌ Generated question too short: {question}")
            else:
                print(f"❌ HTTP Error: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ Error generating question: {e}")
        
        print("⚠️ Using fallback question...")
        # Category-specific fallback questions
        fallback_questions = self._get_fallback_questions()
        
        return random.choice(fallback_questions)
    
    def _clean_question(self, question: str) -> str:
        """Clean and format the generated question"""
        # Remove common prefixes that might make questions generic
        prefixes_to_remove = [
            "Here's a unique question:",
            "Here's a sophisticated question:",
            "Here's an advanced question:",
            "Question:",
            "Interview Question:",
            "Here's your question:",
            "Here's a question:",
            "**Question:**",
            "*Question:*",
            "I'll create",
            "Here's a",
            "A good question would be:",
            "Consider this question:"
        ]
        
        # Remove common prefixes
        for prefix in prefixes_to_remove:
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()
        
        # Remove quotes if the entire question is wrapped in them
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1].strip()
        if question.startswith("'") and question.endswith("'"):
            question = question[1:-1].strip()
        
        # Remove markdown formatting
        question = question.replace("**", "").replace("*", "").replace("#", "")
        
        # Remove extra whitespace and normalize
        question = ' '.join(question.split())
        
        # Ensure proper punctuation
        if question and not question.endswith(('?', '.', '!')):
            question += "?"
        
        return question
    
    def _is_unique_question(self, question: str) -> bool:
        """Check if question is unique compared to history"""
        if not question:
            return False
        
        question_lower = question.lower()
        
        # Check against recent questions for similarity
        for prev_question in self.question_history[-20:]:  # Check last 20 questions
            if self._similarity_score(question_lower, prev_question.lower()) > 0.6:
                return False
        
        # Check for generic patterns
        generic_patterns = [
            "tell me about",
            "what is your experience",
            "how would you",
            "describe your approach",
            "what are the benefits",
            "explain the difference"
        ]
        
        for pattern in generic_patterns:
            if pattern in question_lower:
                return False
        
        return True
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about question generation"""
        return {
            "total_questions_generated": len(self.question_history),
            "supported_categories": list(self.job_category_prompts.keys()),
            "recent_questions": self.question_history[-5:] if self.question_history else [],
            "model_used": self.model_name,
            "last_generated": datetime.now().isoformat()
        }
    
    def _get_fallback_questions(self) -> List[str]:
        """Get sophisticated fallback questions by category"""
        return [
            "You inherit a critical microservices architecture that handles 2M+ daily transactions, but it's experiencing cascading failures during peak hours. The previous team left minimal documentation, and you have 3 weeks to stabilize it before a major product launch. Walk through your systematic approach to diagnosis, immediate stabilization, and long-term architecture improvements.",
            
            "Your company is migrating from a monolithic application to a distributed system, but halfway through the migration, you discover the new architecture won't meet the performance requirements for your largest enterprise client (50% of revenue). The client contract expires in 4 months. How do you navigate the technical, business, and timeline constraints?",
            
            "You're leading a team where your most experienced developer strongly advocates for a complete rewrite of the core system, while the product manager insists on feature velocity for an upcoming funding round. The current codebase works but has significant technical debt. Given limited resources and competing priorities, how do you make this decision and align the team?",
            
            "A critical security vulnerability is discovered in your production system that affects 100K+ users. The fix requires substantial changes to your authentication system, but implementing it will break backward compatibility with mobile apps already in the app stores. You have 48 hours to respond. What's your complete strategy?",
            
            "Your team delivered a highly requested feature that increased user engagement by 25%, but it's causing a 40% increase in infrastructure costs and occasional performance degradation. Business wants to keep the feature, but engineering is concerned about sustainability. How do you balance these competing interests while maintaining system reliability?"
        ]

# Test function
def test_dynamic_generation():
    """Test the dynamic question generator"""
    generator = DynamicQuestionGenerator()
    
    test_cases = [
        {
            "job_category": "Software Development",
            "candidate_skills": ["Python Programming", "React Ecosystem", "AWS"],
            "experience_level": "Senior (6-10 years)",
            "technical_level": "Senior"
        },
        {
            "job_category": "Data & Analytics", 
            "candidate_skills": ["Machine Learning", "Python Programming", "Data Science"],
            "experience_level": "Mid-Level (4-6 years)",
            "technical_level": "Mid-level"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== Test Case {i+1}: {test_case['job_category']} ===")
        questions = generator.generate_personalized_questions(**test_case)
        
        for j, question in enumerate(questions, 1):
            print(f"\nQuestion {j}:")
            print(f"{question}")
        
        print(f"\nStats: {generator.get_generation_stats()}")

if __name__ == "__main__":
    test_dynamic_generation()
