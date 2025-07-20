from pymongo import MongoClient
from datetime import datetime
from typing import Optional, Dict, List
from bson import ObjectId
from ..config import MONGODB_URI, DATABASE_NAME

class CandidateService:
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[DATABASE_NAME]
        self.candidates_collection = self.db["candidates"]
    
    def create_candidate_profile(self, name: str, job_profile: dict) -> str:
        """Create a new candidate profile in the database"""
        candidate_data = {
            "name": name,
            "job_profile": job_profile,
            "created_at": datetime.utcnow(),
            "status": "profile_created",
            "interview_completed": False,
            "quiz_score": None,
            "total_questions": None
        }
        
        result = self.candidates_collection.insert_one(candidate_data)
        return str(result.inserted_id)
    
    def update_candidate_score(self, candidate_id: str, quiz_score: float, total_questions: int, 
                             detailed_scores: Optional[Dict] = None, responses: Optional[List] = None,
                             skill_summary: Optional[Dict] = None):
        """Update candidate's quiz score and detailed information"""
        
        update_data = {
            "quiz_score": quiz_score,
            "total_questions": total_questions,
            "interview_completed": True,
            "status": "interview_completed",
            "completed_at": datetime.utcnow()
        }
        
        # Add detailed scores if provided
        if detailed_scores:
            update_data["detailed_scores"] = detailed_scores
            
        # Add responses if provided
        if responses:
            update_data["responses"] = responses
            
        # Add skill summary if provided (essential data only)
        if skill_summary:
            update_data["skill_summary"] = skill_summary
            
        try:
            # Convert string ID to ObjectId if needed
            if isinstance(candidate_id, str):
                obj_id = ObjectId(candidate_id)
            else:
                obj_id = candidate_id
                
            print(f"Using ObjectId: {obj_id}")
                
            result = self.candidates_collection.update_one(
                {"_id": obj_id},
                {"$set": update_data}
            )
            
            print(f"Update result: {result.matched_count} matched, {result.modified_count} modified")
            
            if result.matched_count == 0:
                print(f"No candidate found with ID: {obj_id}")
                # Try to find by string ID as fallback
                result2 = self.candidates_collection.update_one(
                    {"_id": candidate_id},
                    {"$set": update_data}
                )
                print(f"Fallback update result: {result2.matched_count} matched, {result2.modified_count} modified")
            elif result.modified_count == 0:
                print(f"Candidate found but no changes made for ID: {obj_id}")
                
        except Exception as e:
            print(f"Error updating candidate score: {str(e)}")
            raise
    
    def get_candidate_profile(self, candidate_id: str) -> dict:
        """Get candidate profile by ID"""
        try:
            if isinstance(candidate_id, str):
                candidate_id = ObjectId(candidate_id)
            return self.candidates_collection.find_one({"_id": candidate_id})
        except Exception as e:
            print(f"Error getting candidate profile: {str(e)}")
            return None
