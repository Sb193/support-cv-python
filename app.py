from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from models.database import get_db, CVEvaluation
from models.schemas import CVRequest, APIResponse, EvaluationResponse
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import json
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_prompt(cv_data: CVRequest) -> str:
    """Create a prompt from CV data, excluding personal contact information."""
    prompt = f"""
    Position: {cv_data.position}
    Term Goal: {cv_data.term_goal}
    
    Education:
    {chr(10).join([f"- {edu.degree} from {edu.institution} ({edu.year})" for edu in cv_data.education])}
    
    Experience:
    {chr(10).join([f"- {exp.job_title} at {exp.company}: {exp.description}" for exp in cv_data.experience])}
    
    Skills:
    {', '.join(cv_data.skills)}
    
    Projects:
    {chr(10).join([f"- {proj.name}: {proj.description} (Technologies: {', '.join(proj.technologies)})" for proj in cv_data.projects])}
    
    Certifications:
    {chr(10).join([f"- {cert.name} from {cert.institution} ({cert.year})" for cert in cv_data.certifications])}
    """
    return prompt.strip()

def get_vector_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_ai_evaluation(prompt: str) -> dict:
    """Get evaluation from the AI API."""
    api_url = "https://gencv.sbac.workers.dev/"
    try:
        response = requests.post(api_url, json={"prompt": prompt})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling AI API: {str(e)}")

@app.post("/evaluate-cv", response_model=APIResponse)
async def evaluate_cv(cv_data: CVRequest, db: Session = Depends(get_db)):
    try:
        # Create prompt and get vector
        prompt = create_prompt(cv_data)
        prompt_vector = model.encode(prompt).tolist()

        # Check for similar CVs in database
        existing_evaluations = db.query(CVEvaluation).all()
        for eval in existing_evaluations:
            similarity = get_vector_similarity(prompt_vector, eval.prompt_vector)
            if similarity > 0.9:  # 90% similarity threshold
                return APIResponse(
                    success=True,
                    data=EvaluationResponse(
                        score=eval.evaluation["score"],
                        reasons=eval.evaluation["reasons"],
                        is_from_cache=True
                    )
                )
        new_prompt = "You are a recruiter. You are given a CV and you need to evaluate the candidate based on the following criteria: " + prompt
        # If no similar CV found, get new evaluation
        evaluation = get_ai_evaluation(new_prompt)
        
        # Save to database
        db_evaluation = CVEvaluation(
            prompt=prompt,
            prompt_vector=prompt_vector,
            evaluation=evaluation
        )
        db.add(db_evaluation)
        db.commit()

        return APIResponse(
            success=True,
            data=EvaluationResponse(
                score=evaluation["score"],
                reasons=evaluation["reasons"],
                is_from_cache=False
            )
        )

    except Exception as e:
        return APIResponse(
            success=False,
            error=str(e)
        )
