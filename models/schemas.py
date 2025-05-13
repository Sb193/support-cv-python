from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Contact(BaseModel):
    email: str
    phone: str
    address: str
    dob: datetime

class Education(BaseModel):
    degree: str
    institution: str
    year: int

class Experience(BaseModel):
    job_title: str
    company: str
    description: str

class Project(BaseModel):
    name: str
    description: str
    technologies: List[str]

class Certification(BaseModel):
    name: str
    institution: str
    year: int

class CVRequest(BaseModel):
    name: str
    position: str
    contact: Contact
    term_goal: str
    education: List[Education]
    experience: List[Experience]
    skills: List[str]
    projects: List[Project]
    certifications: List[Certification]

class EvaluationResponse(BaseModel):
    score: float
    reasons: List[str]
    is_from_cache: bool = False

class APIResponse(BaseModel):
    success: bool
    data: Optional[EvaluationResponse] = None
    error: Optional[str] = None 