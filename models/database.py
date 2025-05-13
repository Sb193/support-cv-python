from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./cv_evaluations.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CVEvaluation(Base):
    __tablename__ = "cv_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String)
    prompt_vector = Column(JSON)  # Store vector as JSON
    evaluation = Column(JSON)  # Store AI evaluation as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 