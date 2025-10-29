Here's the comprehensive Phase 3 markdown file:

```markdown
# Phase 3: Core Development

**Status:** ⏳ **Pending**  
**Timeline:** Week 3-4  
**Primary Owner:** Backend Team  
**Prerequisites:** Phase 2 Completion ✅

---

## 🎯 Phase Objective

Implement the core backend functionality, including API endpoints, database models, authentication system, and basic file processing pipeline.

---

## 📋 Phase Deliverables

- [ ] FastAPI application structure complete
- [ ] Database models and migrations implemented
- [ ] Authentication and authorization system working
- [ ] File upload and validation endpoints functional
- [ ] Celery task queue processing background jobs
- [ ] Basic error handling and logging system
- [ ] API documentation auto-generated
- [ ] Unit tests for core functionality

---

## 🏗️ Core Application Structure

### 3.1 Application Entry Point

#### `src/app/main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.api import api_router
from app.core.celery_app import celery_app
from app.database import engine, Base

# Create database tables
Base.metadata.create_all(bind=engine)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered Resume Analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "AI Resume Analyzer API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",  # Add actual DB check
        "redis": "connected"     # Add actual Redis check
    }
```

### 3.2 Configuration Management

#### `src/app/core/config.py`
```python
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AI Resume Analyzer"
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str
    
    # Gemini AI
    GEMINI_API_KEY: str
    
    # Security
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_EXPIRE_DAYS: int = 7
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 5
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = ["pdf", "docx"]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 🗄️ Database Models & Migrations

### 3.3 SQLAlchemy Models

#### `src/app/models.py`
```python
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ResumeAnalysis(Base):
    __tablename__ = "resume_analyses"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Nullable for Phase 3
    file_hash = Column(String, nullable=False, unique=True, index=True)
    original_filename = Column(String, nullable=False)
    file_size = Column(Integer)  # Size in bytes
    language = Column(String, nullable=False)  # 'en', 'fa', 'ar'
    raw_text = Column(Text, nullable=False)
    job_description = Column(Text, nullable=True)
    analysis_report = Column(JSON, nullable=True)  # Stores the complete analysis result
    status = Column(String, default="processing")  # processing, completed, failed
    task_id = Column(String, nullable=True)  # Celery task ID
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Index for better query performance
    __table_args__ = (
        Index('ix_resume_analyses_user_status', 'user_id', 'status'),
        Index('ix_resume_analyses_created_at', 'created_at'),
    )
```

### 3.4 Database Connection & Session Management

#### `src/app/database.py`
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## 🔐 Authentication System

### 3.5 Security Utilities

#### `src/app/core/security.py`
```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None
```

### 3.6 Authentication Service

#### `src/app/services/auth_service.py`
```python
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.models import User
from app.core.security import verify_password, get_password_hash, create_access_token

class AuthService:
    @staticmethod
    def authenticate_user(db: Session, email: str, password: str):
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user
    
    @staticmethod
    def create_user(db: Session, email: str, password: str):
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(password)
        user = User(email=email, hashed_password=hashed_password)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    
    @staticmethod
    def create_access_token_for_user(user: User):
        token_data = {"sub": user.email, "user_id": user.id}
        return create_access_token(data=token_data)
```

---

## 📡 API Endpoints Implementation

### 3.7 Authentication Endpoints

#### `src/app/api/endpoints/auth.py`
```python
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from app.database import get_db
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    user = AuthService.create_user(db, user_data.email, user_data.password)
    return {"user_id": user.id, "email": user.email, "message": "User created successfully"}

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = AuthService.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    access_token = AuthService.create_access_token_for_user(user)
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 3600  # 1 hour
    }
```

### 3.8 Analysis Endpoints

#### `src/app/api/endpoints/analysis.py`
```python
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Optional
import hashlib
import os

from app.database import get_db
from app.models import ResumeAnalysis
from app.workers.tasks import analyze_resume_task
from app.services.file_service import FileService

router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
async def analyze_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Resume file (PDF/DOCX)"),
    job_description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    # Validate file type
    if not FileService.is_valid_file_type(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="File type not supported. Please upload PDF or DOCX files."
        )
    
    # Validate file size
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit."
        )
    
    # Calculate file hash for duplicate detection
    file_hash = hashlib.md5(file_content).hexdigest()
    
    # Check for existing analysis
    existing_analysis = db.query(ResumeAnalysis).filter(
        ResumeAnalysis.file_hash == file_hash
    ).first()
    
    if existing_analysis:
        return {
            "analysis_id": existing_analysis.id,
            "status": existing_analysis.status,
            "message": "Analysis already exists"
        }
    
    # Create new analysis record
    analysis = ResumeAnalysis(
        file_hash=file_hash,
        original_filename=file.filename,
        file_size=len(file_content),
        job_description=job_description,
        status="processing"
    )
    
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    # Start background task
    background_tasks.add_task(
        analyze_resume_task,
        analysis_id=analysis.id,
        file_content=file_content,
        file_name=file.filename,
        job_description=job_description
    )
    
    return {
        "analysis_id": analysis.id,
        "status": "processing",
        "message": "Analysis started successfully"
    }

@router.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    return {
        "analysis_id": analysis.id,
        "status": analysis.status,
        "result": analysis.analysis_report if analysis.status == "completed" else None,
        "created_at": analysis.created_at,
        "completed_at": analysis.completed_at
    }
```

---

## 🔄 Celery Task Queue

### 3.9 Celery Configuration

#### `src/app/core/celery_app.py`
```python
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "resume_analyzer",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
)
```

### 3.10 Background Tasks

#### `src/app/workers/tasks.py`
```python
from celery import current_task
from sqlalchemy.orm import Session
import time

from app.core.celery_app import celery_app
from app.database import SessionLocal
from app.models import ResumeAnalysis
from app.services.file_service import FileService
from app.services.llm_service import LLMService

@celery_app.task(bind=True, name="analyze_resume_task")
def analyze_resume_task(self, analysis_id: str, file_content: bytes, file_name: str, job_description: str = None):
    db = SessionLocal()
    try:
        # Update task status
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        if not analysis:
            return {"error": "Analysis not found"}
        
        # Step 1: Extract text from file
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 1, "total": 4, "status": "Extracting text from file..."}
        )
        
        try:
            raw_text = FileService.extract_text_from_file(file_content, file_name)
            analysis.raw_text = raw_text
            db.commit()
        except Exception as e:
            analysis.status = "failed"
            db.commit()
            return {"error": f"Text extraction failed: {str(e)}"}
        
        # Step 2: Detect language
        current_task.update_state(
            state="PROGRESS", 
            meta={"current": 2, "total": 4, "status": "Detecting language..."}
        )
        
        try:
            language = FileService.detect_language(raw_text)
            analysis.language = language
            db.commit()
        except Exception as e:
            analysis.status = "failed"
            db.commit()
            return {"error": f"Language detection failed: {str(e)}"}
        
        # Step 3: Analyze with AI (placeholder for now)
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 3, "total": 4, "status": "Analyzing content with AI..."}
        )
        
        try:
            # Temporary mock analysis - will be replaced with actual LLM integration in Phase 4
            mock_analysis = {
                "score": 75.0,
                "language": language,
                "matched_skills": [
                    {"name": "Python", "confidence": 0.95, "category": "Programming"},
                    {"name": "FastAPI", "confidence": 0.85, "category": "Framework"}
                ],
                "missing_skills": ["Kubernetes", "Docker"],
                "experience_summary": "Extracted experience summary placeholder",
                "improvement_suggestions": ["Add more quantitative achievements to your resume."]
            }
            analysis.analysis_report = mock_analysis
        except Exception as e:
            analysis.status = "failed"
            db.commit()
            return {"error": f"AI analysis failed: {str(e)}"}
        
        # Step 4: Mark as completed
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 4, "total": 4, "status": "Finalizing analysis..."}
        )
        
        analysis.status = "completed"
        from datetime import datetime
        analysis.completed_at = datetime.utcnow()
        db.commit()
        
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "result": analysis.analysis_report
        }
        
    except Exception as e:
        # Update analysis status to failed
        if 'analysis' in locals():
            analysis.status = "failed"
            db.commit()
        return {"error": f"Task failed: {str(e)}"}
    finally:
        db.close()
```

---

## 📁 File Processing Service

### 3.11 File Service Implementation

#### `src/app/services/file_service.py`
```python
import pdfplumber
from docx import Document
from langdetect import detect, LangDetectError
import io

class FileService:
    ALLOWED_EXTENSIONS = {'.pdf', '.docx'}
    
    @staticmethod
    def is_valid_file_type(filename: str) -> bool:
        return any(filename.lower().endswith(ext) for ext in FileService.ALLOWED_EXTENSIONS)
    
    @staticmethod
    def extract_text_from_file(file_content: bytes, filename: str) -> str:
        file_extension = filename.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return FileService._extract_text_from_pdf(file_content)
        elif file_extension == 'docx':
            return FileService._extract_text_from_docx(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    @staticmethod
    def _extract_text_from_pdf(file_content: bytes) -> str:
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
        
        if not text.strip():
            raise Exception("No text could be extracted from PDF")
        
        return text.strip()
    
    @staticmethod
    def _extract_text_from_docx(file_content: bytes) -> str:
        try:
            doc = Document(io.BytesIO(file_content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"DOCX extraction failed: {str(e)}")
        
        if not text.strip():
            raise Exception("No text could be extracted from DOCX")
        
        return text.strip()
    
    @staticmethod
    def detect_language(text: str) -> str:
        try:
            # Sample first 1000 characters for language detection
            sample_text = text[:1000]
            language = detect(sample_text)
            return language
        except LangDetectError:
            return 'en'  # Default to English if detection fails
```

---

## 🧪 Testing Implementation

### 3.12 Basic Test Suite

#### `tests/test_api/test_auth.py`
```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import get_db, Base

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override get_db dependency
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_register_user(test_db):
    response = client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "testpassword123"
    })
    assert response.status_code == 201
    data = response.json()
    assert "user_id" in data
    assert data["email"] == "test@example.com"

def test_login_user(test_db):
    # First register
    client.post("/api/v1/auth/register", json={
        "email": "test@example.com", 
        "password": "testpassword123"
    })
    
    # Then login
    response = client.post("/api/v1/auth/login", data={
        "username": "test@example.com",
        "password": "testpassword123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
```

---

## 🚀 Current Tasks

### 3.13 Development Tasks

| Task | Owner | Status | Due Date |
|------|-------|--------|----------|
| Implement database models | @dev1 | ⏳ Pending | 2024-01-23 |
| Create authentication system | @dev2 | ⏳ Pending | 2024-01-24 |
| Implement file upload endpoints | @dev1 | ⏳ Pending | 2024-01-25 |
| Set up Celery task queue | @dev2 | ⏳ Pending | 2024-01-26 |
| Create file processing service | @dev1 | ⏳ Pending | 2024-01-27 |
| Implement basic error handling | @dev2 | ⏳ Pending | 2024-01-28 |
| Write unit tests for core functionality | @dev1 | ⏳ Pending | 2024-01-29 |
| Set up API documentation | @dev2 | ⏳ Pending | 2024-01-30 |

---

## 📝 Completion Criteria

### Ready for Phase 4 When:
- [ ] All API endpoints are functional and tested
- [ ] Authentication system works end-to-end
- [ ] File upload and processing pipeline works
- [ ] Celery tasks can process files in background
- [ ] Database operations are efficient and error-free
- [ ] Basic error handling and logging implemented
- [ ] Unit test coverage >80% for core functionality
- [ ] API documentation accessible at `/docs`

---

## 🔗 Related Documents

- [← Back to Main Project Hub](../MAIN.md)
- [→ Proceed to Phase 4: AI Integration](./phase4-ai-integration.md)
- [← Review Phase 2: Environment Setup](./phase2-setup.md)

---
