Here's the comprehensive Phase 2 markdown file:

```markdown
# Phase 2: Environment Setup & Foundation

**Status:** 🔄 **In Progress**  
**Timeline:** Week 2  
**Primary Owner:** DevOps Engineer  
**Prerequisites:** Phase 1 Completion ✅

---

## 🎯 Phase Objective

Establish a robust development environment, configure all external services, and set up the project foundation for seamless development and future deployment.

---

## 📋 Phase Deliverables

- [ ] Project repository initialized with proper structure
- [ ] Development environment configured locally
- [ ] Docker containerization working
- [ ] All external services connected and tested
- [ ] CI/CD pipeline foundation established
- [ ] Basic project dependencies installed and configured

---

## 🏗️ Project Structure

```
ai-resume-analyzer/
├── 📁 src/
│   ├── 📁 app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── config.py               # Configuration management
│   │   ├── database.py             # Database connection and setup
│   │   ├── models.py               # SQLAlchemy models
│   │   ├── schemas.py              # Pydantic schemas
│   │   ├── dependencies.py         # FastAPI dependencies
│   │   ├── 📁 api/                 # API routes
│   │   │   ├── __init__.py
│   │   │   ├── endpoints/
│   │   │   │   ├── auth.py
│   │   │   │   ├── analysis.py
│   │   │   │   └── history.py
│   │   │   └── middleware/
│   │   ├── 📁 services/            # Business logic
│   │   │   ├── auth_service.py
│   │   │   ├── file_service.py
│   │   │   └── llm_service.py
│   │   ├── 📁 core/               # Core utilities
│   │   │   ├── security.py        # JWT and password hashing
│   │   │   ├── celery_app.py      # Celery configuration
│   │   │   └── redis_client.py
│   │   └── 📁 workers/            # Celery tasks
│   │       ├── __init__.py
│   │       └── tasks.py           # Background tasks
├── 📁 tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api/
│   ├── test_services/
│   └── test_workers/
├── 📁 docs/
│   ├── phase1-design.md
│   ├── phase2-setup.md           # This file
│   └── api-reference.md
├── 📁 scripts/
│   ├── setup_env.sh
│   ├── run_tests.sh
│   └── deploy.sh
├── 📁 docker/
│   ├── Dockerfile
│   ├── Dockerfile.celery
│   └── docker-compose.yml
├── 📁 config/
│   ├── development.env
│   ├── production.env
│   └── prompts.yaml
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔧 Development Environment Setup

### 2.1 Prerequisites Installation

#### System Requirements
- **Python:** 3.11 or higher
- **Docker & Docker Compose**
- **Git**
- **Redis** (or use Docker version)

#### Python Environment Setup
```bash
# Clone repository
git clone https://github.com/your-org/ai-resume-analyzer.git
cd ai-resume-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### 2.2 Configuration Files

#### Environment Variables (`.env`)
```bash
# Application
APP_NAME="AI Resume Analyzer"
APP_ENV=development
DEBUG=true
SECRET_KEY=your-super-secret-key-here

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/resume_analyzer

# Redis
REDIS_URL=redis://localhost:6379/0

# Gemini AI
GEMINI_API_KEY=your-gemini-api-key-here

# Security
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
JWT_REFRESH_EXPIRE_DAYS=7

# Rate Limiting
RATE_LIMIT_PER_MINUTE=5
```

#### Prompts Configuration (`config/prompts.yaml`)
```yaml
resume_analysis_en: |
  You are an expert resume analyst. Analyze the following resume and job description. 
  Extract skills, calculate a compatibility score (0-100) based on 40% keyword match 
  and 60% semantic similarity, identify missing skills, and provide 3 actionable 
  improvement suggestions. Output MUST be a valid JSON matching the specified schema.

resume_analysis_fa: |
  شما یک متخصص تحلیل رزومه هستید. رزومه و شرح شغل زیر را تحلیل کنید.
  مهارت‌ها را استخراج کنید، امتیاز سازگاری (۰-۱۰۰) را محاسبه کنید...
  خروجی باید یک JSON معتبر باشد.

language_detection: |
  Detect the primary language of the following text. 
  Return only the language code (e.g., 'en', 'fa', 'ar').
```

---

## 🐳 Docker Configuration

### 2.3 Docker Setup

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

#### Docker Compose (`docker-compose.yml`)
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/resume_analyzer
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./src:/app/src
      - ./config:/app/config
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  celery-worker:
    build: 
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/resume_analyzer
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./src:/app/src
      - ./config:/app/config
    command: celery -A app.core.celery_app worker --loglevel=info

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=resume_analyzer
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

---

## 🔗 External Services Configuration

### 2.4 Service Connections

#### Google Gemini API Setup
1. **Access Google AI Studio:** https://aistudio.google.com/
2. **Create API Key:** Generate new API key in credentials section
3. **Test Connection:**
```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("Hello")
print(response.text)
```

#### Supabase Database Setup
1. **Create Project:** https://supabase.com
2. **Get Connection String:** Settings → Database → Connection string
3. **Initial Schema:** Run initial migration to create tables
4. **Enable Row Level Security (RLS):** Configure for multi-tenant security

#### Redis Setup
1. **Local Installation:** `docker run -d -p 6379:6379 redis:7-alpine`
2. **Cloud Alternative:** Redis Cloud or Railway Redis
3. **Test Connection:**
```python
import redis
r = redis.Redis(host='localhost', port=6379, db=0)
r.ping()  # Should return True
```

---

## 📦 Dependencies Management

### 2.5 Requirements Files

#### `requirements.txt`
```text
fastapi==0.104.1
uvicorn[standard]==0.24.0
celery==5.3.4
redis==5.0.1
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
google-generativeai==0.3.2
pdfplumber==0.10.3
python-docx==1.1.0
langdetect==1.0.9
pydantic==2.5.0
pydantic-settings==2.1.0
```

#### `requirements-dev.txt`
```text
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.11.0
isort==5.13.2
flake8==6.1.0
mypy==1.7.1
pre-commit==3.5.0
locust==2.20.1
httpx==0.25.2
```

---

## 🔄 CI/CD Pipeline Foundation

### 2.6 GitHub Actions Setup

#### `.github/workflows/ci.yml`
```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: user
          POSTGRES_PASSWORD: password
          POSTGRES_DB: resume_analyzer
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

---

## 🧪 Verification & Testing

### 2.7 Setup Verification Script

#### `scripts/verify_setup.py`
```python
#!/usr/bin/env python3
"""
Setup verification script for Phase 2 completion
"""

import os
import sys
import asyncio
from pathlib import Path

def check_environment():
    """Verify all environment variables are set"""
    required_vars = [
        'DATABASE_URL',
        'REDIS_URL', 
        'GEMINI_API_KEY',
        'SECRET_KEY'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        return False
    
    print("✅ All environment variables are set")
    return True

def check_dependencies():
    """Verify all required packages are installed"""
    try:
        import fastapi
        import celery
        import sqlalchemy
        import google.generativeai
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def check_docker():
    """Verify Docker is running and containers can be started"""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("✅ Docker is running")
        return True
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False

async def main():
    print("🔍 Verifying Phase 2 Setup...")
    print("-" * 40)
    
    checks = [
        check_environment(),
        check_dependencies(),
        check_docker()
    ]
    
    if all(checks):
        print("-" * 40)
        print("🎉 Phase 2 setup verification PASSED!")
        print("➡️ Proceed to Phase 3: Core Development")
        return 0
    else:
        print("-" * 40)
        print("❌ Phase 2 setup verification FAILED!")
        print("Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))
```

---

## 🚀 Current Tasks

### 2.8 Immediate Action Items

| Task | Owner | Status | Due Date |
|------|-------|--------|----------|
| Initialize Git repository | @dev1 | ✅ Done | 2024-01-16 |
| Set up Python virtual environment | @dev1 | 🔄 In Progress | 2024-01-17 |
| Configure Docker and docker-compose | @ops1 | ⏳ Pending | 2024-01-18 |
| Set up Supabase database | @dev1 | ⏳ Pending | 2024-01-19 |
| Configure Gemini API access | @ai1 | ⏳ Pending | 2024-01-19 |
| Create basic project structure | @dev1 | ⏳ Pending | 2024-01-20 |
| Set up CI/CD pipeline | @ops1 | ⏳ Pending | 2024-01-22 |

---

## 📝 Next Steps

### Completion Criteria
- [ ] All Docker containers start without errors
- [ ] Database connection established and migrations can run
- [ ] Redis connection working for Celery tasks
- [ ] Gemini API can be called successfully
- [ ] Basic FastAPI server starts and responds to requests
- [ ] CI pipeline passes all initial checks

### Ready for Phase 3 When:
1. ✅ Development environment is fully functional
2. ✅ All external services are connected and tested
3. ✅ Project structure follows established patterns
4. ✅ Team can run the application locally

---

## 🔗 Related Documents

- [← Back to Main Project Hub](../MAIN.md)
- [→ Proceed to Phase 3: Core Development](./phase3-development.md)
- [← Review Phase 1: Planning & Design](./phase1-design.md)

---
