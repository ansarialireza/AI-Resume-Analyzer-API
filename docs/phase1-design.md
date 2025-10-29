I'll create a comprehensive markdown file for the Phase 1 Design Document. Here's the complete document:

```markdown
# AI Resume Analyzer API: Phase 1 - Design Document

**Document Version:** 1.0  
**Date:** $(date)  
**Project:** AI Resume Analyzer API  
**Focus:** Backend System  
**Objective:** To define the complete specification, architecture, and design for a scalable, AI-powered resume analysis API.

---

## 1. Functional Specifications

The system shall provide the following core functionalities:

### 1.1 File Upload & Validation
- Accept resume files via a dedicated API endpoint
- Supported formats: PDF, DOCX
- Maximum file size: 10 MB
- Return a clear, structured error for invalid file types or sizes

### 1.2 Asynchronous Resume Analysis
- Process the analysis as a background task to avoid blocking the HTTP request
- Extract raw text from the uploaded file
- Automatically detect the document's primary language (e.g., English, Farsi, Arabic)
- Send the extracted text and an optional job description to a Large Language Model (LLM) for analysis
- The LLM will be instructed to return a structured JSON object containing:
  - A compatibility score (0-100)
  - A list of matched skills (with confidence scores and categories)
  - A list of missing skills (present in the job description but absent from the resume)
  - A summary of professional experience
  - Actionable suggestions for improving the resume

### 1.3 Result Storage & Retrieval
- Persist the analysis result in a database with a unique identifier
- Provide an endpoint for clients to poll for the analysis status and result using the unique ID
- Support soft-deletion of analysis history for user data management

### 1.4 System Reliability & Security
- Implement rate limiting to prevent abuse and ensure service stability
- Process files in a memory-efficient manner to handle high traffic

---

## 2. Data Design

### 2.1 Core Entities & Attributes

#### User Entity
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID, Primary Key | Unique user identifier |
| `email` | String, Unique | User's email address |
| `created_at` | DateTime | Account creation timestamp |

#### ResumeAnalysis Entity
| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID, Primary Key | Unique analysis identifier |
| `user_id` | UUID, Foreign Key | Link to the user (Nullable in initial phase) |
| `file_hash` | String, Unique | Hash of the file to prevent duplicate processing |
| `original_filename` | String | Name of the uploaded file |
| `file_size` | Integer | File size in bytes |
| `language` | String | Detected language code (e.g., 'en', 'fa') |
| `raw_text` | Text | Plain text extracted from the resume file |
| `job_description` | Text, Nullable | Job description provided by the user |
| `analysis_report` | JSONB | The complete structured result from the LLM |
| `status` | String | Processing state (`processing`, `completed`, `failed`) |
| `task_id` | String, Nullable | ID of the background Celery task |
| `is_deleted` | Boolean | Flag for soft deletion |
| `created_at` | DateTime | Analysis request timestamp |
| `completed_at` | DateTime, Nullable | Analysis completion timestamp |

### 2.2 Analysis Report JSON Schema

The `analysis_report` field will store a JSON object conforming to the following schema:

```json
{
  "score": 85.5,
  "language": "fa",
  "matched_skills": [
    {
      "name": "Python",
      "confidence": 0.98,
      "category": "Programming"
    },
    {
      "name": "Team Leadership",
      "confidence": 0.85,
      "category": "Soft Skills"
    }
  ],
  "missing_skills": [
    "Kubernetes",
    "Prometheus"
  ],
  "experience_summary": "5 years of backend development experience...",
  "improvement_suggestions": [
    "Quantify achievements: Change 'managed a team' to 'led a 5-developer team, resulting in 20% faster deployment cycles.'"
  ]
}
```

---

Here's the enhanced endpoints section with authentication endpoints added:

### 3.1 Endpoints

| Method | Endpoint | Description | Request Body | Success Response | Key Errors |
|--------|----------|-------------|--------------|------------------|------------|
| **POST** | `/api/v1/auth/register` | Register a new user | `JSON`: `{ "email": "string", "password": "string" }` | `201 Created`: `{ "user_id": "uuid", "email": "string", "message": "User created successfully" }` | `400: EMAIL_ALREADY_EXISTS` <br> `400: INVALID_EMAIL` <br> `400: WEAK_PASSWORD` |
| **POST** | `/api/v1/auth/login` | Authenticate user and get tokens | `JSON`: `{ "email": "string", "password": "string" }` | `200 OK`: `{ "access_token": "string", "refresh_token": "string", "token_type": "bearer", "expires_in": 3600 }` | `401: INVALID_CREDENTIALS` <br> `400: MISSING_FIELDS` <br> `429: RATE_LIMIT_EXCEEDED` |
| **POST** | `/api/v1/auth/refresh` | Refresh access token | `JSON`: `{ "refresh_token": "string" }` | `200 OK`: `{ "access_token": "string", "token_type": "bearer", "expires_in": 3600 }` | `401: INVALID_REFRESH_TOKEN` <br> `400: TOKEN_EXPIRED` <br> `400: MISSING_TOKEN` |
| **POST** | `/api/v1/auth/logout` | Logout user and invalidate tokens | `Headers`: `Authorization: Bearer {token}` | `200 OK`: `{ "message": "Successfully logged out" }` | `401: UNAUTHORIZED` <br> `400: INVALID_TOKEN` |
| **GET** | `/api/v1/auth/me` | Get current user profile | `Headers`: `Authorization: Bearer {token}` | `200 OK`: `{ "user_id": "uuid", "email": "string", "created_at": "datetime", "analysis_count": 5 }` | `401: UNAUTHORIZED` <br> `404: USER_NOT_FOUND` |
| **POST** | `/api/v1/analyze` | Submit a resume for analysis | `Headers`: `Authorization: Bearer {token}` <br> `Form-Data`: `file` (Required), `job_description` (Optional) | `201 Created`: `{ "analysis_id": "uuid", "status": "processing" }` | `415: UNSUPPORTED_FILE_TYPE` <br> `413: FILE_TOO_LARGE` <br> `429: RATE_LIMIT_EXCEEDED` <br> `401: UNAUTHORIZED` |
| **GET** | `/api/v1/results/{analysis_id}` | Retrieve analysis status and result | `Headers`: `Authorization: Bearer {token}` | `200 OK`: `{ "analysis_id": "uuid", "status": "completed", "result": { ... } }` | `404: ANALYSIS_NOT_FOUND` <br> `403: FORBIDDEN` <br> `401: UNAUTHORIZED` |
| **GET** | `/api/v1/history` | Get user's analysis history | `Headers`: `Authorization: Bearer {token}` <br> `Query Params`: `page=1`, `limit=20`, `status=completed` | `200 OK`: `{ "analyses": [...], "pagination": { "page": 1, "limit": 20, "total": 45, "pages": 3 } }` | `401: UNAUTHORIZED` <br> `400: INVALID_PAGINATION` |
| **DELETE** | `/api/v1/history/{analysis_id}` | Soft-delete an analysis record | `Headers`: `Authorization: Bearer {token}` | `204 No Content` | `404: ANALYSIS_NOT_FOUND` <br> `403: FORBIDDEN` <br> `401: UNAUTHORIZED` |
| **POST** | `/api/v1/auth/forgot-password` | Request password reset email | `JSON`: `{ "email": "string" }` | `200 OK`: `{ "message": "If the email exists, a reset link has been sent" }` | `429: RATE_LIMIT_EXCEEDED` |
| **POST** | `/api/v1/auth/reset-password` | Reset password with token | `JSON`: `{ "token": "string", "new_password": "string" }` | `200 OK`: `{ "message": "Password reset successfully" }` | `400: INVALID_TOKEN` <br> `400: TOKEN_EXPIRED` <br> `400: WEAK_PASSWORD` |

### Authentication Flow Summary:

1. **Registration** → User creates account with email/password
2. **Login** → User gets access_token and refresh_token
3. **Protected Endpoints** → Include `Authorization: Bearer {access_token}` header
4. **Token Refresh** → Use refresh_token to get new access_token when expired
5. **Logout** → Invalidate tokens on client side (optional server-side blacklist)

### Security Headers:
- All authenticated requests require: `Authorization: Bearer {jwt_token}`
- Rate limiting applied to auth endpoints to prevent brute force
- JWT tokens expire in 1 hour (access) and 7 days (refresh)
---

## 4. External Dependencies & Technology Stack

| Component | Selected Technology / Service | Purpose | Required Action |
|-----------|-------------------------------|---------|-----------------|
| **API Framework** | FastAPI | To build the REST API with automatic docs, validation, and async support | - |
| **Task Queue** | Celery + Redis | To manage and execute background jobs (resume processing) asynchronously | Install Redis locally or provision a cloud instance |
| **LLM Provider** | Google Gemini Flash | The core AI engine for resume analysis and structured JSON generation | Obtain API Key from Google AI Studio |
| **Database** | PostgreSQL (via Supabase) | To persistently store users, analysis requests, and results. JSONB support is crucial | Create a project on Supabase and retrieve the connection string |
| **Deployment** | Railway / Render | For cloud-based container deployment and hosting | - |

---

## 5. High-Level Architecture & Data Flow

The system follows a layered, event-driven architecture:

### 5.1 Architecture Layers

1. **Gate Layer (FastAPI)**
   - Request validation and rate limiting
   - Authentication and authorization
   - Immediate response with task ID

2. **Bridge Layer (Celery + Redis)**
   - Asynchronous task management
   - Job queuing and distribution
   - Retry mechanisms and error handling

3. **AI Layer (LLM Service)**
   - Text extraction and preprocessing
   - Language detection
   - Prompt engineering and LLM communication
   - Response validation and parsing

4. **Data Layer (PostgreSQL)**
   - Persistent data storage
   - Transaction management
   - Query optimization and indexing

### 5.2 Data Flow

1. **Request Ingestion (FastAPI):**
   - Client sends `POST /analyze` request with resume file
   - API gateway performs initial validation (file type, size)
   - Creates new `ResumeAnalysis` record with status `processing`
   - Enqueues `analyze_resume_task` in Redis queue
   - Immediately returns `analysis_id` to client

2. **Background Processing (Celery Worker):**
   - Celery worker picks up `analyze_resume_task` from queue
   - Executes processing chain:
     - **Text Extraction**: Uses `pdfplumber` and `python-docx` libraries
     - **Language Detection**: Uses `langdetect` library
     - **AI Analysis**: Sends data with language-specific prompt to Gemini API
     - **Response Validation**: Parses and validates JSON response

3. **Data Persistence (PostgreSQL):**
   - Worker updates `ResumeAnalysis` record
   - Saves validated JSON result to `analysis_report` field
   - Sets `status` to `completed`

4. **Result Retrieval (FastAPI):**
   - Client polls `GET /results/{analysis_id}` endpoint
   - API fetches record from database
   - Returns current status and final result (if available)

---

## 6. Success Metrics & Phase 1 Deliverables

### 6.1 Key Performance Indicators (KPIs)

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Accuracy** | >95% for English, >92% for Farsi | Manual testing on sample resumes vs human evaluation |
| **Performance** | <10 seconds end-to-end processing | Automated timing tests with Locust |
| **Reliability** | 99.9% API uptime | Monitoring with UptimeRobot |
| **Success Rate** | 99% of valid files processed without errors | Log analysis and error tracking |

### 6.2 Phase 1 Deliverables

This phase is complete upon the creation and review of this document and its supporting assets:

1. **Software Design Document (SDD)**
   - This comprehensive design specification
   - Approved by all stakeholders

2. **Stubbed Project Repository**
   - Git repository with basic structure
   - Directory layout: `/src`, `/tests`, `/docs`
   - Configuration files: `requirements.txt`, `docker-compose.yml`

3. **External Service Access**
   - API keys for Google Gemini AI Studio
   - Supabase project with connection string
   - Redis instance configuration

4. **Test Data Set**
   - Sample resumes (PDF/DOCX) in multiple languages
   - Corresponding job descriptions
   - Expected output formats for validation

---

## 7. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| LLM API rate limiting | High | Medium | Implement exponential backoff, caching, fallback models |
| Large file processing failures | Medium | High | Chunked processing, memory monitoring, timeout handling |
| Database performance issues | Low | High | Proper indexing, connection pooling, query optimization |
| Language detection inaccuracies | Medium | Medium | Confidence thresholds, user override option, fallback to English |

---

## 8. Next Steps

Upon approval of this design document, the project will proceed to:

1. **Phase 2: Setup & Foundation**
   - Initialize development environment
   - Set up CI/CD pipeline
   - Configure monitoring and logging

2. **Phase 3: Core Development**
   - Implement API endpoints
   - Build database models and migrations
   - Create Celery tasks and workers

3. **Phase 4: AI Integration**
   - Implement LLM service integration
   - Develop prompt engineering strategies
   - Create response validation systems

---