Here's the comprehensive Phase 4 markdown file:

```markdown
# Phase 4: AI Integration

**Status:** ⏳ **Pending**  
**Timeline:** Week 5  
**Primary Owner:** AI/ML Team  
**Prerequisites:** Phase 3 Completion ✅

---

## 🎯 Phase Objective

Integrate Google Gemini AI for intelligent resume analysis, implement advanced prompt engineering, and create a robust AI service layer that can process resumes in multiple languages with high accuracy.

---

## 📋 Phase Deliverables

- [ ] Google Gemini API integration complete
- [ ] Advanced prompt engineering for resume analysis
- [ ] Multilingual support (English, Farsi, Arabic)
- [ ] Response validation and parsing system
- [ ] AI service error handling and retry logic
- [ ] Performance optimization for AI calls
- [ ] Enhanced analysis reporting
- [ ] AI-specific testing and validation

---

## 🧠 AI Service Architecture

### 4.1 LLM Service Core

#### `src/app/services/llm_service.py`
```python
import google.generativeai as genai
import logging
import json
import asyncio
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.prompt_engineer import PromptEngineer

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self._configure_gemini()
        self.prompt_engineer = PromptEngineer()
        self.model = genai.GenerativeModel('gemini-pro')
    
    def _configure_gemini(self):
        """Configure Gemini API with settings"""
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            logger.info("Gemini AI configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini AI: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    async def analyze_resume(
        self, 
        resume_text: str, 
        job_description: Optional[str] = None,
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Analyze resume with AI and return structured results
        """
        try:
            # Generate appropriate prompt based on language
            prompt = self.prompt_engineer.get_analysis_prompt(
                resume_text=resume_text,
                job_description=job_description,
                language=language
            )
            
            # Call Gemini API
            response = await self._call_gemini_with_retry(prompt)
            
            # Parse and validate response
            analysis_result = self._parse_ai_response(response, language)
            
            # Enhance with additional metrics
            enhanced_result = self._enhance_analysis_result(analysis_result, resume_text)
            
            logger.info(f"AI analysis completed successfully for language: {language}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            raise
    
    async def _call_gemini_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call Gemini API with retry logic for reliability
        """
        for attempt in range(max_retries):
            try:
                # Use async execution for better performance
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.model.generate_content(prompt)
                )
                
                if response.text:
                    return response.text
                else:
                    raise ValueError("Empty response from Gemini API")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for Gemini API call")
                    raise
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}): {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def _parse_ai_response(self, response_text: str, language: str) -> Dict[str, Any]:
        """
        Parse and validate AI response into structured JSON
        """
        try:
            # Clean response text - remove markdown code blocks if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            # Parse JSON
            result = json.loads(cleaned_text)
            
            # Validate required fields
            required_fields = ['score', 'matched_skills', 'missing_skills', 'improvement_suggestions']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field in AI response: {field}")
            
            # Validate score range
            if not (0 <= result['score'] <= 100):
                raise ValueError(f"Invalid score range: {result['score']}")
            
            # Validate skills structure
            self._validate_skills_structure(result['matched_skills'])
            
            logger.debug("AI response parsed and validated successfully")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {str(e)}")
            logger.error(f"Raw response: {response_text}")
            raise ValueError("AI returned invalid JSON format")
        except Exception as e:
            logger.error(f"AI response validation failed: {str(e)}")
            raise
    
    def _validate_skills_structure(self, skills: list):
        """Validate skills list structure"""
        for skill in skills:
            if not isinstance(skill, dict):
                raise ValueError("Skills must be dictionaries")
            if 'name' not in skill:
                raise ValueError("Skill missing 'name' field")
            if 'confidence' not in skill:
                raise ValueError("Skill missing 'confidence' field")
            if not (0 <= skill['confidence'] <= 1):
                raise ValueError(f"Invalid confidence value: {skill['confidence']}")
    
    def _enhance_analysis_result(self, result: Dict[str, Any], resume_text: str) -> Dict[str, Any]:
        """
        Enhance analysis result with additional metrics and insights
        """
        # Add text statistics
        word_count = len(resume_text.split())
        result['text_statistics'] = {
            'word_count': word_count,
            'readability_score': self._calculate_readability_score(resume_text),
            'sections_detected': self._detect_sections(resume_text)
        }
        
        # Add analysis metadata
        result['analysis_metadata'] = {
            'model_used': 'gemini-pro',
            'analysis_timestamp': str(asyncio.get_event_loop().time()),
            'version': '1.0'
        }
        
        # Categorize skills if not already done
        result['matched_skills'] = self._categorize_skills(result['matched_skills'])
        
        return result
    
    def _calculate_readability_score(self, text: str) -> float:
        """Calculate simple readability score"""
        words = text.split()
        sentences = text.split('.')
        
        if len(words) == 0 or len(sentences) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple heuristic score (higher is better)
        readability = max(0, 100 - (avg_sentence_length * 0.5 + avg_word_length * 2))
        return min(100.0, readability)
    
    def _detect_sections(self, text: str) -> list:
        """Detect common resume sections"""
        sections = []
        text_lower = text.lower()
        
        section_keywords = {
            'experience': ['experience', 'work history', 'employment', 'career'],
            'education': ['education', 'academic', 'degree', 'university'],
            'skills': ['skills', 'technical skills', 'competencies'],
            'projects': ['projects', 'portfolio', 'achievements'],
            'certifications': ['certifications', 'certificate', 'licenses']
        }
        
        for section, keywords in section_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                sections.append(section)
        
        return sections
    
    def _categorize_skills(self, skills: list) -> list:
        """Categorize skills into predefined categories"""
        skill_categories = {
            'programming': ['python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'php', 'ruby'],
            'framework': ['fastapi', 'django', 'flask', 'react', 'vue', 'angular', 'spring', 'laravel'],
            'database': ['postgresql', 'mysql', 'mongodb', 'redis', 'sqlite', 'oracle'],
            'devops': ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'jenkins'],
            'tools': ['git', 'linux', 'bash', 'vim', 'vscode', 'pycharm'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem-solving']
        }
        
        for skill in skills:
            skill_name = skill['name'].lower()
            skill['category'] = 'other'  # Default category
            
            for category, keywords in skill_categories.items():
                if any(keyword in skill_name for keyword in keywords):
                    skill['category'] = category
                    break
        
        return skills

# Global instance
llm_service = LLMService()
```

---

## 🎨 Prompt Engineering System

### 4.2 Advanced Prompt Management

#### `src/app/core/prompt_engineer.py`
```python
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptEngineer:
    def __init__(self, prompts_file: str = "config/prompts.yaml"):
        self.prompts_file = Path(prompts_file)
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML configuration file"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as file:
                prompts = yaml.safe_load(file)
            logger.info("Prompts loaded successfully")
            return prompts
        except Exception as e:
            logger.error(f"Failed to load prompts: {str(e)}")
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict[str, Any]:
        """Fallback default prompts"""
        return {
            'resume_analysis_en': self._get_english_prompt(),
            'resume_analysis_fa': self._get_persian_prompt(),
            'resume_analysis_ar': self._get_arabic_prompt(),
            'language_detection': "Detect the primary language of this text. Return only the language code (e.g., 'en', 'fa', 'ar'): {text}"
        }
    
    def get_analysis_prompt(self, resume_text: str, job_description: Optional[str], language: str) -> str:
        """Get appropriate analysis prompt based on language"""
        prompt_key = f"resume_analysis_{language}"
        
        if prompt_key not in self.prompts:
            logger.warning(f"No prompt found for language {language}, falling back to English")
            prompt_key = "resume_analysis_en"
        
        base_prompt = self.prompts[prompt_key]
        
        # Format prompt with actual content
        formatted_prompt = base_prompt.format(
            resume_text=resume_text[:15000],  # Limit text length
            job_description=job_description or "No job description provided"
        )
        
        return formatted_prompt
    
    def _get_english_prompt(self) -> str:
        return """You are an expert resume analyst and career coach. Analyze the following resume and job description thoroughly.

RESUME TEXT:
{resume_text}

JOB DESCRIPTION:
{job_description}

Please provide a comprehensive analysis in the following EXACT JSON format. Do not include any other text or explanations:

{{
  "score": 85.5,
  "language": "en",
  "matched_skills": [
    {{
      "name": "Python",
      "confidence": 0.95,
      "category": "Programming",
      "evidence": "Mentioned in experience section"
    }}
  ],
  "missing_skills": ["Kubernetes", "Docker"],
  "experience_summary": "Clear summary of professional background...",
  "improvement_suggestions": [
    "Quantify achievements with metrics and numbers",
    "Add more specific technical skills",
    "Include project outcomes and impacts"
  ],
  "strengths": [
    "Strong technical background in backend development",
    "Good project leadership experience"
  ],
  "red_flags": [
    "Employment gaps not explained",
    "Limited mention of specific achievements"
  ]
}}

Scoring Guidelines:
- 90-100: Excellent match, highly qualified
- 80-89: Strong match, well qualified  
- 70-79: Good match, generally qualified
- 60-69: Fair match, some gaps
- Below 60: Poor match, significant gaps

Be thorough, objective, and provide actionable feedback."""

    def _get_persian_prompt(self) -> str:
        return """شما یک متخصص تحلیل رزومه و مشاور شغلی هستید. رزومه و شرح شغل زیر را به طور کامل تحلیل کنید.

متن رزومه:
{resume_text}

شرح شغل:
{job_description}

لطفاً تحلیل جامع را در قالب JSON دقیق زیر ارائه دهید. هیچ متن یا توضیح دیگری اضافه نکنید:

{{
  "score": 85.5,
  "language": "fa",
  "matched_skills": [
    {{
      "name": "پایتون",
      "confidence": 0.95,
      "category": "برنامه‌نویسی",
      "evidence": "ذکر شده در بخش سوابق کاری"
    }}
  ],
  "missing_skills": ["Kubernetes", "Docker"],
  "experience_summary": "خلاصه واضح از سوابق حرفه‌ای...",
  "improvement_suggestions": [
    "دستاوردها را با اعداد و ارقام کمی کنید",
    "مهارت‌های فنی خاص بیشتری اضافه کنید", 
    "نتایج و تاثیرات پروژه‌ها را включить کنید"
  ],
  "strengths": [
    "پس‌زمینه فنی قوی در توسعه بک‌اند",
    "تجربه رهبری پروژه خوب"
  ],
  "red_flags": [
    "فاصله‌های اشتغال توضیح داده نشده",
    "ذکر محدود دستاوردهای خاص"
  ]
}}

دستورالعمل امتیازدهی:
- ۱۰۰-۹۰: تطابق عالی، بسیار واجد شرایط
- ۸۹-۸۰: تطابق قوی، واجد شرایط
- ۷۹-۷۰: تطابق خوب، عموماً واجد شرایط
- ۶۹-۶۰: تطابق متوسط، برخی شکاف‌ها
- زیر ۶۰: تطابق ضعیف، شکاف‌های قابل توجه

کامل، عینی و بازخورد قابل اجرا ارائه دهید."""

    def _get_arabic_prompt(self) -> str:
        return """أنت خبير في تحليل السير الذاتية ومستشار مهني. قم بتحليل السيرة الذاتية ووصف الوظيفة التاليين بدقة.

نص السيرة الذاتية:
{resume_text}

وصف الوظيفة:
{job_description}

يرجى تقديم تحليل شامل بتنسيق JSON الدقيق التالي. لا تضف أي نص أو تفسيرات أخرى:

{{
  "score": 85.5,
  "language": "ar", 
  "matched_skills": [
    {{
      "name": "باثون",
      "confidence": 0.95,
      "category": "برمجة",
      "evidence": "مذكور في قسم الخبرة"
    }}
  ],
  "missing_skills": ["Kubernetes", "Docker"],
  "experience_summary": "ملخص واضح للخلفية المهنية...",
  "improvement_suggestions": [
    "قم بتحديد الإنجازات باستخدام المقاييس والأرقام",
    "أضف مهارات تقنية أكثر تحديداً",
    "قم بتضمين نتائج وتأثيرات المشاريع"
  ],
  "strengths": [
    "خلفية تقنية قوية في تطوير الواجهة الخلفية",
    "خبرة جيدة في قيادة المشاريع"
  ],
  "red_flags": [
    "فجوات التوظيف غير موضحة", 
    "ذكر محدود لإنجازات محددة"
  ]
}}

إرشادات التقييم:
- 100-90: تطابق ممتاز، مؤهل بشكل كبير
- 89-80: تطابق قوي، مؤهل جيداً
- 79-70: تطابق جيد، مؤهل بشكل عام  
- 69-60: تطابق مقبول، بعض الفجوات
- أقل من 60: تطابق ضعيف، فجوات كبيرة

كن دقيقاً وموضوعياً وقدم ملاحظات قابلة للتطبيق."""
```

---

## 🔧 Enhanced Configuration

### 4.3 Updated Prompts Configuration

#### `config/prompts.yaml`
```yaml
# AI Resume Analyzer - Prompt Templates
version: "1.0"
last_updated: "2024-01-15"

resume_analysis_en: |
  You are an expert resume analyst and career coach. Analyze the following resume and job description thoroughly.

  RESUME TEXT:
  {resume_text}

  JOB DESCRIPTION:
  {job_description}

  Provide comprehensive analysis in EXACT JSON format:
  {{
    "score": 85.5,
    "language": "en",
    "matched_skills": [
      {{
        "name": "Skill Name",
        "confidence": 0.95,
        "category": "Category",
        "evidence": "Where this skill was mentioned"
      }}
    ],
    "missing_skills": ["Skill1", "Skill2"],
    "experience_summary": "Clear professional background summary...",
    "improvement_suggestions": [
      "Actionable suggestion 1",
      "Actionable suggestion 2"
    ],
    "strengths": ["Strength 1", "Strength 2"],
    "red_flags": ["Issue 1", "Issue 2"],
    "overall_assessment": "Brief overall assessment..."
  }}

  Scoring Guidelines (be strict):
  - 90-100: Excellent match, highly qualified
  - 80-89: Strong match, well qualified  
  - 70-79: Good match, generally qualified
  - 60-69: Fair match, some gaps
  - Below 60: Poor match, significant gaps

  Be thorough, objective, and provide actionable feedback.

resume_analysis_fa: |
  شما یک متخصص تحلیل رزومه و مشاور شغلی هستید. رزومه و شرح شغل زیر را به طور کامل تحلیل کنید.

  متن رزومه:
  {resume_text}

  شرح شغل:
  {job_description}

  تحلیل جامع را در قالب JSON دقیق زیر ارائه دهید:
  {{
    "score": 85.5,
    "language": "fa",
    "matched_skills": [
      {{
        "name": "نام مهارت",
        "confidence": 0.95,
        "category": "دسته‌بندی", 
        "evidence": "محل ذکر این مهارت"
      }}
    ],
    "missing_skills": ["مهارت۱", "مهارت۲"],
    "experience_summary": "خلاصه واضح از سوابق حرفه‌ای...",
    "improvement_suggestions": [
      "پیشنهاد قابل اجرا ۱",
      "پیشنهاد قابل اجرا ۲"
    ],
    "strengths": ["قوت ۱", "قوت ۲"],
    "red_flags": ["مشکل ۱", "مشکل ۲"],
    "overall_assessment": "ارزیابی کلی مختصر..."
  }}

  دستورالعمل امتیازدهی (سخت‌گیرانه):
  - ۱۰۰-۹۰: تطابق عالی، بسیار واجد شرایط
  - ۸۹-۸۰: تطابق قوی، واجد شرایط
  - ۷۹-۷۰: تطابق خوب، عموماً واجد شرایط
  - ۶۹-۶۰: تطابق متوسط، برخی شکاف‌ها
  - زیر ۶۰: تطابق ضعیف، شکاف‌های قابل توجه

  کامل، عینی و بازخورد قابل اجرا ارائه دهید.

resume_analysis_ar: |
  أنت خبير في تحليل السير الذاتية ومستشار مهني. قم بتحليل السيرة الذاتية ووصف الوظيفة التاليين بدقة.

  نص السيرة الذاتية:
  {resume_text}

  وصف الوظيفة:
  {job_description}

  قدم تحليل شامل بتنسيق JSON الدقيق التالي:
  {{
    "score": 85.5,
    "language": "ar",
    "matched_skills": [
      {{
        "name": "اسم المهارة",
        "confidence": 0.95, 
        "category": "الفئة",
        "evidence": "مكان ذكر هذه المهارة"
      }}
    ],
    "missing_skills": ["المهارة١", "المهارة٢"],
    "experience_summary": "ملخص واضح للخلفية المهنية...",
    "improvement_suggestions": [
      "اقتراح قابل للتطبيق ١",
      "اقتراح قابل للتطبيق ٢"
    ],
    "strengths": ["نقطة قوة ١", "نقطة قوة ٢"],
    "red_flags": ["مشكلة ١", "مشكلة ٢"],
    "overall_assessment": "تقييم عام موجز..."
  }}

  إرشادات التقييم (كن صارمًا):
  - 100-90: تطابق ممتاز، مؤهل بشكل كبير
  - 89-80: تطابق قوي، مؤهل جيداً
  - 79-70: تطابق جيد، مؤهل بشكل عام
  - 69-60: تطابق مقبول، بعض الفجوات
  - أقل من 60: تطابق ضعيف، فجوات كبيرة

  كن دقيقاً وموضوعياً وقدم ملاحظات قابلة للتطبيق.

language_detection: |
  Detect the primary language of this text. Return only the language code (e.g., 'en', 'fa', 'ar'):
  {text}
```

---

## 🔄 Updated Celery Task with AI Integration

### 4.4 Enhanced Background Task

#### `src/app/workers/tasks.py` (Updated)
```python
from celery import current_task
from sqlalchemy.orm import Session
import time
import logging

from app.core.celery_app import celery_app
from app.database import SessionLocal
from app.models import ResumeAnalysis
from app.services.file_service import FileService
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name="analyze_resume_task")
def analyze_resume_task(self, analysis_id: str, file_content: bytes, file_name: str, job_description: str = None):
    db = SessionLocal()
    try:
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        if not analysis:
            logger.error(f"Analysis not found: {analysis_id}")
            return {"error": "Analysis not found"}
        
        # Step 1: Extract text from file
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 1, "total": 5, "status": "Extracting text from file..."}
        )
        
        try:
            raw_text = FileService.extract_text_from_file(file_content, file_name)
            analysis.raw_text = raw_text
            db.commit()
            logger.info(f"Text extraction completed for analysis {analysis_id}")
        except Exception as e:
            logger.error(f"Text extraction failed for {analysis_id}: {str(e)}")
            analysis.status = "failed"
            db.commit()
            return {"error": f"Text extraction failed: {str(e)}"}
        
        # Step 2: Detect language
        current_task.update_state(
            state="PROGRESS", 
            meta={"current": 2, "total": 5, "status": "Detecting language..."}
        )
        
        try:
            language = FileService.detect_language(raw_text)
            analysis.language = language
            db.commit()
            logger.info(f"Language detected: {language} for analysis {analysis_id}")
        except Exception as e:
            logger.error(f"Language detection failed for {analysis_id}: {str(e)}")
            analysis.status = "failed"
            db.commit()
            return {"error": f"Language detection failed: {str(e)}"}
        
        # Step 3: Analyze with AI
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 3, "total": 5, "status": "Analyzing content with AI..."}
        )
        
        try:
            # Use the actual LLM service for analysis
            import asyncio
            analysis_result = asyncio.run(
                llm_service.analyze_resume(
                    resume_text=raw_text,
                    job_description=job_description,
                    language=language
                )
            )
            analysis.analysis_report = analysis_result
            db.commit()
            logger.info(f"AI analysis completed for {analysis_id}")
        except Exception as e:
            logger.error(f"AI analysis failed for {analysis_id}: {str(e)}")
            analysis.status = "failed"
            db.commit()
            return {"error": f"AI analysis failed: {str(e)}"}
        
        # Step 4: Quality assurance
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 4, "total": 5, "status": "Performing quality checks..."}
        )
        
        try:
            # Validate analysis quality
            if not self._validate_analysis_quality(analysis_result):
                logger.warning(f"Analysis quality check failed for {analysis_id}")
                # Continue anyway, but log the issue
        except Exception as e:
            logger.warning(f"Quality check failed for {analysis_id}: {str(e)}")
        
        # Step 5: Finalize
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 5, "total": 5, "status": "Finalizing analysis..."}
        )
        
        analysis.status = "completed"
        from datetime import datetime
        analysis.completed_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Analysis completed successfully for {analysis_id}")
        
        return {
            "analysis_id": analysis_id,
            "status": "completed",
            "result": analysis.analysis_report
        }
        
    except Exception as e:
        logger.error(f"Task failed for {analysis_id}: {str(e)}")
        if 'analysis' in locals():
            analysis.status = "failed"
            db.commit()
        return {"error": f"Task failed: {str(e)}"}
    finally:
        db.close()
    
    def _validate_analysis_quality(self, analysis_result: dict) -> bool:
        """Validate the quality of AI analysis"""
        try:
            # Check if score is within reasonable range
            score = analysis_result.get('score', 0)
            if not (0 <= score <= 100):
                return False
            
            # Check if we have reasonable number of skills
            matched_skills = analysis_result.get('matched_skills', [])
            if len(matched_skills) == 0:
                return False
            
            # Check if suggestions are provided
            suggestions = analysis_result.get('improvement_suggestions', [])
            if len(suggestions) == 0:
                return False
            
            return True
            
        except Exception:
            return False
```

---

## 🧪 AI Service Testing

### 4.5 Comprehensive Test Suite

#### `tests/test_services/test_llm_service.py`
```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.services.llm_service import LLMService

@pytest.fixture
def llm_service():
    return LLMService()

@pytest.fixture
def sample_resume_text():
    return """
    John Doe
    Senior Backend Developer
    
    EXPERIENCE:
    - Senior Developer at Tech Corp (2020-Present)
    * Developed microservices using Python and FastAPI
    * Led a team of 5 developers
    * Improved system performance by 40%
    
    SKILLS:
    Python, FastAPI, PostgreSQL, Docker, AWS
    """

@pytest.mark.asyncio
async def test_analyze_resume_success(llm_service, sample_resume_text):
    """Test successful resume analysis"""
    
    # Mock Gemini response
    mock_response = Mock()
    mock_response.text = '''
    {
        "score": 85.5,
        "language": "en",
        "matched_skills": [
            {"name": "Python", "confidence": 0.95, "category": "Programming", "evidence": "Mentioned in skills section"},
            {"name": "FastAPI", "confidence": 0.85, "category": "Framework", "evidence": "Mentioned in experience section"}
        ],
        "missing_skills": ["Kubernetes", "Redis"],
        "experience_summary": "5+ years of backend development experience with team leadership",
        "improvement_suggestions": ["Add more metrics to quantify achievements", "Include specific project outcomes"],
        "strengths": ["Strong technical background", "Leadership experience"],
        "red_flags": ["No mention of testing practices"]
    }
    '''
    
    with patch.object(llm_service.model, 'generate_content', return_value=mock_response):
        result = await llm_service.analyze_resume(sample_resume_text)
        
        assert result['score'] == 85.5
        assert result['language'] == 'en'
        assert len(result['matched_skills']) > 0
        assert 'text_statistics' in result
        assert 'analysis_metadata' in result

@pytest.mark.asyncio
async def test_analyze_resume_multilingual(llm_service):
    """Test resume analysis in different languages"""
    
    # Test Persian resume
    persian_resume = """
    رزومه محمد رضایی
    توسعه‌دهنده ارشد بک‌اند
    
    سوابق کاری:
    - توسعه‌دهنده ارشد در شرکت فناوری (۱۴۰۰-اکنون)
    * توسعه سرویس‌های میکرو با پایتون و FastAPI
    * رهبری تیم ۵ نفره توسعه‌دهندگان
    """
    
    mock_response = Mock()
    mock_response.text = '{"score": 80, "language": "fa", "matched_skills": [], "missing_skills": [], "experience_summary": "", "improvement_suggestions": []}'
    
    with patch.object(llm_service.model, 'generate_content', return_value=mock_response):
        result = await llm_service.analyze_resume(persian_resume, language='fa')
        assert result['language'] == 'fa'

def test_parse_ai_response_valid(llm_service):
    """Test parsing valid AI response"""
    valid_response = '''
    {
        "score": 75.0,
        "language": "en", 
        "matched_skills": [{"name": "Python", "confidence": 0.9, "category": "Programming"}],
        "missing_skills": ["Docker"],
        "experience_summary": "Good experience",
        "improvement_suggestions": ["Add more details"]
    }
    '''
    
    result = llm_service._parse_ai_response(valid_response, 'en')
    assert result['score'] == 75.0
    assert len(result['matched_skills']) == 1

def test_parse_ai_response_invalid_json(llm_service):
    """Test parsing invalid JSON response"""
    invalid_response = "This is not JSON"
    
    with pytest.raises(ValueError):
        llm_service._parse_ai_response(invalid_response, 'en')
```

---

## 📊 Performance Optimization

### 4.6 Caching and Optimization

#### `src/app/services/cache_service.py`
```python
import redis
import json
import hashlib
from typing import Optional, Any
from app.core.config import settings

class CacheService:
    def __init__(self):
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.default_ttl = 3600  # 1 hour
    
    def _generate_cache_key(self, resume_text: str, job_description: str = None) -> str:
        """Generate unique cache key for analysis"""
        content = resume_text + (job_description or "")
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"resume_analysis:{content_hash}"
    
    def get_cached_analysis(self, resume_text: str, job_description: str = None) -> Optional[dict]:
        """Get cached analysis result"""
        cache_key = self._generate_cache_key(resume_text, job_description)
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        return None
    
    def set_cached_analysis(self, resume_text: str, job_description: str, result: dict):
        """Cache analysis result"""
        cache_key = self._generate_cache_key(resume_text, job_description)
        self.redis_client.setex(
            cache_key,
            self.default_ttl,
            json.dumps(result)
        )
    
    def invalidate_cache(self, resume_text: str, job_description: str = None):
        """Invalidate cached analysis"""
        cache_key = self._generate_cache_key(resume_text, job_description)
        self.redis_client.delete(cache_key)
```

---

## 🚀 Current Tasks

### 4.7 AI Integration Tasks

| Task | Owner | Status | Due Date |
|------|-------|--------|----------|
| Implement Gemini API integration | @ai1 | ⏳ Pending | 2024-01-31 |
| Create multilingual prompt templates | @ai1 | ⏳ Pending | 2024-02-01 |
| Build response validation system | @ai2 | ⏳ Pending | 2024-02-02 |
| Implement retry logic and error handling | @ai1 | ⏳ Pending | 2024-02-03 |
| Create AI service testing suite | @ai2 | ⏳ Pending | 2024-02-04 |
| Optimize performance with caching | @ai1 | ⏳ Pending | 2024-02-05 |
| Integrate AI service with Celery tasks | @ai2 | ⏳ Pending | 2024-02-06 |
| Validate multilingual accuracy | @ai1 | ⏳ Pending | 2024-02-07 |

---

## 📝 Completion Criteria

### Ready for Phase 5 When:
- [ ] Gemini API integration works reliably
- [ ] Multilingual analysis (English, Farsi, Arabic) functional
- [ ] Response validation handles edge cases properly
- [ ] Error handling and retry logic implemented
- [ ] Performance optimized with caching
- [ ] AI service tests with >85% coverage
- [ ] Analysis accuracy meets target metrics (>95% EN, >92% FA)
- [ ] Integration with existing Celery tasks complete

---

## 🔗 Related Documents

- [← Back to Main Project Hub](../MAIN.md)
- [→ Proceed to Phase 5: Deployment & DevOps](./phase5-deployment.md)
- [← Review Phase 3: Core Development](./phase3-development.md)

---