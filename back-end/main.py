from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, status, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from passlib.context import CryptContext
from typing import Optional, List, Dict
import motor.motor_asyncio
import jwt
import datetime
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from groq import Groq, APIError
import re
import logging
import asyncio
import aiofiles
import tempfile
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError
import cloudinary
import cloudinary.uploader

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"))
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://student-assistant-app-frontend.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Security configurations
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "hsgdter453cnhfgdt658ddlkdk*m54wq")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Configuration
USER_DATA_DIR = "/tmp"  # Use /tmp for temporary files on Vercel
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_PDF_PAGES = 50  # Reduced page limit
MAX_TEXT_LENGTH = 10000  # Max characters for Groq input
MAX_RESPONSE_LENGTH = 5000  # Max characters for API responses
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Cloudinary configuration
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# MongoDB client
client = None
db = None
users_collection = None
courses_collection = None
lectures_collection = None
questions_collection = None

# CORS headers
def get_cors_headers():
    return {
        "Access-Control-Allow-Origin": "https://student-assistant-app-frontend.vercel.app",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Expose-Headers": "*"
    }

# Check volume writability
def check_volume_writable():
    test_file = os.path.join(USER_DATA_DIR, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.debug(f"Volume {USER_DATA_DIR} is writable")
        return True
    except (OSError, PermissionError) as e:
        logger.error(f"Volume {USER_DATA_DIR} is not writable: {str(e)}")
        return False

def verify_environment_variables():
    required_vars = [
        "MONGODB_URI",
        "JWT_SECRET_KEY",
        "GROQ_API_KEY",
        "CLOUDINARY_CLOUD_NAME",
        "CLOUDINARY_API_KEY",
        "CLOUDINARY_API_SECRET"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        raise HTTPException(
            status_code=500,
            detail=f"Missing required environment variables: {', '.join(missing_vars)}"
        )

# MongoDB setup
async def init_mongodb():
    if not MONGODB_URI:
        logger.error("MONGODB_URI not set")
        raise HTTPException(status_code=500, detail="MONGODB_URI not configured")
    
    max_retries = 5
    retry_delay = 3
    
    for attempt in range(max_retries):
        try:
            client = motor.motor_asyncio.AsyncIOMotorClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=30000
            )
            
            await client.admin.command('ping')
            logger.info("MongoDB connected successfully")
            
            db = client.student_assistant
            await db.command('ping')
            logger.info("Database access verified")
            
            return client
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.warning(f"MongoDB connection attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"MongoDB connection failed after {max_retries} attempts")
                raise HTTPException(status_code=503, detail="MongoDB connection failed")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            logger.error(f"Unexpected MongoDB connection error: {str(e)}")
            raise HTTPException(status_code=500, detail="Database initialization failed")

# Startup event
@app.on_event("startup")
async def startup_event():
    global client, db, users_collection, courses_collection, lectures_collection, questions_collection
    
    try:
        verify_environment_variables()
        
        # Initialize MongoDB connection
        client = await init_mongodb()
        if not client:
            raise Exception("Failed to initialize MongoDB client")
            
        db = client.student_assistant
        users_collection = db.users
        courses_collection = db.courses
        lectures_collection = db.lectures
        questions_collection = db.questions
        
        # Test collection access
        try:
            await users_collection.find_one()
            logger.info("Database collections initialized successfully")
        except Exception as e:
            logger.error(f"Database access test failed: {str(e)}")
            raise Exception("Database connection test failed")
        
        # Create indexes
        try:
            await asyncio.gather(
                users_collection.create_index("username", unique=True),
                courses_collection.create_index([("username", 1), ("course_name", 1)], unique=True),
                lectures_collection.create_index([("username", 1), ("course_name", 1), ("lecture_name", 1)], unique=True),
                questions_collection.create_index("id", unique=True)
            )
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.error(f"Index creation failed: {str(e)}")
            # Don't fail startup for index issues
            
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Student Assistant API is running",
        "status": "healthy",
        "endpoints": {
            "login": "POST /login",
            "register": "POST /register",
            "courses": "GET /courses",
            "lectures": "GET /lectures/{course_name}"
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    try:
        if client:
            await client.admin.command('ping')
            db_status = "connected"
        else:
            db_status = "disconnected"
            
        return {
            "status": "healthy",
            "database": db_status,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# Input validation
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

def validate_name(name: str, field: str):
    if not name or not NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=400,
            detail=f"{field} must contain only letters, numbers, underscores, or hyphens"
        )

def cleanup_temp_file(temp_file_path: str):
    try:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Removed temp file: {temp_file_path}")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    details = "; ".join([f"{err['loc'][-1]}: {err['msg']}" for err in errors])
    logger.error(f"Validation error: {details}")
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": details},
        headers=get_cors_headers()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=get_cors_headers()
    )

@app.exception_handler(MemoryError)
async def memory_error_handler(request: Request, exc: MemoryError):
    logger.error(f"Memory error: {str(exc)}")
    return JSONResponse(
        status_code=507,
        content={"error": "Out of memory. Try a smaller file."},
        headers=get_cors_headers()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
        headers=get_cors_headers()
    )

# OPTIONS route
router = APIRouter()

@router.options("/{path:path}")
async def handle_options():
    logger.debug("Handling OPTIONS request")
    return Response(status_code=200, headers=get_cors_headers())

app.include_router(router)

# Pydantic models
class UserCredentials(BaseModel):
    username: str
    password: str

class CourseCreate(BaseModel):
    course_name: str

class LectureCreate(BaseModel):
    lecture_name: str
    course_name: str

class StudyRequest(BaseModel):
    task: str
    lecture_name: str
    question: Optional[str] = None

class ExamRequest(BaseModel):
    lecture_name: str
    exam_type: str
    difficulty: str

class AnswerSubmit(BaseModel):
    question_id: str
    answer: str

# MongoDB functions
async def get_user(username: str) -> Optional[Dict]:
    if users_collection is None:
        logger.error("Users collection not initialized")
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        user = await users_collection.find_one({"username": username})
        return user
    except Exception as e:
        logger.error(f"Error fetching user {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch user")

async def create_user(username: str, hashed_password: str):
    if users_collection is None:
        logger.error("Users collection not initialized")
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        validate_name(username, "Username")
        if await get_user(username):
            raise HTTPException(status_code=400, detail="Username exists")
        user = {"username": username, "hashed_password": hashed_password}
        await users_collection.insert_one(user)
        logger.info(f"User {username} created")
        return user
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Username exists")
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create user")

async def get_user_courses(username: str) -> List[str]:
    if courses_collection is None:
        logger.error("Courses collection not initialized")
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        courses = await courses_collection.find({"username": username}).to_list(None)
        return [course["course_name"] for course in courses]
    except Exception as e:
        logger.error(f"Error fetching courses for {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch courses")

async def create_course_db(username: str, course_name: str):
    if courses_collection is None:
        logger.error("Courses collection not initialized")
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        validate_name(course_name, "Course name")
        if await courses_collection.find_one({"username": username, "course_name": course_name}):
            raise HTTPException(status_code=400, detail="Course exists")
        await courses_collection.insert_one({"username": username, "course_name": course_name})
        logger.info(f"Course {course_name} created for {username}")
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Course exists")
    except Exception as e:
        logger.error(f"Error creating course {course_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create course")

async def get_user_lectures(username: str, course_name: str) -> List[Dict]:
    if lectures_collection is None:
        logger.error("Lectures collection not initialized")
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        lectures = await lectures_collection.find({"username": username, "course_name": course_name}).to_list(None)
        return [{"name": lec["lecture_name"], "path": lec["file_path"]} for lec in lectures]
    except Exception as e:
        logger.error(f"Error fetching lectures for {username}/{course_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch lectures")

async def create_lecture_db(username: str, course_name: str, lecture_name: str, file_url: str, lecture_text: str):
    if lectures_collection is None:
        logger.error("Lectures collection not initialized")
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        validate_name(lecture_name, "Lecture name")
        if await lectures_collection.find_one({"username": username, "course_name": course_name, "lecture_name": lecture_name}):
            raise HTTPException(status_code=400, detail="Lecture exists")
        lecture = {
            "username": username,
            "course_name": course_name,
            "lecture_name": lecture_name,
            "file_path": file_url,
            "lecture_text": lecture_text[:MAX_TEXT_LENGTH]  # Truncate text
        }
        await lectures_collection.insert_one(lecture)
        logger.info(f"Lecture {lecture_name} created for {username}/{course_name}")
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Lecture exists")
    except Exception as e:
        logger.error(f"Error creating lecture {lecture_name}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create lecture")

# Authentication functions
def hash_password(password: str) -> str:
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Error hashing password: {str(e)}")
        raise HTTPException(status_code=500, detail="Password hashing failed")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        raise HTTPException(status_code=500, detail="Password verification failed")

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        logger.debug(f"Authenticated user: {username}")
        return username
    except jwt.PyJWTError as e:
        logger.error(f"JWT error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

# File processing
def validate_pdf(file_path: str) -> None:
    logger.debug(f"Validating PDF: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            if reader.is_encrypted:
                raise HTTPException(status_code=400, detail="PDF is encrypted")
            if len(reader.pages) == 0:
                raise HTTPException(status_code=400, detail="PDF is empty")
            if len(reader.pages) > MAX_PDF_PAGES:
                raise HTTPException(status_code=400, detail=f"PDF exceeds {MAX_PDF_PAGES} pages")
            text = reader.pages[0].extract_text() or ""
            if not text.strip():
                raise HTTPException(status_code=400, detail="PDF has no extractable text")
    except PdfReadError:
        logger.error("Invalid PDF file")
        raise HTTPException(status_code=400, detail="Invalid PDF")
    except Exception as e:
        logger.error(f"PDF validation error: {str(e)}")
        raise HTTPException(status_code=500, detail="PDF validation failed")

async def extract_text_from_pdf(file_path: str, timeout: int = 60) -> str:
    logger.debug(f"Extracting text from PDF: {file_path}")
    try:
        await asyncio.wait_for(asyncio.to_thread(validate_pdf, file_path), timeout=30)
        text = []
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            total_pages = min(len(reader.pages), MAX_PDF_PAGES)
            logger.debug(f"PDF has {total_pages} pages")
            for page_num in range(total_pages):
                try:
                    page_text = reader.pages[page_num].extract_text() or ""
                    text.append(page_text)
                    logger.debug(f"Extracted text from page {page_num + 1} (length: {len(page_text)})")
                except Exception as e:
                    logger.warning(f"Page {page_num + 1} extraction failed: {str(e)}")
                    text.append("")
        full_text = "\n".join(text)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No extractable text in PDF")
        logger.debug(f"Total extracted text length: {len(full_text)}")
        return full_text[:MAX_TEXT_LENGTH]  # Truncate text
    except asyncio.TimeoutError:
        logger.error("PDF validation timed out")
        raise HTTPException(status_code=504, detail="PDF validation timed out")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="PDF processing failed")

async def upload_lecture_to_cloudinary(file_path: str, username: str, lecture_name: str) -> str:
    try:
        public_id = f"{username}/lectures/{lecture_name}"
        response = cloudinary.uploader.upload(
            file_path,
            public_id=public_id,
            resource_type="raw",
            format="pdf"
        )
        file_url = response['secure_url']
        logger.info(f"Uploaded {lecture_name} to Cloudinary: {file_url}")
        return file_url
    except Exception as e:
        logger.error(f"Cloudinary upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to upload to Cloudinary")

# Prompt templates
EXAM_PROMPT = """
Based on the following lecture content:
{text}
Create an exam with the specified parameters.
Difficulty level: {level}
Exam type: {exam_type}
If Exam type is "MCQs":
Generate 5 Multiple Choice Questions (MCQs) with options A-D.
Format:
**MCQs**
1. [Question]?
A) [Option1]
B) [Option2]
C) [Option3]
D) [Option4]
Answer: [Letter]
If Exam type is "Essay Questions":
Generate 5 Essay Questions.
Format:
**Essay Questions**
1. [Essay Question]
"""

GRADING_PROMPT = """
You are a helpful and educational Student Assistant.
Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {answer}
Please evaluate the student's answer and provide:
1. A score out of 10
2. Detailed feedback with examples or direct explanations of what was good and what could be improved
3. The correct answer or approach
Your response should be encouraging and educational with examples to help understand.
"""

STUDY_PROMPTS = {
    "Summarize": """
    Based on the following lecture content:
    {text}
    Create a comprehensive summary.
    Include all key concepts and important points.
    Use clear examples to explain difficult concepts.
    Provide the summary in a well-structured format with headings, bullet points, and examples.
    """,
    "Explain": """
    Based on the following lecture content:
    {text}
    Explain the content in detail. Break down all complex concepts
    and provide simple explanations with examples.
    Your explanation should be easy to understand for a student. Use analogies and examples
    where appropriate to clarify difficult concepts.
    """,
    "Examples": """
    Based on the following lecture content:
    {text}
    Create practical examples.
    Provide at least 5 different examples that illustrate the concepts.
    Each example should demonstrate a different aspect of the material.
    """,
    "Custom Question": """
    Based on the following lecture content:
    {text}
    Answer this specific question:
    {question}
    Provide a thorough answer with examples and explanations.
    """
}

async def parse_exam(exam_text: str, exam_type: str, lecture_name: str) -> List[Dict]:
    if questions_collection is None:
        logger.error("Questions collection not initialized")
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        mcqs = []
        essays = []
        lines = exam_text.replace('\r\n', '\n').strip().split('\n')
        current_section = None
        current_question = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('**MCQs**'):
                current_section = 'mcqs'
                continue
            elif line.startswith('**Essay Questions**'):
                if current_question:
                    if current_section == 'mcqs':
                        mcqs.append('\n'.join(current_question))
                    current_question = []
                current_section = 'essays'
                continue
            if current_section == 'mcqs':
                if re.match(r"^\d+\.\s", line):
                    if current_question:
                        mcqs.append('\n'.join(current_question))
                        current_question = []
                    current_question.append(line)
                elif line.startswith(('A)', 'B)', 'C)', 'D)', 'Answer:')):
                    current_question.append(line)
            elif current_section == 'essays':
                if re.match(r"^\d+\.\s", line):
                    if current_question:
                        essays.append('\n'.join(current_question))
                        current_question = []
                    current_question.append(line)
        if current_question:
            if current_section == 'mcqs':
                mcqs.append('\n'.join(current_question))
            elif current_section == 'essays':
                essays.append('\n'.join(current_question))
        flattened = []
        if exam_type == "MCQs":
            for idx, q in enumerate(mcqs[:5]):  # Limit to 5 questions
                lines = q.split('\n')
                question_text = next((line for line in lines if re.match(r"^\d+\.\s", line)), "")
                options = [line for line in lines if re.match(r"^[A-D]\)", line)]
                answer_line = next((line for line in lines if line.startswith("Answer:")), "")
                answer = answer_line.replace("Answer:", "").strip() if answer_line else ""
                question_id = f"mcq_{lecture_name}_{idx}"
                question = {
                    "id": question_id,
                    "lecture_name": lecture_name,
                    "question": question_text,
                    "type": "mcq",
                    "options": options,
                    "correct_answer": answer
                }
                flattened.append(question)
                await questions_collection.update_one({"id": question_id}, {"$set": question}, upsert=True)
        elif exam_type == "Essay Questions":
            for idx, q in enumerate(essays[:5]):  # Limit to 5 questions
                question_id = f"essay_{lecture_name}_{idx}"
                question = {
                    "id": question_id,
                    "lecture_name": lecture_name,
                    "question": q,
                    "type": "essay",
                    "options": [],
                    "correct_answer": ""
                }
                flattened.append(question)
                await questions_collection.update_one({"id": question_id}, {"$set": question}, upsert=True)
        logger.info(f"Parsed {len(flattened)} questions for {lecture_name}")
        return flattened
    except Exception as e:
        logger.error(f"Exam parsing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not parse exam")

# Initialize Groq client
def get_groq_client():
    try:
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY not set in environment")
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
        
        stripped_key = GROQ_API_KEY.strip()
        if not stripped_key.startswith("gsk_") or len(stripped_key) < 20:
            logger.error(f"Invalid GROQ_API_KEY format: {stripped_key[:5]}...")
            raise HTTPException(status_code=500, detail="Invalid GROQ_API_KEY format")
        
        logger.debug(f"Attempting to use GROQ_API_KEY: {stripped_key[:5]}...{stripped_key[-5:]}")
        
        client = Groq(api_key=stripped_key)
        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            logger.info(f"GROQ_API_KEY validated successfully: {response.id}")
        except APIError as e:
            logger.error(f"GROQ_API_KEY validation failed: {str(e)}. Response: {getattr(e, 'response', 'No response')}")
            raise HTTPException(status_code=503, detail=f"Invalid GROQ_API_KEY: {str(e)}")
        
        runtime_key = os.getenv("GROQ_API_KEY", "").strip()
        if runtime_key != stripped_key:
            logger.error(f"Environment GROQ_API_KEY mismatch. Loaded: {stripped_key[:5]}..., Runtime: {runtime_key[:5]}...")
            raise HTTPException(status_code=500, detail="GROQ_API_KEY environment mismatch")
        
        return client
    except APIError as e:
        logger.error(f"Groq API error: {str(e)}. Response: {getattr(e, 'response', 'No response')}")
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {str(e)}", exc_info=True)
        raise HTTPException(status_code=503, detail="AI service unavailable")

# API Endpoints
@app.post("/register", response_model=dict)
async def register(credentials: UserCredentials):
    if not credentials.username or not credentials.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    hashed_password = hash_password(credentials.password)
    try:
        await create_user(credentials.username, hashed_password)
        access_token = create_access_token(
            data={"sub": credentials.username},
            expires_delta=datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        logger.info(f"User {credentials.username} registered successfully")
        return JSONResponse(
            content={"message": "Registered successfully", "token": access_token},
            headers=get_cors_headers()
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Registration failed")
        
@app.post("/login", response_model=dict)
async def login(credentials: UserCredentials):
    try:
        if not client:
            raise HTTPException(status_code=503, detail="Database not initialized")
        if not credentials.username or not credentials.password:
            raise HTTPException(status_code=400, detail="Username and password required")
        user = await users_collection.find_one({"username": credentials.username})
        if not user:
            logger.warning(f"Login attempt for non-existent user: {credentials.username}")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not verify_password(credentials.password, user["hashed_password"]):
            logger.warning(f"Invalid password for user: {credentials.username}")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        access_token = create_access_token(
            data={"sub": credentials.username},
            expires_delta=datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        logger.info(f"User {credentials.username} logged in successfully")
        return JSONResponse(
            content={"token": access_token},
            headers=get_cors_headers()
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/profile", response_model=dict)
async def get_profile(username: str = Depends(get_current_user)):
    try:
        courses = await get_user_courses(username)
        profile = {"username": username, "courses": []}
        for course_name in courses:
            lectures = await get_user_lectures(username, course_name)
            profile["courses"].append({
                "course_name": course_name,
                "lectures": [lec["name"] for lec in lectures]
            })
        logger.info(f"Profile retrieved for {username}")
        return JSONResponse(
            content={"profile": profile},
            headers=get_cors_headers()
        )
    except Exception as e:
        logger.error(f"Profile retrieval error for {username}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve profile")

@app.post("/courses", response_model=dict)
async def create_course(course: CourseCreate, username: str = Depends(get_current_user)):
    try:
        await create_course_db(username, course.course_name)
        logger.info(f"Course {course.course_name} created for {username}")
        return JSONResponse(
            content={"message": f"Course '{course.course_name}' created"},
            headers=get_cors_headers()
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Course creation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not create course")

@app.get("/courses", response_model=dict)
async def list_courses(username: str = Depends(get_current_user)):
    try:
        courses = await get_user_courses(username)
        logger.info(f"Courses retrieved for {username}")
        return JSONResponse(
            content={"courses": courses},
            headers=get_cors_headers()
        )
    except Exception as e:
        logger.error(f"Courses retrieval error for {username}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve courses")

@app.post("/lectures", response_model=dict)
async def upload_lecture(
    lecture_name: str = Form(...),
    course_name: str = Form(...),
    file: UploadFile = File(...),
    username: str = Depends(get_current_user)
):
    logger.info(f"Upload request: lecture={lecture_name}, course={course_name}, file={file.filename}, size={file.size}")
    temp_file_path = None

    try:
        if not check_volume_writable():
            raise HTTPException(status_code=500, detail="Storage not writable")

        validate_name(lecture_name, "Lecture name")
        validate_name(course_name, "Course name")
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        file_size = file.size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE/1024/1024}MB")
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        courses = await get_user_courses(username)
        if course_name not in courses:
            raise HTTPException(status_code=404, detail="Course not found")

        if await lectures_collection.find_one({"username": username, "course_name": course_name, "lecture_name": lecture_name}):
            raise HTTPException(status_code=400, detail="Lecture exists")

        with tempfile.NamedTemporaryFile(delete=False, dir=USER_DATA_DIR, suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            logger.debug(f"Saving temp PDF: {temp_file_path}")
            async with aiofiles.open(temp_file_path, 'wb') as f:
                total_bytes = 0
                while chunk := await file.read(8192):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_FILE_SIZE:
                        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE/1024/1024}MB")
                    await f.write(chunk)

        lecture_text = await extract_text_from_pdf(temp_file_path)
        file_url = await upload_lecture_to_cloudinary(temp_file_path, username, lecture_name)
        cleanup_temp_file(temp_file_path)
        await create_lecture_db(username, course_name, lecture_name, file_url, lecture_text)
        logger.info(f"Lecture '{lecture_name}' uploaded successfully for {username}/{course_name}")
        response = JSONResponse(
            content={"message": f"Lecture '{lecture_name}' uploaded"},
            headers=get_cors_headers()
        )
        logger.debug(f"Response headers: {response.headers}")
        return response
    except HTTPException as he:
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)
        raise he
    except MemoryError:
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)
        logger.error("Memory error during upload")
        raise HTTPException(status_code=507, detail="Out of memory. Try a smaller file.")
    except OSError as e:
        logger.error(f"File operation error: {str(e)}")
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail="File operation failed")
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_temp_file(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/lectures/{course_name}", response_model=dict)
async def list_lectures(course_name: str, username: str = Depends(get_current_user)):
    try:
        courses = await get_user_courses(username)
        if course_name not in courses:
            raise HTTPException(status_code=404, detail="Course not found")
        lectures = await get_user_lectures(username, course_name)
        logger.info(f"Lectures retrieved for {username}/{course_name}")
        return JSONResponse(
            content={"lectures": [lec["name"] for lec in lectures]},
            headers=get_cors_headers()
        )
    except Exception as e:
        logger.error(f"Lectures retrieval error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve lectures")

@app.post("/study", response_model=dict)
async def generate_study_content(request: StudyRequest, username: str = Depends(get_current_user)):
    logger.debug(f"Study request: task={request.task}, lecture={request.lecture_name}, user={username}")
    try:
        lecture = await lectures_collection.find_one({"username": username, "lecture_name": request.lecture_name})
        if not lecture:
            logger.error(f"Lecture {request.lecture_name} not found for {username}")
            raise HTTPException(status_code=404, detail="Lecture not found")
        if not lecture.get("lecture_text"):
            logger.error(f"No text found for lecture {request.lecture_name}")
            raise HTTPException(status_code=400, detail="No lecture text available")
        if request.task == "Custom Question" and not request.question:
            logger.error("Custom Question task requires a question")
            raise HTTPException(status_code=400, detail="Question required")

        groq_client = get_groq_client()
        prompt_text = STUDY_PROMPTS[request.task].format(
            text=lecture["lecture_text"],
            question=request.question or ""
        )
        logger.debug(f"Prompt length: {len(prompt_text)} characters")
        
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    groq_client.chat.completions.create,
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=256,
                    temperature=0.7
                ),
                timeout=30
            )
            content = response.choices[0].message.content[:MAX_RESPONSE_LENGTH]  # Truncate response
            logger.info(f"Study content generated for {username}/{request.lecture_name}/{request.task}")
            return JSONResponse(
                content={"content": content},
                headers=get_cors_headers()
            )
        except asyncio.TimeoutError:
            logger.error(f"Groq API timed out for {request.task}")
            raise HTTPException(status_code=504, detail="AI processing timed out")
        except APIError as e:
            logger.error(f"Groq API error for {request.task}: {str(e)}")
            raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Study content generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not generate study content")

@app.post("/exam", response_model=dict)
async def generate_exam(request: ExamRequest, username: str = Depends(get_current_user)):
    logger.debug(f"Exam request: lecture={request.lecture_name}, type={request.exam_type}, difficulty={request.difficulty}, user={username}")
    try:
        lecture = await lectures_collection.find_one({"username": username, "lecture_name": request.lecture_name})
        if not lecture:
            logger.error(f"Lecture {request.lecture_name} not found for {username}")
            raise HTTPException(status_code=404, detail="Lecture not found")
        if not lecture.get("lecture_text"):
            logger.error(f"No text found for lecture {request.lecture_name}")
            raise HTTPException(status_code=400, detail="No lecture text available")

        groq_client = get_groq_client()
        prompt_text = EXAM_PROMPT.format(
            text=lecture["lecture_text"],
            level=request.difficulty,
            exam_type=request.exam_type
        )
        logger.debug(f"Prompt length: {len(prompt_text)} characters")
        
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    groq_client.chat.completions.create,
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=256,
                    temperature=0.7
                ),
                timeout=30
            )
            questions = await parse_exam(response.choices[0].message.content, request.exam_type, request.lecture_name)
            logger.info(f"Exam generated for {username}/{request.lecture_name}/{request.exam_type}")
            return JSONResponse(
                content={"questions": questions},
                headers=get_cors_headers()
            )
        except asyncio.TimeoutError:
            logger.error(f"Groq API timed out for exam generation")
            raise HTTPException(status_code=504, detail="AI processing timed out")
        except APIError as e:
            logger.error(f"Groq API error for exam: {str(e)}")
            raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Exam generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not generate exam")

@app.post("/exam/grade", response_model=dict)
async def grade_answer_endpoint(answer: AnswerSubmit, username: str = Depends(get_current_user)):
    logger.debug(f"Grade request: question_id={answer.question_id}, user={username}")
    try:
        question = await questions_collection.find_one({"id": answer.question_id})
        if not question:
            logger.error(f"Question {answer.question_id} not found")
            raise HTTPException(status_code=404, detail="Question not found")

        groq_client = get_groq_client()
        prompt_text = GRADING_PROMPT.format(
            question=question["question"],
            answer=answer.answer,
            correct_answer=question["correct_answer"] if question["type"] == "mcq" else "No predefined answer"
        )
        logger.debug(f"Prompt length: {len(prompt_text)} characters")
        
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    groq_client.chat.completions.create,
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=256,
                    temperature=0.7
                ),
                timeout=30
            )
            feedback = response.choices[0].message.content[:MAX_RESPONSE_LENGTH]  # Truncate response
            logger.info(f"Answer graded for {username}/{question['lecture_name']}/{answer.question_id}")
            return JSONResponse(
                content={"feedback": feedback},
                headers=get_cors_headers()
            )
        except asyncio.TimeoutError:
            logger.error("Groq API timed out for grading")
            raise HTTPException(status_code=504, detail="AI processing timed out")
        except APIError as e:
            logger.error(f"Groq API error for grading: {str(e)}")
            raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Grading error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not grade answer")

@app.get("/health")
async def health_check():
    try:
        if client:
            await client.admin.command('ping')
        else:
            raise HTTPException(status_code=500, detail="MongoDB not initialized")
        volume_writable = check_volume_writable()
        logger.info("Health check completed")
        return JSONResponse(
            content={
                "status": "healthy" if volume_writable else "unhealthy",
                "mongodb": "connected" if client else "disconnected",
                "volume_writable": volume_writable
            },
            headers=get_cors_headers()
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Health check failed")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
