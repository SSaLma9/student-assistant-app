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
import PyPDF2
import os
import shutil
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pathlib import Path
import re
import logging
import json
import time
import gc
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, DuplicateKeyError
import psutil
import asyncio
import aiofiles
import tempfile
import uuid

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://student-assistant-frontend-production.up.railway.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configurations
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "hsgdter453cnhfgdt658ddlkdk*m54wq")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Configuration
USER_DATA_DIR = "/app/user_data"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
MAX_PDF_PAGES = 50  # Limit PDF pages
os.makedirs(USER_DATA_DIR, exist_ok=True)

# MongoDB client
client = None
db = None
users_collection = None
courses_collection = None
lectures_collection = None
questions_collection = None

# Utility function for CORS headers
def get_cors_headers():
    return {
        "Access-Control-Allow-Origin": "https://student-assistant-frontend-production.up.railway.app",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    }

# Check volume writability
def check_volume_writable():
    test_file = os.path.join(USER_DATA_DIR, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.info(f"Volume {USER_DATA_DIR} is writable")
        return True
    except (OSError, PermissionError) as e:
        logger.error(f"Volume {USER_DATA_DIR} is not writable: {str(e)}")
        return False

# Check disk space
def check_disk_space():
    disk = psutil.disk_usage(USER_DATA_DIR)
    if disk.percent > 90:
        logger.error(f"Disk usage too high: {disk.percent}%")
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="Insufficient disk space"
        )
    return disk

# Lazy initialize embedding model
embeddings_model = None
def get_embeddings_model():
    global embeddings_model
    if embeddings_model is None:
        logger.info("Initializing HuggingFaceEmbeddings")
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    embeddings_model = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True, 'batch_size': 4}
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep((attempt + 1) * 5)
            if embeddings_model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="AI service unavailable"
                )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service unavailable"
            )
    return embeddings_model

# MongoDB setup
async def init_mongodb():
    if not MONGODB_URI:
        logger.error("MONGODB_URI not set")
        raise HTTPException(status_code=500, detail="MONGODB_URI not configured")
    max_retries = 5
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            client = motor.motor_asyncio.AsyncIOMotorClient(
                MONGODB_URI, serverSelectionTimeoutMS=5000
            )
            await client.admin.command('ping')
            logger.info("MongoDB connected")
            return client
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            if attempt == max_retries - 1:
                logger.error(f"MongoDB failed after {max_retries} attempts: {str(e)}")
                raise HTTPException(status_code=503, detail="MongoDB connection failed")
            logger.warning(f"MongoDB attempt {attempt + 1} failed. Retrying...")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2

# Startup event
@app.on_event("startup")
async def startup_event():
    global client, db, users_collection, courses_collection, lectures_collection, questions_collection
    try:
        client = await init_mongodb()
        db = client.student_assistant
        users_collection = db.users
        courses_collection = db.courses
        lectures_collection = db.lectures
        questions_collection = db.questions
        logger.info("MongoDB initialized")
        
        # Create indexes with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await users_collection.create_index("username", unique=True)
                await courses_collection.create_index([("username", 1), ("course_name", 1)], unique=True)
                await lectures_collection.create_index([("username", 1), ("course_name", 1), ("lecture_name", 1)], unique=True)
                await questions_collection.create_index("id", unique=True)
                logger.info("MongoDB indexes created")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Index creation failed: {str(e)}")
                    raise HTTPException(status_code=500, detail="Index creation failed")
                await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"MongoDB initialization failed: {str(e)}")
        raise HTTPException(status_code=500, detail="MongoDB initialization failed")

# Input validation
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

def validate_name(name: str, field: str):
    if not name or not NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=400,
            detail=f"{field} must contain only letters, numbers, underscores, or hyphens"
        )

def check_memory_usage():
    mem = psutil.virtual_memory()
    logger.info(f"Memory: {mem.percent}% (Total: {mem.total/1024/1024:.2f}MB)")
    if mem.percent > 85:
        raise HTTPException(status_code=507, detail="Memory overloaded")

def cleanup_lecture_files(lecture_path: str, faiss_path: str):
    try:
        if lecture_path and os.path.exists(lecture_path):
            os.remove(lecture_path)
            logger.info(f"Removed lecture file: {lecture_path}")
        if faiss_path and os.path.exists(faiss_path):
            shutil.rmtree(faiss_path)
            logger.info(f"Removed FAISS index: {faiss_path}")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    details = "; ".join([f"{err['loc'][-1]}: {err['msg']}" for err in errors])
    return JSONResponse(
        status_code=422,
        content={"error": "Validation error", "details": details},
        headers=get_cors_headers()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
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
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        user = await users_collection.find_one({"username": username})
        return user
    except Exception as e:
        logger.error(f"Error fetching user {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch user")

async def create_user(username: str, hashed_password: str):
    if users_collection is None:
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
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        courses = await courses_collection.find({"username": username}).to_list(None)
        return [course["course_name"] for course in courses]
    except Exception as e:
        logger.error(f"Error fetching courses: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch courses")

async def create_course_db(username: str, course_name: str):
    if courses_collection is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        validate_name(course_name, "Course name")
        if await courses_collection.find_one({"username": username, "course_name": course_name}):
            raise HTTPException(status_code=400, detail="Course exists")
        await courses_collection.insert_one({"username": username, "course_name": course_name})
        logger.info(f"Course {course_name} created")
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Course exists")
    except Exception as e:
        logger.error(f"Error creating course: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create course")

async def get_user_lectures(username: str, course_name: str) -> List[Dict]:
    if lectures_collection is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        lectures = await lectures_collection.find({"username": username, "course_name": course_name}).to_list(None)
        return [{"name": lec["lecture_name"], "path": lec["file_path"]} for lec in lectures]
    except Exception as e:
        logger.error(f"Error fetching lectures: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not fetch lectures")

async def create_lecture_db(username: str, course_name: str, lecture_name: str, file_path: str):
    if lectures_collection is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        validate_name(lecture_name, "Lecture name")
        if await lectures_collection.find_one({"username": username, "course_name": course_name, "lecture_name": lecture_name}):
            raise HTTPException(status_code=400, detail="Lecture exists")
        lecture = {
            "username": username,
            "course_name": course_name,
            "lecture_name": lecture_name,
            "file_path": file_path
        }
        await lectures_collection.insert_one(lecture)
        logger.info(f"Lecture {lecture_name} created")
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Lecture exists")
    except Exception as e:
        logger.error(f"Error creating lecture: {str(e)}")
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
    except jwt.PyJWTError as e:
        logger.error(f"JWT error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return username

# AI and file processing
async def create_faiss_index(text: str, timeout: int = 300) -> FAISS:
    logger.info("Creating FAISS index")
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text for indexing")
        check_memory_usage()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = text_splitter.split_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text chunks")
        embeddings = get_embeddings_model()
        batch_size = 2

        async def embed_with_timeout():
            vectors = []
            for i in range(0, len(chunks), batch_size):
                check_memory_usage()
                batch_chunks = chunks[i:i + batch_size]
                batch_vectors = await asyncio.to_thread(embeddings.embed_documents, batch_chunks)
                vectors.extend(batch_vectors)
                del batch_chunks, batch_vectors
                gc.collect()
            return vectors

        vectors = await asyncio.wait_for(embed_with_timeout(), timeout=timeout)
        text_embeddings = list(zip(chunks, vectors))
        faiss_index = await asyncio.to_thread(FAISS.from_embeddings, text_embeddings, embeddings)
        logger.info("FAISS index created")
        return faiss_index
    except asyncio.TimeoutError:
        logger.error("FAISS indexing timed out")
        raise HTTPException(status_code=504, detail="FAISS indexing timed out")
    except MemoryError:
        logger.error("Memory error in FAISS")
        raise HTTPException(status_code=507, detail="Out of memory")
    except Exception as e:
        logger.error(f"FAISS error: {str(e)}")
        raise HTTPException(status_code=500, detail="FAISS creation failed")

def initialize_rag_chain(username: str, lecture_name: str) -> RetrievalQA:
    try:
        check_memory_usage()
        faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
        if not os.path.exists(faiss_path):
            raise HTTPException(status_code=404, detail="Lecture index not found")
        embeddings = get_embeddings_model()
        vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192",
            max_tokens=512
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
    except Exception as e:
        logger.error(f"RAG chain error: {str(e)}")
        raise HTTPException(status_code=503, detail="AI service unavailable")

def validate_pdf(file_path: str) -> None:
    logger.info(f"Validating PDF: {file_path}")
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
        raise HTTPException(status_code=400, detail="Invalid PDF")
    except Exception as e:
        logger.error(f"PDF validation error: {str(e)}")
        raise HTTPException(status_code=500, detail="PDF validation failed")

async def extract_text_from_pdf(file_path: str, username: str, lecture_name: str, timeout: int = 300) -> str:
    logger.info(f"Processing PDF: {file_path}")
    faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
    try:
        await asyncio.wait_for(asyncio.to_thread(validate_pdf, file_path), timeout=30)
        text = []
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            total_pages = min(len(reader.pages), MAX_PDF_PAGES)
            logger.info(f"PDF has {total_pages} pages")
            for page_num in range(total_pages):
                if page_num % 5 == 0:
                    check_memory_usage()
                    gc.collect()
                try:
                    text.append(reader.pages[page_num].extract_text() or "")
                except Exception as e:
                    logger.warning(f"Page {page_num+1} extraction failed: {str(e)}")
                    text.append("")
        full_text = "\n".join(text)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No extractable text")
        faiss_index = await create_faiss_index(full_text, timeout=timeout)
        os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
        await asyncio.to_thread(faiss_index.save_local, faiss_path)
        logger.info(f"FAISS index saved: {faiss_path}")
        return full_text
    except asyncio.TimeoutError:
        cleanup_lecture_files(file_path, faiss_path)
        raise HTTPException(status_code=504, detail="PDF validation timed out")
    except HTTPException as he:
        cleanup_lecture_files(file_path, faiss_path)
        raise he
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        cleanup_lecture_files(file_path, faiss_path)
        raise HTTPException(status_code=500, detail="PDF processing failed")

# Prompt templates
EXAM_PROMPT = PromptTemplate(
    input_variables=["text", "level", "exam_type"],
    template="""
    Create an exam based on the provided lecture content.
    Difficulty level: {level}
    Exam type: {exam_type}
    If Exam type is "MCQs":
    Generate 10 Multiple Choice Questions (MCQs) with options A-D.
    Format:
    **MCQs**
    1. [Question]?
    A) [Option1]
    B) [Option2]
    C) [Option3]
    D) [Option4]
    Answer: [Letter]
    If Exam type is "Essay Questions":
    Generate 10 Essay Questions.
    Format:
    **Essay Questions**
    1. [Essay Question]
    """
)

GRADING_PROMPT = PromptTemplate(
    input_variables=["question", "answer", "correct_answer"],
    template="""
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
)

STUDY_PROMPTS = {
    "Summarize": PromptTemplate(
        input_variables=["text"],
        template="""
        Create a comprehensive summary of the provided lecture content.
        Include all key concepts and important points.
        Use clear examples to explain difficult concepts.
        Provide the summary in a well-structured format with headings, bullet points, and examples.
        """
    ),
    "Explain": PromptTemplate(
        input_variables=["text"],
        template="""
        Explain the provided lecture content in detail. Break down all complex concepts
        and provide simple explanations with examples.
        Your explanation should be easy to understand for a student. Use analogies and examples
        where appropriate to clarify difficult concepts.
        """
    ),
    "Examples": PromptTemplate(
        input_variables=["text"],
        template="""
        Create practical examples based on the provided lecture content.
        Provide at least 5 different examples that illustrate the concepts.
        Each example should demonstrate a different aspect of the material.
        """
    ),
    "Custom Question": PromptTemplate(
        input_variables=["text", "question"],
        template="""
        Answer this specific question based on the provided lecture content:
        {question}
        Provide a thorough answer with examples and explanations.
        """
    )
}

async def parse_exam(exam_text: str, exam_type: str, lecture_name: str) -> List[Dict]:
    if questions_collection is None:
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
            for idx, q in enumerate(mcqs):
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
            for idx, q in enumerate(essays):
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
        return flattened
    except Exception as e:
        logger.error(f"Exam parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not parse exam")

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
        return {"message": "Registered successfully", "token": access_token}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/login", response_model=dict)
async def login(credentials: UserCredentials):
    if not credentials.username or not credentials.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    try:
        user = await get_user(credentials.username)
        if not user or not verify_password(credentials.password, user["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        access_token = create_access_token(
            data={"sub": credentials.username},
            expires_delta=datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        return {"token": access_token}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
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
        return {"profile": profile}
    except Exception as e:
        logger.error(f"Profile error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve profile")

@app.post("/courses", response_model=dict)
async def create_course(course: CourseCreate, username: str = Depends(get_current_user)):
    try:
        await create_course_db(username, course.course_name)
        return {"message": f"Course '{course.course_name}' created"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Course creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not create course")

@app.get("/courses", response_model=dict)
async def list_courses(username: str = Depends(get_current_user)):
    try:
        courses = await get_user_courses(username)
        return {"courses": courses}
    except Exception as e:
        logger.error(f"Courses retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve courses")

@app.post("/lectures", response_model=dict)
async def upload_lecture(
    lecture_name: str = Form(...),
    course_name: str = Form(...),
    file: UploadFile = File(...),
    username: str = Depends(get_current_user)
):
    logger.info(f"Upload request: lecture={lecture_name}, course={course_name}, file={file.filename}, size={file.size}")
    lecture_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}.pdf")
    faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
    temp_file_path = None

    try:
        # Resource checks
        check_memory_usage()
        check_disk_space()
        if not check_volume_writable():
            raise HTTPException(status_code=500, detail="Storage not writable")

        # Validate inputs
        validate_name(lecture_name, "Lecture name")
        validate_name(course_name, "Course name")
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files allowed")

        # Check file size
        file_size = file.size
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE/1024/1024}MB")
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Verify course exists
        courses = await get_user_courses(username)
        if course_name not in courses:
            raise HTTPException(status_code=404, detail="Course not found")

        # Check if lecture exists
        if await lectures_collection.find_one({"username": username, "course_name": course_name, "lecture_name": lecture_name}):
            raise HTTPException(status_code=400, detail="Lecture exists")

        # Save file to temporary location
        os.makedirs(os.path.dirname(lecture_path), exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(lecture_path), suffix=".pdf") as temp_file:
            temp_file_path = temp_file.name
            logger.info(f"Saving temp PDF: {temp_file_path}")
            async with aiofiles.open(temp_file_path, 'wb') as f:
                total_bytes = 0
                while chunk := await file.read(8192):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_FILE_SIZE:
                        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE/1024/1024}MB")
                    await f.write(chunk)

        # Process PDF and create FAISS index
        try:
            lecture_text = await extract_text_from_pdf(temp_file_path, username, lecture_name, timeout=120)  # Reduced timeout
            os.rename(temp_file_path, lecture_path)
            logger.info(f"Renamed to: {lecture_path}")
            await create_lecture_db(username, course_name, lecture_name, lecture_path)
            logger.info(f"Lecture '{lecture_name}' uploaded successfully")
            return {"message": f"Lecture '{lecture_name}' uploaded"}
        except Exception as e:
            logger.error(f"Error processing lecture: {str(e)}")
            raise
    except HTTPException as he:
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_lecture_files(temp_file_path, faiss_path)
        raise he
    except MemoryError:
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_lecture_files(temp_file_path, faiss_path)
        raise HTTPException(status_code=507, detail="Out of memory. Try a smaller file.")
    except OSError as e:
        logger.error(f"File operation error: {str(e)}")
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_lecture_files(temp_file_path, faiss_path)
        raise HTTPException(status_code=500, detail="File operation failed")
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        if temp_file_path and os.path.exists(temp_file_path):
            cleanup_lecture_files(temp_file_path, faiss_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/lectures/{course_name}", response_model=dict)
async def list_lectures(course_name: str, username: str = Depends(get_current_user)):
    try:
        courses = await get_user_courses(username)
        if course_name not in courses:
            raise HTTPException(status_code=404, detail="Course not found")
        lectures = await get_user_lectures(username, course_name)
        return {"lectures": [lec["name"] for lec in lectures]}
    except Exception as e:
        logger.error(f"Lectures retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve lectures")

@app.post("/study", response_model=dict)
async def generate_study_content(request: StudyRequest, username: str = Depends(get_current_user)):
    try:
        lecture = await lectures_collection.find_one({"username": username, "lecture_name": request.lecture_name})
        if not lecture:
            raise HTTPException(status_code=404, detail="Lecture not found")
        rag_chain = initialize_rag_chain(username, request.lecture_name)
        if request.task == "Custom Question" and not request.question:
            raise HTTPException(status_code=400, detail="Question required")
        query = STUDY_PROMPTS[request.task].format(text="", question=request.question or "")
        response = rag_chain.invoke({"query": query})
        return {"content": response["result"]}
    except Exception as e:
        logger.error(f"Study content error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not generate study content")

@app.post("/exam", response_model=dict)
async def generate_exam(request: ExamRequest, username: str = Depends(get_current_user)):
    try:
        lecture = await lectures_collection.find_one({"username": username, "lecture_name": request.lecture_name})
        if not lecture:
            raise HTTPException(status_code=404, detail="Lecture not found")
        rag_chain = initialize_rag_chain(username, request.lecture_name)
        prompt = EXAM_PROMPT.format(text="", level=request.difficulty, exam_type=request.exam_type)
        response = rag_chain.invoke({"query": prompt})
        questions = await parse_exam(response["result"], request.exam_type, request.lecture_name)
        return {"questions": questions}
    except Exception as e:
        logger.error(f"Exam generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not generate exam")

@app.post("/exam/grade", response_model=dict)
async def grade_answer_endpoint(answer: AnswerSubmit, username: str = Depends(get_current_user)):
    try:
        question = await questions_collection.find_one({"id": answer.question_id})
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        rag_chain = initialize_rag_chain(username, question["lecture_name"])
        prompt = GRADING_PROMPT.format(
            question=question["question"],
            answer=answer.answer,
            correct_answer=question["correct_answer"] if question["type"] == "mcq" else "No predefined answer"
        )
        response = rag_chain.invoke({"query": prompt})
        return {"feedback": response["result"]}
    except Exception as e:
        logger.error(f"Grading error: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not grade answer")

@app.get("/health")
async def health_check():
    try:
        if client:
            await client.admin.command('ping')
        else:
            raise HTTPException(status_code=500, detail="MongoDB not initialized")
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        volume_writable = check_volume_writable()
        return {
            "status": "healthy" if volume_writable else "unhealthy",
            "mongodb": "connected" if client else "disconnected",
            "memory_percent": mem.percent,
            "disk_percent": disk.percent,
            "volume_writable": volume_writable
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/resources")
async def resource_check():
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    volume_writable = check_volume_writable()
    return {
        "memory": {
            "total": f"{mem.total/1024/1024:.2f} MB",
            "available": f"{mem.available/1024/1024:.2f} MB",
            "used": f"{mem.used/1024/1024:.2f} MB",
            "percent": mem.percent
        },
        "disk": {
            "total": f"{disk.total/1024/1024:.2f} MB",
            "free": f"{disk.free/1024/1024:.2f} MB",
            "used": f"{disk.used/1024/1024:.2f} MB",
            "percent": disk.percent
        },
        "volume_writable": volume_writable,
        "status": "ok" if mem.percent < 85 and disk.percent < 90 and volume_writable else "warning"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
