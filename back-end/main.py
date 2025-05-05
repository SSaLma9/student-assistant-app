from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
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

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI()

# Security configurations
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "hsgdter453cnhfgdt658ddlkdk*m54wq")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Add CORS middleware
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "https://student-assistant-frontend-production.up.railway.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
USER_DATA_DIR = "/app/user_data"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
os.makedirs(USER_DATA_DIR, exist_ok=True)

# MongoDB client (will be initialized in startup event)
client = None
db = None
users_collection = None
courses_collection = None
lectures_collection = None
questions_collection = None

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
                    logger.info(f"Attempt {attempt + 1}: Loading sentence-transformers/all-MiniLM-L6-v2")
                    embeddings_model = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={
                            'normalize_embeddings': True,
                            'batch_size': 2  # Reduced batch size
                        }
                    )
                    logger.info("HuggingFaceEmbeddings initialized successfully")
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            if embeddings_model is None:
                raise RuntimeError("Embeddings model failed to initialize after retries")
        except MemoryError as me:
            logger.error(f"Memory error initializing HuggingFaceEmbeddings: {str(me)}")
            raise HTTPException(
                status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
                detail="Server ran out of memory during model initialization."
            )
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFaceEmbeddings: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"AI service unavailable: {str(e)}"
            )
    return embeddings_model

# MongoDB setup with retries
async def init_mongodb():
    max_retries = 5
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            client = motor.motor_asyncio.AsyncIOMotorClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000
            )
            await client.admin.command('ping')
            logger.info("MongoDB connection established")
            return client
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            if attempt == max_retries - 1:
                logger.error(f"MongoDB connection failed after {max_retries} attempts: {str(e)}")
                raise Exception(f"MongoDB connection failed: {str(e)}")
            logger.warning(f"MongoDB connection attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2

# Initialize MongoDB on startup
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
        logger.info("MongoDB client initialized successfully")
        
        # Create indexes
        try:
            await users_collection.create_index("username", unique=True)
            await courses_collection.create_index([("username", 1), ("course_name", 1)], unique=True)
            await lectures_collection.create_index([("username", 1), ("course_name", 1), ("lecture_name", 1)], unique=True)
            await questions_collection.create_index("id", unique=True)
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB client: {str(e)}")
        raise Exception(f"MongoDB connection failed: {str(e)}")

# Input validation regex
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

def validate_name(name: str, field: str) -> None:
    if not NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{field} must contain only letters, numbers, underscores, or hyphens"
        )

def check_memory_usage():
    mem = psutil.virtual_memory()
    logger.info(f"Memory usage: {mem.percent}% (Total: {mem.total/1024/1024:.2f}MB, Used: {mem.used/1024/1024:.2f}MB)")
    if mem.percent > 85:
        logger.error(f"Memory usage too high: {mem.percent}%")
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="Server memory overloaded, please try again later or use a smaller file"
        )

def cleanup_lecture_files(lecture_path: str, faiss_path: str):
    try:
        if os.path.exists(lecture_path):
            os.remove(lecture_path)
            logger.info(f"Cleaned up lecture file: {lecture_path}")
        if os.path.exists(faiss_path):
            shutil.rmtree(faiss_path)
            logger.info(f"Cleaned up FAISS index: {faiss_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up files: {str(e)}")

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    details = "; ".join([f"{err['loc'][-1]}: {err['msg']}" for err in errors])
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation error", "details": details},
        headers={"Access-Control-Allow-Origin": FRONTEND_URL}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers={"Access-Control-Allow-Origin": FRONTEND_URL}
    )

@app.exception_handler(MemoryError)
async def memory_error_handler(request: Request, exc: MemoryError):
    logger.error(f"Memory error occurred: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
        content={"error": "Server ran out of memory. Try a smaller file or upgrade your plan."},
        headers={"Access-Control-Allow-Origin": FRONTEND_URL}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": f"Internal server error: {str(exc)}"},
        headers={"Access-Control-Allow-Origin": FRONTEND_URL}
    )

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
    logger.info(f"Fetching user: {username}")
    try:
        user = await users_collection.find_one({"username": username})
        if user is None:
            logger.warning(f"User {username} not found")
        return user
    except Exception as e:
        logger.error(f"Error fetching user {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not fetch user: {str(e)}"
        )

async def create_user(username: str, hashed_password: str):
    logger.info(f"Creating user: {username}")
    try:
        validate_name(username, "Username")
        existing_user = await get_user(username)
        if existing_user:
            logger.error("Username already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        user = {"username": username, "hashed_password": hashed_password}
        await users_collection.insert_one(user)
        logger.info(f"User {username} created successfully")
    except DuplicateKeyError:
        logger.error("Duplicate username")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    except Exception as e:
        logger.error(f"Error creating user {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create user: {str(e)}"
        )

async def get_user_courses(username: str) -> List[str]:
    logger.info(f"Fetching courses for user: {username}")
    try:
        courses = await courses_collection.find({"username": username}).to_list(None)
        course_names = [course["course_name"] for course in courses]
        logger.info(f"Found {len(course_names)} courses for {username}")
        return course_names
    except Exception as e:
        logger.error(f"Error fetching courses for {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not fetch courses: {str(e)}"
        )

async def create_course_db(username: str, course_name: str):
    logger.info(f"Creating course '{course_name}' for user: {username}")
    try:
        validate_name(course_name, "Course name")
        existing_course = await courses_collection.find_one({"username": username, "course_name": course_name})
        if existing_course:
            logger.error("Course already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Course already exists"
            )
        course = {"username": username, "course_name": course_name}
        await courses_collection.insert_one(course)
        logger.info(f"Course '{course_name}' created successfully")
    except DuplicateKeyError:
        logger.error("Duplicate course name")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Course already exists"
        )
    except Exception as e:
        logger.error(f"Error creating course {course_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create course: {str(e)}"
        )

async def get_user_lectures(username: str, course_name: str) -> List[Dict]:
    logger.info(f"Fetching lectures for user: {username}, course: {course_name}")
    try:
        lectures = await lectures_collection.find({"username": username, "course_name": course_name}).to_list(None)
        lecture_list = [{"name": lec["lecture_name"], "path": lec["file_path"]} for lec in lectures]
        logger.info(f"Found {len(lecture_list)} lectures for {username}/{course_name}")
        return lecture_list
    except Exception as e:
        logger.error(f"Error fetching lectures for {username}/{course_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not fetch lectures: {str(e)}"
        )

async def create_lecture_db(username: str, course_name: str, lecture_name: str, file_path: str):
    logger.info(f"Creating lecture '{lecture_name}' for user: {username}, course: {course_name}")
    try:
        validate_name(lecture_name, "Lecture name")
        existing_lecture = await lectures_collection.find_one({
            "username": username,
            "course_name": course_name,
            "lecture_name": lecture_name
        })
        if existing_lecture:
            logger.error(f"Lecture '{lecture_name}' already exists")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Lecture already exists"
            )
        lecture = {
            "username": username,
            "course_name": course_name,
            "lecture_name": lecture_name,
            "file_path": file_path
        }
        await lectures_collection.insert_one(lecture)
        logger.info(f"Lecture '{lecture_name}' created successfully")
    except DuplicateKeyError:
        logger.error(f"Duplicate lecture name: {lecture_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Lecture name already exists"
        )
    except Exception as e:
        logger.error(f"Error creating lecture {lecture_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create lecture: {str(e)}"
        )

# Authentication functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username

# AI and file processing functions
def create_faiss_index(text: str) -> FAISS:
    logger.info("Creating FAISS index")
    try:
        if not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text provided for indexing"
            )
        
        check_memory_usage()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=5
        )
        
        logger.info("Splitting text into chunks")
        chunks = text_splitter.split_text(text)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid text chunks could be processed"
            )
        logger.info(f"Created {len(chunks)} text chunks")
        
        embeddings = get_embeddings_model()
        
        batch_size = 2  # Reduced batch size
        vectors = []
        logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}")
        for i in range(0, len(chunks), batch_size):
            check_memory_usage()
            batch_chunks = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
            try:
                batch_vectors = embeddings.embed_documents(batch_chunks)
                vectors.extend(batch_vectors)
                logger.info(f"Embedded batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                raise
            del batch_chunks, batch_vectors
            gc.collect()
            logger.info(f"Cleaned up batch {i//batch_size + 1}")
        
        logger.info("Creating text-embedding pairs")
        text_embeddings = list(zip(chunks, vectors))
        check_memory_usage()
        
        logger.info("Building FAISS index")
        faiss_index = FAISS.from_embeddings(text_embeddings, embeddings)
        logger.info("FAISS index created successfully")
        del text_embeddings, vectors, chunks
        gc.collect()
        check_memory_usage()
        return faiss_index
    except MemoryError as me:
        logger.error(f"Memory error creating FAISS index: {str(me)}")
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="Server ran out of memory. Try a smaller file."
        )
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create FAISS index: {str(e)}"
        )

def initialize_rag_chain(username: str, lecture_name: str) -> RetrievalQA:
    try:
        check_memory_usage()
        
        faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
        logger.info(f"Loading FAISS index from {faiss_path}")
        
        if not os.path.exists(faiss_path):
            logger.error(f"FAISS index not found at {faiss_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lecture index not found - please re-upload the lecture"
            )

        embeddings = get_embeddings_model()
        
        try:
            vector_store = FAISS.load_local(
                faiss_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load lecture data"
            )

        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192",
            max_tokens=512
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 1}
        )

        check_memory_usage()

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=False
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service temporarily unavailable - try again later"
        )

def validate_pdf(file_path: str) -> None:
    logger.info(f"Validating PDF: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            if reader.is_encrypted:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="PDF is encrypted and cannot be processed"
                )
            if len(reader.pages) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="PDF is empty or invalid"
                )
            # Test text extraction on first page
            first_page = reader.pages[0]
            if not first_page.extract_text():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="PDF contains no extractable text (likely image-based or scanned)"
                )
    except PdfReadError as e:
        logger.error(f"PDF validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted PDF file: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during PDF validation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not validate PDF: {str(e)}"
        )

def extract_text_from_pdf(file_path: str, username: str, lecture_name: str) -> str:
    logger.info(f"Processing PDF: {file_path}")
    faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
    
    try:
        validate_pdf(file_path)
        
        text = ""
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            total_pages = len(reader.pages)
            logger.info(f"PDF has {total_pages} pages")
            
            for page_num in range(total_pages):
                if page_num % 5 == 0:
                    check_memory_usage()
                    gc.collect()
                
                page = reader.pages[page_num]
                try:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    if page_num % 10 == 0 and total_pages > 20:
                        del page_text
                        gc.collect()
                except Exception as e:
                    logger.warning(f"Page {page_num+1} extraction failed: {str(e)}")
        
        if not text.strip():
            cleanup_lecture_files(file_path, faiss_path)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF contains no extractable text"
            )
            
        logger.info(f"Creating FAISS index for lecture: {lecture_name}")
        faiss_index = create_faiss_index(text)
        
        logger.info(f"Saving FAISS index to: {faiss_path}")
        os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
        
        try:
            faiss_index.save_local(faiss_path)
            logger.info(f"FAISS index saved to {faiss_path}")
        except OSError as e:
            logger.error(f"Failed to save FAISS index to {faiss_path}: {str(e)}")
            cleanup_lecture_files(file_path, faiss_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not save FAISS index: {str(e)}"
            )
            
        return text
    except HTTPException as he:
        cleanup_lecture_files(file_path, faiss_path)
        raise he
    except MemoryError as me:
        cleanup_lecture_files(file_path, faiss_path)
        logger.error(f"Memory error processing PDF: {str(me)}")
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="Server ran out of memory. Try a smaller PDF or upgrade your plan."
        )
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        cleanup_lecture_files(file_path, faiss_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not process PDF file: {str(e)}"
        )

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
                logger.info(f"Inserting MCQ {question_id} for lecture '{lecture_name}'")
                await questions_collection.update_one(
                    {"id": question_id},
                    {"$set": question},
                    upsert=True
                )
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
                logger.info(f"Inserting essay question {question_id} for lecture '{lecture_name}'")
                await questions_collection.update_one(
                    {"id": question_id},
                    {"$set": question},
                    upsert=True
                )
        return flattened
    except Exception as e:
        logger.error(f"Error parsing exam: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not parse exam content"
        )

# API Endpoints
@app.post("/register", response_model=dict)
async def register(credentials: UserCredentials):
    logger.info(f"Register attempt for username: {credentials.username}")
    if not credentials.username or not credentials.password:
        logger.error("Username or password missing")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password required"
        )
    
    hashed_password = hash_password(credentials.password)
    try:
        await create_user(credentials.username, hashed_password)
        logger.info(f"User {credentials.username} registered successfully")
        access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": credentials.username}, expires_delta=access_token_expires
        )
        return {"message": "Registered successfully", "token": access_token}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not complete registration: {str(e)}"
        )

@app.post("/login", response_model=dict)
async def login(credentials: UserCredentials):
    logger.info(f"Login attempt for username: {credentials.username}")
    user = await get_user(credentials.username)
    if not user or not verify_password(credentials.password, user["hashed_password"]):
        logger.error("Invalid credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": credentials.username}, expires_delta=access_token_expires
        )
        logger.info(f"User {credentials.username} logged in successfully")
        return {"token": access_token}
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not complete login: {str(e)}"
        )

@app.get("/profile", response_model=dict)
async def get_profile(username: str = Depends(get_current_user)):
    try:
        logger.info(f"Fetching profile for user: {username}")
        courses = await get_user_courses(username)
        profile = {"username": username, "courses": []}
        
        for course_name in courses:
            lectures = await get_user_lectures(username, course_name)
            lecture_names = [lec["name"] for lec in lectures]
            profile["courses"].append({
                "course_name": course_name,
                "lectures": lecture_names
            })
        
        logger.info(f"Profile retrieved for user: {username}")
        return {"profile": profile}
    except Exception as e:
        logger.error(f"Error retrieving profile for {username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve profile: {str(e)}"
        )

@app.post("/courses", response_model=dict)
async def create_course(
    course: CourseCreate,
    username: str = Depends(get_current_user)
):
    try:
        await create_course_db(username, course.course_name)
        return {"message": f"Course '{course.course_name}' created"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Course creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create course: {str(e)}"
        )

@app.get("/courses", response_model=dict)
async def list_courses(username: str = Depends(get_current_user)):
    try:
        courses = await get_user_courses(username)
        logger.info(f"Retrieved courses for user: {username}")
        return {"courses": courses}
    except Exception as e:
        logger.error(f"Error retrieving courses: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve courses: {str(e)}"
        )

@app.post("/lectures", response_model=dict)
async def upload_lecture(
    lecture_name: str = Form(...),
    course_name: str = Form(...),
    file: UploadFile = File(...),
    username: str = Depends(get_current_user)
):
    logger.info(f"Received upload request: lecture_name={lecture_name}, course_name={course_name}, file={file.filename}, size={file.size}")
    
    lecture_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}.pdf")
    faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
    
    try:
        check_memory_usage()
        
        if not check_volume_writable():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Storage volume is not writable"
            )
        
        validate_name(lecture_name, "Lecture name")
        validate_name(course_name, "Course name")

        if file.content_type != "application/pdf":
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed"
            )

        file.file.seek(0, 2)
        file_size = file.file.tell()
        logger.info(f"File size: {file_size} bytes")
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size is {MAX_FILE_SIZE/1024/1024}MB"
            )
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        file.file.seek(0)

        courses = await get_user_courses(username)
        if course_name not in courses:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )

        existing_lecture = await lectures_collection.find_one({
            "username": username,
            "course_name": course_name,
            "lecture_name": lecture_name
        })
        if existing_lecture:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Lecture already exists"
            )

        os.makedirs(os.path.dirname(lecture_path), exist_ok=True)
        logger.info(f"Saving PDF to: {lecture_path}")
        try:
            with open(lecture_path, "wb") as f:
                while chunk := await file.read(8192):
                    f.write(chunk)
            logger.info(f"PDF saved successfully: {lecture_path}")
        except PermissionError as e:
            logger.error(f"Permission denied writing to {lecture_path}: {str(e)}")
            cleanup_lecture_files(lecture_path, faiss_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server file permission error"
            )
        except OSError as e:
            logger.error(f"Failed to save file to {lecture_path}: {str(e)}")
            cleanup_lecture_files(lecture_path, faiss_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {str(e)}"
            )

        lecture_text = extract_text_from_pdf(lecture_path, username, lecture_name)

        await create_lecture_db(username, course_name, lecture_name, lecture_path)

        return {"message": f"Lecture '{lecture_name}' uploaded"}
    
    except HTTPException as he:
        cleanup_lecture_files(lecture_path, faiss_path)
        raise he
    except MemoryError:
        cleanup_lecture_files(lecture_path, faiss_path)
        raise HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail="Server ran out of memory. Try a smaller PDF or upgrade your plan."
        )
    except Exception as e:
        logger.error(f"Unexpected error during lecture upload: {str(e)}")
        cleanup_lecture_files(lecture_path, faiss_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not upload lecture: {str(e)}"
        )

@app.get("/lectures/{course_name}", response_model=dict)
async def list_lectures(course_name: str, username: str = Depends(get_current_user)):
    try:
        courses = await get_user_courses(username)
        if course_name not in courses:
            logger.error(f"Course '{course_name}' not found for user '{username}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        lectures = await get_user_lectures(username, course_name)
        lecture_names = [lec["name"] for lec in lectures]
        logger.info(f"Retrieved {len(lecture_names)} lectures for course '{course_name}' by user: {username}")
        return {"lectures": lecture_names}
    except Exception as e:
        logger.error(f"Error retrieving lectures: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve lectures: {str(e)}"
        )

@app.post("/study", response_model=dict)
async def generate_study_content(
    request: StudyRequest,
    username: str = Depends(get_current_user)
):
    try:
        lecture = await lectures_collection.find_one({
            "username": username,
            "lecture_name": request.lecture_name
        })
        if not lecture:
            logger.error(f"Lecture '{request.lecture_name}' not found for user '{username}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lecture not found"
            )
        
        rag_chain = initialize_rag_chain(username, request.lecture_name)
        
        if request.task == "Custom Question":
            if not request.question:
                logger.error("Question required for Custom Question task")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Question required for Custom Question task"
                )
            query = STUDY_PROMPTS[request.task].format(text="", question=request.question)
        else:
            query = STUDY_PROMPTS[request.task].format(text="")
            
        response = rag_chain.invoke({"query": query})
        logger.info(f"Generated study content for task '{request.task}' by user: {username}")
        return {"content": response["result"]}
    except Exception as e:
        logger.error(f"Error generating study content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not generate study content: {str(e)}"
        )

@app.post("/exam", response_model=dict)
async def generate_exam(
    request: ExamRequest,
    username: str = Depends(get_current_user)
):
    try:
        lecture = await lectures_collection.find_one({
            "username": username,
            "lecture_name": request.lecture_name
        })
        if not lecture:
            logger.error(f"Lecture '{request.lecture_name}' not found for user '{username}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Lecture not found"
            )
        
        rag_chain = initialize_rag_chain(username, request.lecture_name)
        prompt = EXAM_PROMPT.format(
            text="",
            level=request.difficulty,
            exam_type=request.exam_type
        )
        response = rag_chain.invoke({"query": prompt})
        questions = await parse_exam(response["result"], request.exam_type, request.lecture_name)
        logger.info(f"Generated exam for lecture '{request.lecture_name}' by user: {username}")
        return {"questions": questions}
    except Exception as e:
        logger.error(f"Error generating exam: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not generate exam: {str(e)}"
        )

@app.post("/exam/grade", response_model=dict)
async def grade_answer_endpoint(
    answer: AnswerSubmit,
    username: str = Depends(get_current_user)
):
    try:
        question = await questions_collection.find_one({"id": answer.question_id})
        if not question:
            logger.error(f"Question '{answer.question_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
        
        lecture_name = question["lecture_name"]
        question_text = question["question"]
        q_type = question["type"]
        options = question["options"]
        correct_answer = question["correct_answer"]
        
        rag_chain = initialize_rag_chain(username, lecture_name)
        
        prompt = GRADING_PROMPT.format(
            question=question_text,
            answer=answer.answer,
            correct_answer=correct_answer if q_type == "mcq" else "No predefined correct answer for essay questions"
        )
        
        response = rag_chain.invoke({"query": prompt})
        logger.info(f"Graded answer for question '{answer.question_id}' by user: {username}")
        return {"feedback": response["result"]}
    except Exception as e:
        logger.error(f"Error grading answer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not grade answer: {str(e)}"
        )

@app.get("/health")
async def health_check():
    try:
        await client.admin.command('ping')
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        volume_writable = check_volume_writable()
        logger.info(f"Health check: MongoDB connected, Memory: {mem.percent}%, Disk: {disk.percent}%, Volume writable: {volume_writable}")
        return {
            "status": "healthy" if volume_writable else "unhealthy",
            "mongodb": "connected",
            "memory_percent": mem.percent,
            "disk_percent": disk.percent,
            "volume_writable": volume_writable
        }
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MongoDB connection error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

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
    logger.info(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
