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
from dotenv import load_dotenv
from PyPDF2 import PdfReader
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
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI()

# Security configurations
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
USER_DATA_DIR = "/app/user_data"  # Railway volume mount path
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Lazy initialize embedding model
embeddings_model = None
def get_embeddings_model():
    global embeddings_model
    if embeddings_model is None:
        logger.info("Initializing HuggingFaceEmbeddings")
        embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings_model

# MongoDB setup with connection timeout
try:
    client = motor.motor_asyncio.AsyncIOMotorClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000  # 5-second timeout
    )
    db = client.student_assistant
    users_collection = db.users
    courses_collection = db.courses
    lectures_collection = db.lectures
    questions_collection = db.questions
    logger.info("MongoDB client initialized successfully")
    
    # Create indexes for performance
    async def create_indexes():
        await users_collection.create_index("username", unique=True)
        await courses_collection.create_index([("username", 1), ("course_name", 1)], unique=True)
        await lectures_collection.create_index([("username", 1), ("course_name", 1), ("lecture_name", 1)], unique=True)
        await questions_collection.create_index("id", unique=True)
        logger.info("MongoDB indexes created successfully")
    
    # Run index creation
    import asyncio
    asyncio.create_task(create_indexes())
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

# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    details = "; ".join([f"{err['loc'][-1]}: {err['msg']}" for err in errors])
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation error", "details": details},
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
        headers={"Access-Control-Allow-Origin": "*"}
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

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

# MongoDB functions
async def get_user(username: str) -> Optional[Dict]:
    logger.info(f"Fetching user: {username}")
    try:
        user = await users_collection.find_one({"username": username})
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
    except HTTPException as he:
        raise he
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
        return [course["course_name"] for course in courses]
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
        logger.info(f"Course '{course_name}' created successfully for user: {username}")
    except HTTPException as he:
        raise he
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
        return [{"name": lec["lecture_name"], "path": lec["file_path"]} for lec in lectures]
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
            logger.error("Lecture already exists")
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
        logger.info(f"Lecture '{lecture_name}' created successfully for user: {username}")
    except HTTPException as he:
        raise he
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
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced to lower memory usage
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text)
        embeddings = get_embeddings_model()
        return FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process document"
        )

def initialize_rag_chain(username: str, lecture_name: str) -> RetrievalQA:
    try:
        faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
        embeddings = get_embeddings_model()
        vector_store = FAISS.load_local(
            faiss_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not initialize AI service"
        )

def extract_text_from_pdf(file_path: str, username: str, lecture_name: str) -> str:
    try:
        logger.info(f"Opening PDF file: {file_path}")
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages, 1):
            logger.debug(f"Extracting text from page {page_num}")
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                logger.warning(f"No text extracted from page {page_num}")
        if not text.strip():
            logger.error("No text could be extracted from PDF")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF contains no extractable text"
            )
        # Create and save FAISS index
        logger.info(f"Creating FAISS index for lecture: {lecture_name}")
        faiss_index = create_faiss_index(text)
        faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
        logger.info(f"Saving FAISS index to: {faiss_path}")
        os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
        faiss_index.save_local(faiss_path)
        return text
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}", exc_info=True)
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
        logger.error(f"Error parsing exam: {str(e)}", exc_info=True)
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
        logger.error(f"Registration failed: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}", exc_info=True)
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
        logger.error(f"Unexpected error during login: {str(e)}", exc_info=True)
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
        logger.error(f"Error retrieving profile for {username}: {str(e)}", exc_info=True)
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
        logger.error(f"Course creation failed: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during course creation: {str(e)}", exc_info=True)
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
        logger.error(f"Error retrieving courses: {str(e)}", exc_info=True)
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
    logger.info(f"Uploading lecture '{lecture_name}' for course '{course_name}' by user '{username}'")
    
    try:
        # Validate inputs
        validate_name(lecture_name, "Lecture name")
        validate_name(course_name, "Course name")
        logger.debug(f"Input validation passed: lecture_name={lecture_name}, course_name={course_name}")

        # Validate file type
        if file.content_type != "application/pdf":
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed"
            )
        logger.debug(f"File type validated: {file.content_type}")

        # Validate file size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        logger.info(f"File size: {file_size} bytes")
        if file_size > MAX_FILE_SIZE:
            logger.error(f"File too large: {file_size} bytes, max is {MAX_FILE_SIZE} bytes")
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size is {MAX_FILE_SIZE/1024/1024}MB"
            )
        if file_size == 0:
            logger.error("Empty file uploaded")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        file.file.seek(0)
        logger.debug("File size validated")

        # Check if course exists
        courses = await get_user_courses(username)
        if course_name not in courses:
            logger.error(f"Course '{course_name}' not found for user '{username}'")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Course not found"
            )
        logger.debug(f"Course '{course_name}' exists")

        # Prepare file storage path
        lecture_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}.pdf")
        logger.info(f"Preparing to save lecture to {lecture_path}")
        os.makedirs(os.path.dirname(lecture_path), exist_ok=True)
        
        # Save the uploaded file
        logger.debug(f"Writing file to {lecture_path}")
        try:
            with open(lecture_path, "wb") as f:
                content = await file.read()
                logger.info(f"Writing {len(content)} bytes to {lecture_path}")
                f.write(content)
            logger.debug(f"File successfully saved to {lecture_path}")
        except OSError as e:
            logger.error(f"Failed to write file to {lecture_path}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file: {str(e)}"
            )

        # Process the PDF
        logger.info(f"Extracting text from PDF at {lecture_path}")
        lecture_text = extract_text_from_pdf(lecture_path, username, lecture_name)
        logger.debug(f"PDF text extraction completed, length: {len(lecture_text)} characters")

        # Store lecture in database
        logger.info(f"Storing lecture metadata in MongoDB")
        await create_lecture_db(username, course_name, lecture_name, lecture_path)
        logger.info(f"Lecture '{lecture_name}' successfully uploaded and stored")

        return {"message": f"Lecture '{lecture_name}' uploaded"}
    except HTTPException as he:
        logger.error(f"Upload failed: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during lecture upload: {str(e)}", exc_info=True)
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
        logger.info(f"Retrieved lectures for course '{course_name}' by user: {username}")
        return {"lectures": lecture_names}
    except Exception as e:
        logger.error(f"Error retrieving lectures: {str(e)}", exc_info=True)
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
        # Check if lecture exists
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
        logger.error(f"Error generating study content: {str(e)}", exc_info=True)
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
        # Check if lecture exists
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
        logger.error(f"Error generating exam: {str(e)}", exc_info=True)
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
        # Retrieve question details from database
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
        
        # Initialize RAG chain with the correct lecture
        rag_chain = initialize_rag_chain(username, lecture_name)
        
        # Format prompt with question and correct answer
        prompt = GRADING_PROMPT.format(
            question=question_text,
            answer=answer.answer,
            correct_answer=correct_answer if q_type == "mcq" else "No predefined correct answer for essay questions"
        )
        
        response = rag_chain.invoke({"query": prompt})
        logger.info(f"Graded answer for question '{answer.question_id}' by user: {username}")
        return {"feedback": response["result"]}
    except Exception as e:
        logger.error(f"Error grading answer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not grade answer: {str(e)}"
        )

@app.get("/health")
async def health_check():
    try:
        # Check MongoDB connection
        await client.admin.command('ping')
        logger.info("Health check: MongoDB connection successful")
        return {"status": "healthy", "mongodb": "connected"}
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Health check failed: MongoDB connection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: MongoDB connection error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))  # Use Railway's PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
