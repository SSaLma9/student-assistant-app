from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from passlib.context import CryptContext
from typing import Optional, List, Dict
import sqlite3
import jwt
import datetime
import PyPDF2
import os
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pathlib import Path
import re
import logging

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Security configurations
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")  # Ensure this is set in .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, or use your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Configuration
USER_DATA_DIR = "user_data"
USERS_FILE = "users.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Initialize embedding model globally
embeddings_model = HuggingFaceEmbeddings( model_name="sentence-transformers_all-MiniLM-L6-v2")

# Database setup
def init_db():
    conn = sqlite3.connect('student_assistant.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS questions
                 (id TEXT PRIMARY KEY, lecture_name TEXT, question TEXT, type TEXT, options TEXT, correct_answer TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Custom exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    details = "; ".join([f"{err['loc'][-1]}: {err['msg']}" for err in errors])
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Validation error", "details": details},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"},
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

# Database and user management functions
def load_users():
    try:
        if not os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'w') as f:
                json.dump({}, f)
            return {}
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not load user data"
        )

def save_users(users: dict):
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save user data"
        )

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

def load_user_profile(username: str) -> dict:
    try:
        user_file = os.path.join(USER_DATA_DIR, f"{username}.json")
        user_dir = os.path.join(USER_DATA_DIR, username)
        lectures_dir = os.path.join(user_dir, "lectures")
        
        os.makedirs(lectures_dir, exist_ok=True)
        
        if not os.path.exists(user_file):
            profile = {"courses": {}}
            with open(user_file, 'w') as f:
                json.dump(profile, f, indent=4)
            return profile
            
        with open(user_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not load user profile"
        )

def save_user_profile(username: str, profile: dict):
    try:
        user_file = os.path.join(USER_DATA_DIR, f"{username}.json")
        with open(user_file, 'w') as f:
            json.dump(profile, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save user profile"
        )

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
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        return FAISS.from_texts(chunks, embeddings_model)
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process document"
        )

def initialize_rag_chain(username: str, lecture_name: str) -> RetrievalQA:
    try:
        faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
        vector_store = FAISS.load_local(
            faiss_path,
            embeddings_model,
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
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        # Create and save FAISS index
        faiss_index = create_faiss_index(text)
        faiss_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}_faiss")
        faiss_index.save_local(faiss_path)
        
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process PDF file"
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

def parse_exam(exam_text: str, exam_type: str, lecture_name: str) -> List[Dict]:
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
        conn = sqlite3.connect('student_assistant.db')
        c = conn.cursor()
        if exam_type == "MCQs":
            for idx, q in enumerate(mcqs):
                lines = q.split('\n')
                question_text = next((line for line in lines if re.match(r"^\d+\.\s", line)), "")
                options = [line for line in lines if re.match(r"^[A-D]\)", line)]
                answer_line = next((line for line in lines if line.startswith("Answer:")), "")
                answer = answer_line.replace("Answer:", "").strip() if answer_line else ""
                question_id = f"mcq_{lecture_name}_{idx}"
                flattened.append({
                    "id": question_id,
                    "type": "mcq",
                    "question": question_text,
                    "options": options,
                    "correct_answer": answer
                })
                # Store in database
                c.execute(
                    "INSERT OR REPLACE INTO questions (id, lecture_name, question, type, options, correct_answer) VALUES (?, ?, ?, ?, ?, ?)",
                    (question_id, lecture_name, question_text, "mcq", json.dumps(options), answer)
                )
        elif exam_type == "Essay Questions":
            for idx, q in enumerate(essays):
                question_id = f"essay_{lecture_name}_{idx}"
                flattened.append({
                    "id": question_id,
                    "type": "essay",
                    "question": q,
                    "correct_answer": ""
                })
                # Store in database
                c.execute(
                    "INSERT OR REPLACE INTO questions (id, lecture_name, question, type, options, correct_answer) VALUES (?, ?, ?, ?, ?, ?)",
                    (question_id, lecture_name, q, "essay", json.dumps([]), "")
                )
        conn.commit()
        conn.close()
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
    users = load_users()
    if credentials.username in users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    if not credentials.username or not credentials.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password required"
        )
    
    users[credentials.username] = hash_password(credentials.password)
    save_users(users)
    
    try:
        os.makedirs(os.path.join(USER_DATA_DIR, credentials.username, "lectures"), exist_ok=True)
        # Generate token for immediate login after registration
        access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": credentials.username}, expires_delta=access_token_expires
        )
        return {"message": "Registered successfully", "token": access_token}
    except Exception as e:
        logger.error(f"Error creating user directories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not complete registration"
        )

@app.post("/login", response_model=dict)
async def login(credentials: UserCredentials):
    users = load_users()
    if (credentials.username not in users or 
        not verify_password(credentials.password, users[credentials.username])):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": credentials.username}, expires_delta=access_token_expires
    )
    return {"token": access_token}

@app.post("/courses", response_model=dict)
async def create_course(
    course: CourseCreate,
    username: str = Depends(get_current_user)
):
    profile = load_user_profile(username)
    if course.course_name in profile["courses"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Course already exists"
        )
        
    profile["courses"][course.course_name] = []
    save_user_profile(username, profile)
    return {"message": f"Course '{course.course_name}' created"}

@app.get("/courses", response_model=dict)
async def list_courses(username: str = Depends(get_current_user)):
    profile = load_user_profile(username)
    return {"courses": list(profile["courses"].keys())}

@app.post("/lectures", response_model=dict)
async def upload_lecture(
    lecture_name: str = Form(...),
    course_name: str = Form(...),
    file: UploadFile = File(...),
    username: str = Depends(get_current_user)
):
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )
    
    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size is {MAX_FILE_SIZE/1024/1024}MB"
        )
    file.file.seek(0)
    
    profile = load_user_profile(username)
    if course_name not in profile["courses"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found"
        )
    
    try:
        lecture_path = os.path.join(USER_DATA_DIR, username, "lectures", f"{lecture_name}.pdf")
        
        # Save the uploaded file
        with open(lecture_path, "wb") as f:
            f.write(await file.read())
        
        # Process the PDF
        lecture_text = extract_text_from_pdf(lecture_path, username, lecture_name)
        
        # Update user profile
        lecture_data = {"name": lecture_name, "path": lecture_path}
        profile["courses"][course_name].append(lecture_data)
        save_user_profile(username, profile)
        
        return {"message": f"Lecture '{lecture_name}' uploaded"}
    except Exception as e:
        logger.error(f"Error uploading lecture: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not upload lecture"
        )

@app.get("/lectures/{course_name}", response_model=dict)
async def list_lectures(course_name: str, username: str = Depends(get_current_user)):
    profile = load_user_profile(username)
    if course_name not in profile["courses"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Course not found"
        )
    lectures = [lec["name"] for lec in profile["courses"][course_name]]
    return {"lectures": lectures}

@app.post("/study", response_model=dict)
async def generate_study_content(
    request: StudyRequest,
    username: str = Depends(get_current_user)
):
    profile = load_user_profile(username)
    lecture_exists = any(
        lec["name"] == request.lecture_name
        for course in profile["courses"].values()
        for lec in course
    )
    if not lecture_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lecture not found"
        )
    
    try:
        rag_chain = initialize_rag_chain(username, request.lecture_name)
        
        if request.task == "Custom Question":
            if not request.question:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Question required for Custom Question task"
                )
            query = STUDY_PROMPTS[request.task].format(text="", question=request.question)
        else:
            query = STUDY_PROMPTS[request.task].format(text="")
            
        response = rag_chain.invoke({"query": query})
        return {"content": response["result"]}
    except Exception as e:
        logger.error(f"Error generating study content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate study content"
        )

@app.post("/exam", response_model=dict)
async def generate_exam(
    request: ExamRequest,
    username: str = Depends(get_current_user)
):
    profile = load_user_profile(username)
    lecture_exists = any(
        lec["name"] == request.lecture_name
        for course in profile["courses"].values()
        for lec in course
    )
    if not lecture_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lecture not found"
        )
    
    try:
        rag_chain = initialize_rag_chain(username, request.lecture_name)
        prompt = EXAM_PROMPT.format(
            text="",
            level=request.difficulty,
            exam_type=request.exam_type
        )
        response = rag_chain.invoke({"query": prompt})
        questions = parse_exam(response["result"], request.exam_type, request.lecture_name)
        return {"questions": questions}
    except Exception as e:
        logger.error(f"Error generating exam: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate exam"
        )

@app.post("/exam/grade", response_model=dict)
async def grade_answer_endpoint(
    answer: AnswerSubmit,
    username: str = Depends(get_current_user)
):
    try:
        # Retrieve question details from database
        conn = sqlite3.connect('student_assistant.db')
        c = conn.cursor()
        c.execute(
            "SELECT lecture_name, question, type, options, correct_answer FROM questions WHERE id = ?",
            (answer.question_id,)
        )
        result = c.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Question not found"
            )
        
        lecture_name, question_text, q_type, options_json, correct_answer = result
        options = json.loads(options_json) if options_json else []
        
        # Initialize RAG chain with the correct lecture
        rag_chain = initialize_rag_chain(username, lecture_name)
        
        # Format prompt with question and correct answer
        prompt = GRADING_PROMPT.format(
            question=question_text,
            answer=answer.answer,
            correct_answer=correct_answer if q_type == "mcq" else "No predefined correct answer for essay questions"
        )
        
        response = rag_chain.invoke({"query": prompt})
        return {"feedback": response["result"]}
    except Exception as e:
        logger.error(f"Error grading answer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not grade answer"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
