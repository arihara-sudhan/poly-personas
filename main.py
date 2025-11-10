import json
import math
import re
from pathlib import Path
from typing import Generator, List, Optional, Sequence
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

from llm import LLMBotService
from persona_agent import PersonaAgent

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
AVATAR_DIR = STATIC_DIR / "avatars"
DATABASE_URL = f"sqlite:///{BASE_DIR / 'app.db'}"

load_dotenv(BASE_DIR / ".env")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

bot_service = LLMBotService()
persona_agent = PersonaAgent(bot_service)

GENERAL_PERSONA_NAME = "General Chat"
GENERAL_PERSONA_PROMPT = "You are a friendly general-purpose assistant for the user. Provide helpful, concise replies."


def serialize_embedding(vector: Optional[List[float]]) -> Optional[str]:
    if not vector:
        return None
    return json.dumps(vector)


def deserialize_embedding(payload: Optional[str]) -> Optional[List[float]]:
    if not payload:
        return None
    try:
        data = json.loads(payload)
        if isinstance(data, list):
            return [float(x) for x in data]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_relevant_messages(query_vector: List[float], messages: List["Message"], limit: int = 5) -> List[str]:
    scored = []
    for message in messages:
        embedding = deserialize_embedding(message.embedding)
        if not embedding:
            continue
        score = cosine_similarity(query_vector, embedding)
        scored.append((score, message))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_messages = [msg for _, msg in scored[:limit]]
    return [message.content for message in top_messages]

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    avatar_path = Column(String(255), nullable=True)
    space = relationship("UserSpace", back_populates="user", uselist=False, cascade="all, delete-orphan")
    threads = relationship("PersonaThread", back_populates="user", cascade="all, delete-orphan")


class UserSpace(Base):
    __tablename__ = "user_spaces"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    user = relationship("User", back_populates="space")


class PersonaThread(Base):
    __tablename__ = "persona_threads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    persona_name = Column(String(150), nullable=False)
    persona_prompt = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="threads")
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey("persona_threads.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    thread = relationship("PersonaThread", back_populates="messages")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()

STATIC_DIR.mkdir(exist_ok=True)
AVATAR_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")


@app.on_event("startup")
def on_startup() -> None:
    STATIC_DIR.mkdir(exist_ok=True)
    AVATAR_DIR.mkdir(exist_ok=True)
    Base.metadata.create_all(bind=engine)


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request, db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.id).all()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "users": users},
    )


@app.get("/users/new", response_class=HTMLResponse)
async def new_user_form(request: Request):
    return templates.TemplateResponse("create_user.html", {"request": request})


def ensure_user_space(user: User, db: Session) -> UserSpace:
    if user.space is None:
        space = UserSpace(user=user)
        db.add(space)
        try:
            db.commit()
            db.refresh(user)
        except SQLAlchemyError as exc:
            db.rollback()
            raise HTTPException(status_code=500, detail="Failed to create user space.") from exc
    return user.space


def create_persona_thread(db: Session, user: User, persona_name: str, persona_prompt: str) -> "PersonaThread":
    thread = PersonaThread(user=user, persona_name=persona_name, persona_prompt=persona_prompt)
    db.add(thread)
    try:
        db.commit()
        db.refresh(thread)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create persona thread.") from exc
    return thread


def find_persona_thread_by_name(db: Session, user: User, persona_name: str) -> Optional["PersonaThread"]:
    normalized_name = persona_name.strip().lower()
    return (
        db.query(PersonaThread)
        .filter(PersonaThread.user_id == user.id)
        .filter(func.lower(PersonaThread.persona_name) == normalized_name)
        .first()
    )


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", value.lower()).strip()


def detect_matching_persona_thread(
    message: str,
    threads: Sequence["PersonaThread"],
    active_thread: Optional["PersonaThread"],
) -> Optional["PersonaThread"]:
    normalized_message = normalize_text(message)
    if not normalized_message:
        return None
    tokens = normalized_message.split()
    if not tokens:
        return None

    switch_keywords = {"switch", "back", "return", "resume", "continue", "again"}
    best_thread: Optional["PersonaThread"] = None
    best_score = 0.0

    for thread in threads:
        if active_thread and thread.id == active_thread.id:
            continue
        normalized_name = normalize_text(thread.persona_name)
        if not normalized_name:
            continue
        name_tokens = normalized_name.split()
        if not name_tokens:
            continue

        matches = sum(1 for token in name_tokens if token in tokens)
        if normalized_name in normalized_message:
            matches = len(name_tokens)

        score = matches / len(name_tokens)
        if score > best_score:
            best_score = score
            best_thread = thread

    if not best_thread:
        return None

    if best_score >= 1.0:
        return best_thread

    if best_score >= 0.5 and any(keyword in tokens for keyword in switch_keywords):
        return best_thread

    return None


def store_message(db: Session, thread: "PersonaThread", role: str, content: str, embedding: Optional[List[float]]) -> "Message":
    message = Message(thread=thread, role=role, content=content.strip(), embedding=serialize_embedding(embedding))
    db.add(message)
    try:
        db.commit()
        db.refresh(message)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to store chat message.") from exc
    return message


def ensure_general_thread(db: Session, user: User) -> "PersonaThread":
    thread = (
        db.query(PersonaThread)
        .filter(PersonaThread.user_id == user.id, PersonaThread.persona_name == GENERAL_PERSONA_NAME)
        .first()
    )
    if thread:
        return thread
    return create_persona_thread(db, user, GENERAL_PERSONA_NAME, GENERAL_PERSONA_PROMPT)


@app.get("/users/{user_id}", response_class=HTMLResponse)
async def get_user_space(
    user_id: int,
    request: Request,
    thread_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    space = ensure_user_space(user, db)

    general_thread = ensure_general_thread(db, user)
    threads = db.query(PersonaThread).filter(PersonaThread.user_id == user.id).order_by(PersonaThread.created_at.asc()).all()
    active_thread: Optional[PersonaThread] = None
    if thread_id:
        active_thread = next((t for t in threads if t.id == thread_id), None)
    if not active_thread:
        active_thread = next((t for t in threads if t.id == general_thread.id), None)
    if not active_thread and threads:
        active_thread = threads[-1]
    messages: List[Message] = []
    if active_thread:
        messages = (
            db.query(Message)
            .filter(Message.thread_id == active_thread.id)
            .order_by(Message.created_at.asc())
            .all()
        )

    return templates.TemplateResponse(
        "user_space.html",
        {
            "request": request,
            "user": user,
            "space": space,
            "threads": threads,
            "active_thread": active_thread,
            "messages": messages,
            "general_persona_name": GENERAL_PERSONA_NAME,
        },
    )


@app.post("/users/{user_id}/threads/{thread_id}/delete")
async def delete_persona_thread(
    user_id: int,
    thread_id: int,
    db: Session = Depends(get_db),
):
    thread = (
        db.query(PersonaThread)
        .filter(PersonaThread.user_id == user_id, PersonaThread.id == thread_id)
        .first()
    )
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found.")
    if thread.persona_name == GENERAL_PERSONA_NAME:
        raise HTTPException(status_code=400, detail="Cannot delete the general persona.")

    try:
        db.delete(thread)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete persona thread.") from exc

    response = RedirectResponse(url=f"/users/{user_id}", status_code=303)
    return response


@app.post("/users/{user_id}/delete")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    try:
        db.delete(user)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete user.") from exc

    response = RedirectResponse(url="/", status_code=303)
    return response


@app.post("/users")
async def create_user(
    name: str = Form(...),
    avatar: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    avatar_path: Optional[str] = None
    if avatar and avatar.filename:
        avatar_suffix = Path(avatar.filename).suffix or ".png"
        avatar_filename = f"{uuid4().hex}{avatar_suffix}"
        destination = AVATAR_DIR / avatar_filename
        content = await avatar.read()
        destination.write_bytes(content)
        avatar_path = f"/static/avatars/{avatar_filename}"

    user = User(name=name.strip(), avatar_path=avatar_path)
    db.add(user)
    try:
        db.commit()
        db.refresh(user)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create user.") from exc

    response = RedirectResponse(url="/", status_code=303)
    return response


@app.post("/users/{user_id}/chat")
async def chat_with_persona(
    user_id: int,
    message: str = Form(...),
    thread_id: Optional[int] = Form(None),
    db: Session = Depends(get_db),
):
    content = message.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    ensure_user_space(user, db)
    general_thread = ensure_general_thread(db, user)

    threads = (
        db.query(PersonaThread)
        .filter(PersonaThread.user_id == user.id)
        .order_by(PersonaThread.created_at.asc())
        .all()
    )

    active_thread: Optional[PersonaThread] = None
    if thread_id:
        active_thread = (
            db.query(PersonaThread)
            .filter(PersonaThread.user_id == user.id, PersonaThread.id == thread_id)
            .first()
        )
        if not active_thread:
            raise HTTPException(status_code=404, detail="Thread not found.")

    if not active_thread:
        active_thread = general_thread

    matched_thread = detect_matching_persona_thread(content, threads, active_thread)
    if matched_thread:
        active_thread = matched_thread

    existing_messages = (
        db.query(Message)
        .filter(Message.thread_id == active_thread.id)
        .order_by(Message.created_at.asc())
        .all()
    )

    user_embedding = await run_in_threadpool(bot_service.embed_text, content)
    context_snippets = retrieve_relevant_messages(user_embedding, existing_messages)

    agent_input = {
        "message": content,
        "user_name": user.name,
        "active_persona_prompt": active_thread.persona_prompt,
        "context_snippets": context_snippets,
    }

    agent_result = await run_in_threadpool(persona_agent.invoke, agent_input)
    action = agent_result["action"]
    reply = agent_result["reply"]

    if action == "create_persona":
        persona_details = agent_result["persona_details"]
        persona_name = persona_details.persona_name.strip()
        existing_thread = find_persona_thread_by_name(db, user, persona_name)
        if existing_thread:
            active_thread = existing_thread
            existing_messages = (
                db.query(Message)
                .filter(Message.thread_id == active_thread.id)
                .order_by(Message.created_at.asc())
                .all()
            )
        else:
            active_thread = create_persona_thread(
                db,
                user,
                persona_name,
                persona_details.persona_prompt,
            )
            existing_messages = []

    store_message(db, active_thread, "user", content, user_embedding)

    bot_embedding = await run_in_threadpool(bot_service.embed_text, reply)
    store_message(db, active_thread, "bot", reply, bot_embedding)

    response = RedirectResponse(
        url=f"/users/{user.id}?thread_id={active_thread.id}",
        status_code=303,
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)