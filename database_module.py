# database/db.py

import sqlite3
import uuid
from contextlib import contextmanager

# =========================================================
# Database Configuration
# =========================================================

DB_NAME = "diabetes.db"


@contextmanager
def get_db():
    """
    Context manager for SQLite connection
    """
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()




def init_db():
    """
    Create all required tables if they do not exist
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # User profile table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            user_id TEXT PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            weight REAL,
            height REAL,
            diabetes_type TEXT,
            diagnosis_year INTEGER,
            hba1c REAL,
            profile_completed INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)

        # Insulin schedule
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS insulin_schedule (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            insulin_name TEXT,
            dosage TEXT,
            timing TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)

        # Medications
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS medications (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            medicine_name TEXT,
            dosage TEXT,
            frequency TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)

        # Chat history
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            role TEXT,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)


# =========================================================
# User Authentication Functions
# =========================================================

def create_user(email: str, password_hash: str) -> str:
    """
    Create a new user
    """
    user_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (user_id, email, password_hash) VALUES (?, ?, ?)",
            (user_id, email, password_hash)
        )
    return user_id


def get_user_by_email(email: str):
    """
    Fetch user by email
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str):
    """
    Fetch user by user_id
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()
    return dict(row) if row else None


# =========================================================
# Profile Management Functions
# =========================================================

def save_user_profile(user_id: str, profile: dict):
    """
    Insert or update user profile and mark it as completed
    """
    with get_db() as conn:
        conn.execute("""
        INSERT OR REPLACE INTO user_profile (
            user_id, age, gender, weight, height,
            diabetes_type, diagnosis_year, hba1c, profile_completed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            user_id,
            profile.get("age"),
            profile.get("gender"),
            profile.get("weight"),
            profile.get("height"),
            profile.get("diabetes_type"),
            profile.get("diagnosis_year"),
            profile.get("hba1c")
        ))


def get_user_profile(user_id: str):
    """
    Retrieve user profile
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM user_profile WHERE user_id = ?",
            (user_id,)
        ).fetchone()
    return dict(row) if row else None


def is_profile_completed(user_id: str) -> bool:
    """
    Check if user profile is completed
    """
    profile = get_user_profile(user_id)
    return bool(profile and profile.get("profile_completed") == 1)


# =========================================================
# Insulin Schedule Functions
# =========================================================

def add_insulin(user_id: str, insulin_name: str, dosage: str, timing: str):
    with get_db() as conn:
        conn.execute("""
        INSERT INTO insulin_schedule (id, user_id, insulin_name, dosage, timing)
        VALUES (?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            user_id,
            insulin_name,
            dosage,
            timing
        ))


def get_insulin_schedule(user_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT insulin_name, dosage, timing FROM insulin_schedule WHERE user_id = ?",
            (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def delete_insulin_by_user(user_id: str):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM insulin_schedule WHERE user_id = ?",
            (user_id,)
        )


# =========================================================
# Medication Functions
# =========================================================

def add_medication(user_id: str, medicine_name: str, dosage: str, frequency: str):
    with get_db() as conn:
        conn.execute("""
        INSERT INTO medications (id, user_id, medicine_name, dosage, frequency)
        VALUES (?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            user_id,
            medicine_name,
            dosage,
            frequency
        ))


def get_medications(user_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT medicine_name, dosage, frequency FROM medications WHERE user_id = ?",
            (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def delete_medications_by_user(user_id: str):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM medications WHERE user_id = ?",
            (user_id,)
        )


# =========================================================
# Chat History Functions (Conversation Memory)
# =========================================================

def save_chat_message(user_id: str, role: str, message: str):
    """
    role: 'user' or 'assistant'
    """
    with get_db() as conn:
        conn.execute("""
        INSERT INTO chat_history (id, user_id, role, message)
        VALUES (?, ?, ?, ?)
        """, (
            str(uuid.uuid4()),
            user_id,
            role,
            message
        ))


def get_chat_history(user_id: str, limit: int = 10):
    """
    Fetch last N chat messages for context
    """
    with get_db() as conn:
        rows = conn.execute("""
        SELECT role, message, timestamp
        FROM chat_history
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """, (user_id, limit)).fetchall()

    # reverse to maintain chronological order
    return [dict(row) for row in reversed(rows)]


def clear_chat_history(user_id: str):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM chat_history WHERE user_id = ?",
            (user_id,)
        )


# =========================================================
# Context Builder Helper (FOR RAG / GEMINI)
# =========================================================

def build_patient_context(user_id: str) -> dict:
    """
    Returns a complete patient context object
    to inject into the LLM prompt
    """
    return {
        "profile": get_user_profile(user_id),
        "insulin_schedule": get_insulin_schedule(user_id),
        "medications": get_medications(user_id),
        "chat_history": get_chat_history(user_id)
    }
