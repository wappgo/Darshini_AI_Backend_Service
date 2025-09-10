# models.py
import uuid
from pydantic import Field, BaseModel
from beanie import Document
from datetime import datetime
from typing import List, Optional, Any, Dict
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class HistoryMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4()}")
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Optional field to store structured tool results
    tool_results: Optional[List[Dict[str, Any]]] = None

class Conversation(Document):
    session_id: str = Field(..., unique=True)
    user_id: Any  # Can be int or str depending on your user system
    history: List[HistoryMessage] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "darshini_conversations" # Collection name in MongoDB