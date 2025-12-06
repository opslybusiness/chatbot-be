# chat_memory.py
import os
from sqlmodel import SQLModel, Field, create_engine, Session, select
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy import Column
from dotenv import load_dotenv
from embeddings_util import get_engine
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
from uuid import UUID as UUIDType

load_dotenv()



# --- MODEL ---
class ChatMessage(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    user_id: UUIDType = Field(
        ...,
        sa_column=Column(PGUUID(as_uuid=True)),  # Store as UUID type in database, Python UUID object
        description="User ID (UUID from auth.users)"
    )  
    sender: str  # 'user' or 'ai'
    content: str

# create table if not exists
engine = get_engine()
SQLModel.metadata.create_all(engine)

# --- HISTORY CLASS ---
class PostgresChatMessageHistory:
    def __init__(self, user_id: str):
        self.user_id = user_id  # user_id replaces session_id
        self.engine = get_engine()

    @property
    def messages(self):
        with Session(self.engine) as session:
            # ✅ Filter by user_id only (user_id replaces session_id)
            # Convert string user_id to UUID for comparison
            user_uuid = UUIDType(self.user_id) if isinstance(self.user_id, str) else self.user_id
            result = session.exec(
                select(ChatMessage)
                .where(ChatMessage.user_id == user_uuid)
                .order_by(ChatMessage.id)
            ).all()
            msgs = []
            for msg in result:
                if msg.sender == "user":
                    msgs.append(HumanMessage(content=msg.content))
                else:
                    msgs.append(AIMessage(content=msg.content))
            return msgs
    def add_messages(self, messages: list[BaseMessage]):
        """
        Accepts a list of HumanMessage or AIMessage and adds them to the DB.
        """
        for msg in messages:
            sender = "user" if msg.type == "human" else "ai"
            self._add_message(sender, msg.content)


    def add_user_message(self, message: str):
        self._add_message("user", message)

    def add_ai_message(self, message: str):
        self._add_message("ai", message)

    def _add_message(self, sender: str, content: str):
        with Session(self.engine) as session:
            # Convert string user_id to UUID if needed
            user_uuid = UUIDType(self.user_id) if isinstance(self.user_id, str) else self.user_id
            chat_msg = ChatMessage(
                user_id=user_uuid,
                sender=sender,
                content=content
            )
            session.add(chat_msg)
            session.commit()

    def clear(self):
        with Session(self.engine) as session:
            # ✅ Clear all messages for this user (user_id replaces session_id)
            # Convert string user_id to UUID for comparison
            user_uuid = UUIDType(self.user_id) if isinstance(self.user_id, str) else self.user_id
            session.exec(
                select(ChatMessage)
                .where(ChatMessage.user_id == user_uuid)
                .delete()
            )
            session.commit()
