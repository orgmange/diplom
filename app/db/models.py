from sqlalchemy import Column, String, JSON, DateTime, Text
from sqlalchemy.sql import func
from app.db.database import Base

class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, nullable=False, default="pending")
    result = Column(JSON, nullable=True) # Will store XML string or JSON dict
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Example(Base):
    __tablename__ = "examples"

    id = Column(String, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    json_output = Column(Text, nullable=False) # Store as string for easier search if needed
    doc_type = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
