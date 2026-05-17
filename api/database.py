"""SQLAlchemy ORM models and session management for the SentinelAI API."""

from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from src.utils.config import APIConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_config = APIConfig()
DATABASE_URL: str = _config.database_url

Base = declarative_base()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Machine(Base):
    """A monitored industrial machine."""

    __tablename__ = "machines"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    location = Column(String, nullable=False)
    install_date = Column(DateTime, nullable=False)
    status = Column(String, nullable=False, default="active")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class SensorReading(Base):
    """A raw sensor reading snapshot for a machine."""

    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String, ForeignKey("machines.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    sensor_data = Column(Text, nullable=False)


class Prediction(Base):
    """A stored RUL prediction with risk and explainability metadata."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String, ForeignKey("machines.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    rul = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False, default=0.0)
    confidence_lower = Column(Float, nullable=False, default=0.0)
    confidence_upper = Column(Float, nullable=False, default=0.0)
    risk_level = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False, default=0.0)
    top_features = Column(Text, nullable=False, default="[]")
    model_contributions = Column(Text, nullable=False, default="{}")


class Alert(Base):
    """An operational alert raised for a machine."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String, ForeignKey("machines.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    alert_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    message = Column(String, nullable=False)
    acknowledged = Column(Boolean, nullable=False, default=False)


def get_db() -> Generator[Session, None, None]:
    """Yield a database session for use as a FastAPI dependency.

    Yields:
        An open :class:`~sqlalchemy.orm.Session` closed on completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Initialized database schema at %s", DATABASE_URL)
