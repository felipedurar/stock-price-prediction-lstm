import os

from sqlmodel import SQLModel, create_engine, Session

os.makedirs("data", exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/stock_data.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    os.makedirs("data", exist_ok=True)
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"sslmode": "require"}
    )

def init_db():
    """Create all tables. Call this at app startup."""
    SQLModel.metadata.create_all(engine)

def get_session():
    """FastAPI dependency to get a DB session."""
    with Session(engine) as session:
        yield session
