from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

Base = declarative_base()
engine = create_engine("sqlite:///deposits.db", echo=True)  # SQLite DB file
SessionLocal = sessionmaker(bind=engine)


class Deposit(Base):
    __tablename__ = "deposits"

    id = Column(Integer, primary_key=True, index=True)
    deposit_type = Column(String(50))       # e.g., "manual", "auto"
    merchant_code = Column(String(50))      # short code
    customer = Column(String(100))          # username or customer name
    txnid = Column(String(100))             # transaction id
    currency = Column(String(10))           # e.g., "USD", "PHP"
    bank = Column(String(100))
    from_account = Column(String(100))
    to_account = Column(String(100))
    amount = Column(Float)
    original_amount = Column(Float)
    rebalance = Column(Float)
    fee = Column(Float)
    created_time = Column(DateTime)
    updated_time = Column(DateTime)
    transfer_time = Column(DateTime)
    status = Column(String(50))             # e.g., "pending", "approved"
    audit = Column(String(100))
    note_message = Column(String(255))      # longer text, message/note
    refcode = Column(String(100))
    approve_by = Column(String(50))
    matched_by = Column(String(50))
    confirm_by = Column(String(50))
    last_updated = Column(DateTime)
    deleted = Column(Boolean, default=False)


# 👉 call this to create tables
def init_db():
    Base.metadata.create_all(bind=engine)
