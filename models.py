from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

Base = declarative_base()
engine = create_engine("sqlite:///deposits.db", echo=True)  # you can rename DB file if needed
SessionLocal = sessionmaker(bind=engine)


class Deposit(Base):
    __tablename__ = "deposits"

    id = Column(Integer, primary_key=True, index=True)
    deposit_type = Column(String)
    merchant_code = Column(String)
    customer = Column(String)
    txnid = Column(String)
    currency = Column(String)
    bank = Column(String)
    from_account = Column(String)
    to_account = Column(String)
    amount = Column(Float)
    original_amount = Column(Float)
    rebalance = Column(Float)
    fee = Column(Float)
    created_time = Column(DateTime)
    updated_time = Column(DateTime)
    transfer_time = Column(DateTime)
    status = Column(String)
    audit = Column(String)
    note_message = Column(String)
    refcode = Column(String)
    approve_by = Column(String)
    matched_by = Column(String)
    confirm_by = Column(String)
    last_updated = Column(DateTime)
    deleted = Column(Boolean, default=False)


# 👉 add this function
def init_db():
    Base.metadata.create_all(bind=engine)
