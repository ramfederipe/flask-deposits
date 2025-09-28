import os
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

Base = declarative_base()

# --- Engine and Session from .env ---
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    echo=True
)
SessionLocal = sessionmaker(bind=engine)

class Deposit(Base):
    __tablename__ = "deposits"

    id = Column(Integer, primary_key=True, index=True)
    deposit_type = Column(String(255))
    merchant_code = Column(String(255))
    customer = Column(String(255))
    txnid = Column(String, unique=True, nullable=False)
    currency = Column(String(50))
    bank = Column(String(255))
    from_account = Column(String(255))
    to_account = Column(String(255), nullable=False)
    agent_number = Column(String(50))   # ✅ new
    shop_name = Column(String(255))     # ✅ new
    amount = Column(Float)
    original_amount = Column(Float)
    rebalance = Column(Float)
    fee = Column(Float)
    status = Column(String(255))
    audit = Column(String(255))
    note_message = Column(Text)
    refcode = Column(String(255))
    approve_by = Column(String(255))
    matched_by = Column(String(255))
    confirm_by = Column(String(255))
    created_time = Column(DateTime)
    updated_time = Column(DateTime)
    transfer_time = Column(DateTime)
    imported_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, onupdate=func.now())
    deleted = Column(Boolean, default=False)


class Withdrawal(Base):
    __tablename__ = "withdrawals"

    id = Column(Integer, primary_key=True)
    merchant_code = Column(String(255))
    spid = Column(String(255))
    customer = Column(String(255))
    txnid = Column(String, unique=True, nullable=False)
    currency = Column(String(50))
    bank = Column(String(255))
    from_account = Column(String(255), nullable=False)
    to_account = Column(String(255))
    agent_number = Column(String(50))   # ✅ new
    shop_name = Column(String(255))     # ✅ new
    amount = Column(Float)
    fee = Column(Float)
    r_bal = Column(Float)
    ref_code = Column(String(255))
    created_time = Column(DateTime)
    updated_time = Column(DateTime)
    transfered_time = Column(DateTime)
    status = Column(String(255))
    audit = Column(String(255))
    note_message = Column(Text)
    approve_batches_by = Column(String(255))
    transaction_note = Column(String(255))
    approve_by = Column(String(255))
    matched_by = Column(String(255))
    confirm_by = Column(String(255))
    imported_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, onupdate=func.now())
    deleted = Column(Boolean, default=False)



# --- Wallet model ---
class Wallet(Base):
    __tablename__ = "wallets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    currency_name = Column(String(255))
    deposit_channels = Column(String(255))
    merchant_code = Column(String(255))
    group_code = Column(String(255))
    bank_code = Column(String(255))
    account_name = Column(String(255))
    short_name = Column(String(255))
    account_number = Column(String(255))
    phone = Column(String(255))
    status = Column(String(255))
    timestamp = Column(DateTime)

# --- Limit model ---
class Limit(Base):
    __tablename__ = "limits"
    id = Column(Integer, primary_key=True, autoincrement=True)
    bank = Column(String(255))
    channel = Column(String(255))
    group = Column(String(255))
    account = Column(String(255))
    agent_number = Column(String(50))  # new
    shop_name = Column(String(255))    # new
    balance = Column(String(255))
    balance_limit = Column(String(255))
    dp_limit = Column(String(255))
    total_dp = Column(String(255))
    wd_limit = Column(String(255))
    total_wd = Column(String(255))
    update_time = Column(String(255))
    login = Column(String(255))
    status = Column(String(255))
    timestamp = Column(String(255), default=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))


# --- Agent model ---
class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True)
    shop = Column(String)
    status = Column(String)
    opening_balance = Column(Float)
    group_code = Column(String)  # <- use group_code instead of group



class Adjustment(Base):
    __tablename__ = "adjustments"
    id = Column(Integer, primary_key=True)
    amount = Column(Float)
    created_time = Column(DateTime)
    from_account = Column(String)
    to_account = Column(String)



class Setting(Base):
    __tablename__ = "settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    enable_delete = Column(Boolean, default=False)

class Sdp(Base):
    __tablename__ = 'sdps'

    id = Column(Integer, primary_key=True, autoincrement=True)
    shop = Column(String, unique=True, nullable=False)
    sdp = Column(Float, default=0)
    group_code = Column(String)
    chat_id = Column(String)
    tg_link = Column(String)
    remarks = Column(String)

class Settlement(Base):
    __tablename__ = 'settlements'  # Make sure this matches your DB table
    id = Column(Integer, primary_key=True)
    agent = Column(String)
    brand = Column(String)
    amount = Column(Float)
    fee = Column(Float)
    remarks = Column(String)
    mc = Column(String)
    purpose = Column(String)
    wallet = Column(String)
    date = Column(DateTime)

class TopUp(Base):
    __tablename__ = "topups"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=func.now())       # Excel Column A
    brand = Column(String(100))                       # Column B
    from_agent = Column(String(255))                # Column C (FROM)
    remarks_d = Column(String(255))                   # Column D (Remarks D)
    mc = Column(String(100))                          # Column E
    type = Column(String(100))                        # Column F
    to_agent = Column(String(100))                    # Column G
    brand_to = Column(String(100))                    # Column H
    wallet = Column(String(100))                      # Column I
    amount_process = Column(Float, default=0.0)       # Column J
    fee = Column(Float, default=0.0)                  # Column K
    remarks_l = Column(String(255))                   # Column L (Remarks L)
    updated_by = Column(String(100))                  # Column M
    status = Column(String(100))                      # Column N
    checker = Column(String(100))                     # Column O

class Note(Base):
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True)
    page_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# --- Initialize DB ---
def init_db():
    Base.metadata.create_all(bind=engine)

