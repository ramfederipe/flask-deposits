from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from models import Base

class Withdrawal(Base):
    __tablename__ = "withdrawals"

    id = Column(Integer, primary_key=True)
    merchant_code = Column("Merchant", String)
    spid = Column("SPID", String)
    customer = Column("Customer", String)
    txnid = Column("TXNID", String)
    currency = Column("Currency", String)
    bank = Column("Bank", String)
    from_account = Column("From Account", String)
    to_account = Column("To Account", String)
    amount = Column("Amount", Float)
    fee = Column("Fee", Float)
    r_bal = Column("R.Bal", Float)
    ref_code = Column("RefCode", String)
    created_time = Column("Created Time", DateTime)
    updated_time = Column("Updated Time", DateTime)
    transfered_time = Column("Transfered Time", DateTime)
    status = Column("Status", String)
    audit = Column("Audit", String)
    note_message = Column("Note Message", String)
    approve_batches_by = Column("Approve Batches By", String)
    transaction_note = Column("Transaction Note", String)
    approve_by = Column("Approve By", String)
    matched_by = Column("Matched By", String)
    confirm_by = Column("Confirm By", String)
