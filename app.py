# Standard library
import math, os, re, csv, time
from datetime import datetime, timedelta, timezone, date
from calendar import monthrange
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile
from io import StringIO
from dotenv import load_dotenv
from waitress import serve

# Third-party
import pandas as pd
import numpy as np
import gspread
import psycopg2
import whisper
import openai
import tempfile
from openai import OpenAI
from google.oauth2.service_account import Credentials
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, Response, send_file, current_app
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, distinct, func, cast, Date, extract, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert

# Local imports
from models import (
    Base, engine, Deposit, Withdrawal, Wallet, Limit, Agent,
    Adjustment, Setting, Sdp, Settlement, TopUp, Note,
    SessionLocal, init_db
)



session = SessionLocal()
sdps = session.query(Sdp).all()
for s in sdps:
    print(s.id, s.shop, s.sdp, s.group_code, s.chat_id, s.tg_link, s.remarks)
session.close()



# Create all tables defined in your models
print("OpeningBalance table created!")

def init_settings():
    with SessionLocal() as session:
        if not session.query(Setting).first():
            session.add(Setting(enable_delete=False))
            session.commit()


# Load .env file (local development)
# --- Load .env first ---
load_dotenv()

# --- Flask setup ---
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# --- Initialize DB ---
init_db()

def init_settings():
    with SessionLocal() as session:
        if not session.query(Setting).first():
            session.add(Setting(enable_delete=False))
            session.commit()

# --- OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Whisper model ---
# Make sure whisper_models/ exists and contains the base model
whisper_model = whisper.load_model("base", download_root="whisper_models")

# --- Example query to test DB connection ---
if __name__ == "__main__":
    with SessionLocal() as session:
        sdps = session.query(Sdp).all()
        for s in sdps:
            print(s.id, s.shop, s.sdp, s.group_code, s.chat_id, s.tg_link, s.remarks)
    print("✅ App initialized successfully!")


# --- Create tables ---
try:
    Base.metadata.create_all(engine)
    print("✅ Database tables created or already exist")
except Exception as e:
    print("❌ Failed to create tables:", e)

def render_pagination(pagination):
    # return HTML string for pagination
    html = '<div class="pagination">'
    for page in range(1, pagination.pages + 1):
        html += f'<a href="?page={page}">{page}</a>'
    html += '</div>'
    return html



# --- Session wrapper ---
def with_session(func):
    def wrapper(*args, **kwargs):
        session = SessionLocal()
        try:
            return func(session, *args, **kwargs)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    wrapper.__name__ = func.__name__
    return wrapper

def normalize_key(s: str) -> str:
    """
    Make a stable join key from any 'account/agent' string.
    Removes all non-alphanumerics and uppercases the rest.
    E.g. 'Juliet - NG-01 ' -> 'JULIETNG01'
    """
    if not s:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).upper()





def extract_shop_name(account: str) -> str:
    """Extract shop name from account string like '021512151215 - HYDRA-BK-12'."""
    if not account:
        return ""
    parts = str(account).split(" - ")
    return parts[-1].strip().upper()  # normalize

def split_account(account_str):
    """Split '01714409773 - BELAS-BK-23' → agent_number, shop_name"""
    if not account_str:
        return None, None
    parts = account_str.split(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, account_str.strip()



def parse_float(value):
    """
    Safely convert value to float. Returns 0.0 for None, '', 'nan', '-', '--', 'N/A'
    """
    if value is None:
        return 0.0
    try:
        val_str = str(value).strip().replace(",", "")
        if val_str.lower() in ["", "nan", "-", "--", "n/a"]:
            return 0.0
        return float(val_str)
    except (ValueError, TypeError):
        return 0.0

def parse_datetime(value):
    """Try to convert Excel/CSV value into datetime, fallback to None"""
    if not value or str(value).strip() in ["-", "NaN", ""]:
        return None
    if isinstance(value, datetime):
        return value  # already datetime
    try:
        # Try common formats - adjust depending on your Excel export
        return datetime.strptime(str(value).strip(), "%d/%m/%Y %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(str(value).strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def safe_float(value):
    try:
        if value in (None, "", "-", "--", "NaN"):  # treat as zero
            return 0.0
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0




def safe_balance(value):
    """Convert balance to float, fallback to 0.0"""
    if value in (None, "", "-", "--", "NaN"):
        return 0.0
    try:
        val = str(value).replace(",", "").strip()
        if val.startswith("(") and val.endswith(")"):
            val = "-" + val[1:-1]  # handle (22,000.00)
        return float(val)
    except Exception:
        return 0.0

def bulk_insert_deposits_safe(session, records):
    """Safely bulk insert deposit records, skip invalid rows instead of failing."""
    inserted_count = 0
    failed_rows = []

    for idx, record in enumerate(records, start=1):
        try:
            # --- Normalize keys (lowercase, underscores, strip) ---
            record = {k.strip().lower().replace(" ", "_"): v for k, v in record.items()}

            # --- Extract agent_number and shop_name from to_account ---
            to_account = str(record.get("to_account") or "")
            agent_number, shop_name = split_account(to_account)

            # --- Map fields from Excel to DB ---
            row = {
                "deposit_type": record.get("deposit_type"),
                "merchant_code": record.get("merchant_code"),
                "customer": record.get("customer"),
                "txnid": str(record.get("txnid") or ""),
                "currency": record.get("currency"),
                "bank": record.get("bank"),
                "from_account": str(record.get("from_account") or ""),
                "to_account": to_account,
                "agent_number": agent_number,  # ✅ extracted
                "shop_name": shop_name,        # ✅ extracted
                "amount": clean_float(record.get("amount")),
                "original_amount": clean_float(record.get("original_amount")),
                "rebalance": clean_float(record.get("rebalance")),
                "fee": clean_float(record.get("fee")),
                "status": record.get("status"),
                "audit": record.get("audit"),
                "note_message": record.get("note_message"),
                "refcode": record.get("refcode"),
                "approve_by": record.get("approve_by"),
                "matched_by": record.get("matched_by"),
                "confirm_by": record.get("confirm_by"),
                "created_time": parse_excel_datetime(record.get("created_time")),
                "updated_time": parse_excel_datetime(record.get("updated_time")),
                "transfer_time": parse_excel_datetime(record.get("transfer_time")),
            }

            # --- Validation ---
            if not row["to_account"]:  # skip if required field missing
                failed_rows.append((idx, "Missing to_account"))
                continue

            # --- Insert ---
            deposit = Deposit(**row)
            session.add(deposit)
            inserted_count += 1

        except Exception as e:
            failed_rows.append((idx, str(e)))
            continue

    # Commit once after all inserts
    session.commit()
    return inserted_count, failed_rows





@app.template_filter()
def nan_to_empty(value):
    """
    Converts None or NaN values to empty string.
    """
    if value is None:
        return ""
    try:
        # check for float NaN
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return value


@app.template_filter()
def datetime_format(value, fmt="%d/%m/%Y %H:%M:%S"):
    """
    Format datetime objects to dd/mm/yyyy hh:mm:ss.
    If value is None or invalid, return empty string.
    """
    if not value:
        return ""
    try:
        return value.strftime(fmt)
    except Exception:
        return str(value)

@app.template_filter("datetime")
def format_datetime(value, fmt="%d/%m/%Y %H:%M:%S"):
    """Format datetime to dd/mm/yyyy hh:mm:ss, or empty string if None."""
    if not value:
        return ""
    try:
        return value.strftime(fmt)
    except Exception:
        return str(value)


def clean_float(value):
    """Convert Excel numeric strings with commas into float safely."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None  # ✅ keep as NULL in DB

    try:
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None  # fallback if still invalid


def parse_excel_datetime(value):
    """
    Parse Excel/CSV date values safely.
    - Supports dd/mm/yyyy hh:mm:ss format
    - Handles Excel serials, empty values, and pandas Timestamps
    """
    if value is None or str(value).strip() in ["", "-", "NaN"]:
        return None

    if isinstance(value, datetime):
        return value

    # Excel numeric serials
    if isinstance(value, (int, float)):
        try:
            return pd.to_datetime(value, unit="D", origin="1899-12-30").to_pydatetime()
        except Exception:
            pass

    # Strings like 09/03/2025 06:15:32
    try:
        return pd.to_datetime(str(value).strip(), dayfirst=True, errors="coerce").to_pydatetime()
    except Exception:
        return None



# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")  # simple homepage

@app.route("/notebook")
def notebook():
    max_pages = 5  # total notebook pages
    with SessionLocal() as session:
        notes = session.query(Note).order_by(Note.page_number).all()
        notes_list = ["" for _ in range(max_pages)]
        for note in notes:
            if 0 <= note.page_number < max_pages:
                notes_list[note.page_number] = note.content or ""
    return render_template("notebook.html", notes_list=notes_list, now=datetime.utcnow())


@app.route("/notes/save", methods=["POST"])
def save_note():
    page_number = request.form.get("page_number")
    content = request.form.get("content")
    if page_number is None or content is None:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    page_number = int(page_number)
    with SessionLocal() as session:
        note = session.query(Note).filter_by(page_number=page_number).first()
        if note:
            note.content = content
        else:
            note = Note(page_number=page_number, content=content)
            session.add(note)
        session.commit()
    return jsonify({"status": "success"})



@app.route("/dashboard")
def dashboard():
    with SessionLocal() as session:
        # --- Month filter ---
        month_str = request.args.get("month")
        if month_str:
            year, month = map(int, month_str.split("-"))
        else:
            today = datetime.utcnow()
            year, month = today.year, today.month

        # First/last day of selected month
        start_date = datetime(year, month, 1)
        last_day = monthrange(year, month)[1]
        end_date = datetime(year, month, last_day, 23, 59, 59)

        deposits_per_bank = (
            session.query(
                Deposit.bank.label("bank"),
                func.count(Deposit.id).label("count"),
                func.coalesce(func.sum(Deposit.amount), 0).label("total_amount")
            )
            .filter(
                Deposit.status == "Successful",
                Deposit.deleted == False,
                Deposit.created_time >= start_date,
                Deposit.created_time <= end_date
            )
            .group_by(Deposit.bank)
            .order_by(func.coalesce(func.sum(Deposit.amount), 0).desc())  # optional: sort by amount
            .all()
        )

        # If you want withdrawals per bank too, use the same shape:
        withdrawals_per_bank = (
            session.query(
                Withdrawal.bank.label("bank"),
                func.count(Withdrawal.id).label("count"),
                func.coalesce(func.sum(Withdrawal.amount), 0).label("total_amount")
            )
            .filter(
                Withdrawal.status == "Successful",
                Withdrawal.deleted == False,
                Withdrawal.created_time >= start_date,
                Withdrawal.created_time <= end_date
            )
            .group_by(Withdrawal.bank)
            .order_by(func.coalesce(func.sum(Withdrawal.amount), 0).desc())
            .all()
        )





        # --- Daily deposits ---
        deposit_daily = session.query(
            func.date(Deposit.created_time).label("day"),
            func.coalesce(func.sum(Deposit.amount), 0).label("total"),
            func.count(Deposit.id).label("count")
        ).filter(
            Deposit.status == "Successful",
            Deposit.deleted == False,
            Deposit.created_time >= start_date,
            Deposit.created_time <= end_date
        ).group_by(func.date(Deposit.created_time)).all()

        # --- Daily withdrawals ---
        withdrawal_daily = session.query(
            func.date(Withdrawal.created_time).label("day"),
            func.coalesce(func.sum(Withdrawal.amount), 0).label("total")
        ).filter(
            Withdrawal.status == "Successful",
            Withdrawal.deleted == False,
            Withdrawal.created_time >= start_date,
            Withdrawal.created_time <= end_date
        ).group_by(func.date(Withdrawal.created_time)).all()

        # Build day labels (1–last_day of month)
        labels = [str(d) for d in range(1, last_day + 1)]

        # Map daily sums
        deposit_amount_map = {str(r.day.day): float(r.total) for r in deposit_daily}
        deposit_count_map = {str(r.day.day): int(r.count) for r in deposit_daily}
        withdrawal_map = {str(r.day.day): float(r.total) for r in withdrawal_daily}

        deposit_amount_data = [deposit_amount_map.get(str(d), 0.0) for d in range(1, last_day + 1)]
        deposit_count_data = [deposit_count_map.get(str(d), 0) for d in range(1, last_day + 1)]
        deposit_data = deposit_amount_data
        withdrawal_data = [withdrawal_map.get(str(d), 0.0) for d in range(1, last_day + 1)]

        # --- Deposit Totals ---
        deposit_total_count = session.query(func.count(Deposit.id)).filter(
            Deposit.status == "Successful",
            Deposit.deleted == False,
            Deposit.created_time >= start_date,
            Deposit.created_time <= end_date
        ).scalar()

        deposit_total_amount = session.query(func.coalesce(func.sum(Deposit.amount), 0)).filter(
            Deposit.status == "Successful",
            Deposit.deleted == False,
            Deposit.created_time >= start_date,
            Deposit.created_time <= end_date
        ).scalar()

        deposit_merchant_stats = session.query(
            Deposit.merchant_code.label("merchant_code"),
            func.count(Deposit.id).label("count"),
            func.coalesce(func.sum(Deposit.amount), 0).label("total_amount")
        ).filter(
            Deposit.status == "Successful",
            Deposit.deleted == False,
            Deposit.created_time >= start_date,
            Deposit.created_time <= end_date
        ).group_by(Deposit.merchant_code).all()

        deposit_bank_stats = session.query(
            Deposit.bank.label("bank"),
            func.count(Deposit.id).label("count"),
            func.coalesce(func.sum(Deposit.amount), 0).label("total_amount")
        ).filter(
            Deposit.status == "Successful",
            Deposit.deleted == False,
            Deposit.created_time >= start_date,
            Deposit.created_time <= end_date
        ).group_by(Deposit.bank).all()

        # --- Withdrawal Totals ---
        withdrawal_total_count = session.query(func.count(Withdrawal.id)).filter(
            Withdrawal.status == "Successful",
            Withdrawal.deleted == False,
            Withdrawal.created_time >= start_date,
            Withdrawal.created_time <= end_date
        ).scalar()

        withdrawal_total_amount = session.query(func.coalesce(func.sum(Withdrawal.amount), 0)).filter(
            Withdrawal.status == "Successful",
            Withdrawal.deleted == False,
            Withdrawal.created_time >= start_date,
            Withdrawal.created_time <= end_date
        ).scalar()

        withdrawal_merchant_stats = session.query(
            Withdrawal.merchant_code.label("merchant_code"),
            func.count(Withdrawal.id).label("count"),
            func.coalesce(func.sum(Withdrawal.amount), 0).label("total_amount")
        ).filter(
            Withdrawal.status == "Successful",
            Withdrawal.deleted == False,
            Withdrawal.created_time >= start_date,
            Withdrawal.created_time <= end_date
        ).group_by(Withdrawal.merchant_code).all()

        withdrawal_bank_stats = session.query(
            Withdrawal.bank.label("bank"),
            func.count(Withdrawal.id).label("count"),
            func.coalesce(func.sum(Withdrawal.amount), 0).label("total_amount")
        ).filter(
            Withdrawal.status == "Successful",
            Withdrawal.deleted == False,
            Withdrawal.created_time >= start_date,
            Withdrawal.created_time <= end_date
        ).group_by(Withdrawal.bank).all()

        # --- Monthly aggregates (last 12 months ending selected month) ---
        def month_start(dt):
            return datetime(dt.year, dt.month, 1)

        selected_month_start = datetime(year, month, 1)
        first_month = (selected_month_start.replace(day=1) - timedelta(days=365)).replace(day=1)

        deposit_monthly = session.query(
            func.date_trunc('month', Deposit.created_time).label('m'),
            func.coalesce(func.sum(Deposit.amount), 0).label('amount')
        ).filter(
            Deposit.status == "Successful",
            Deposit.deleted == False,
            Deposit.created_time >= first_month,
            Deposit.created_time <= end_date
        ).group_by('m').order_by('m').all()

        withdrawal_monthly = session.query(
            func.date_trunc('month', Withdrawal.created_time).label('m'),
            func.coalesce(func.sum(Withdrawal.amount), 0).label('amount')
        ).filter(
            Withdrawal.status == "Successful",
            Withdrawal.deleted == False,
            Withdrawal.created_time >= first_month,
            Withdrawal.created_time <= end_date
        ).group_by('m').order_by('m').all()

        # Build continuous month list (12 months ending at selected month)
        months = []
        cur = selected_month_start
        for _ in range(12):
            months.append((cur.year, cur.month))
            if cur.month == 1:
                cur = datetime(cur.year - 1, 12, 1)
            else:
                cur = datetime(cur.year, cur.month - 1, 1)
        months.reverse()

        month_labels = [f"{y}-{m:02d}" for (y, m) in months]

        dep_month_map = {f"{r.m.year}-{r.m.month:02d}": float(r.amount) for r in deposit_monthly}
        wd_month_map = {f"{r.m.year}-{r.m.month:02d}": float(r.amount) for r in withdrawal_monthly}

        deposit_month_amounts = [dep_month_map.get(lbl, 0.0) for lbl in month_labels]
        withdrawal_month_amounts = [wd_month_map.get(lbl, 0.0) for lbl in month_labels]

        # Chart labels and values (deposits per bank)
        chart_labels = [row.bank for row in deposits_per_bank]
        chart_deposit_amounts = [float(row.total_amount) for row in deposits_per_bank]
        chart_deposit_counts = [int(row.count) for row in deposits_per_bank]

        # And for withdrawals if desired
        chart_withdrawal_labels = [row.bank for row in withdrawals_per_bank]
        chart_withdrawal_amounts = [float(row.total_amount) for row in withdrawals_per_bank]

        return render_template("dashboard.html",
                               deposit_total_count=deposit_total_count,
                               deposit_total_amount=deposit_total_amount,
                               deposit_merchant_stats=deposit_merchant_stats,
                               deposit_bank_stats=deposit_bank_stats,
                               withdrawal_total_count=withdrawal_total_count,
                               withdrawal_total_amount=withdrawal_total_amount,
                               withdrawal_merchant_stats=withdrawal_merchant_stats,
                               withdrawal_bank_stats=withdrawal_bank_stats,
                               labels=labels,
                               deposit_data=deposit_data,
                               withdrawal_data=withdrawal_data,
                               deposit_count_data=deposit_count_data,
                               deposit_amount_data=deposit_amount_data,
                               month_labels=month_labels,
                               deposit_month_amounts=deposit_month_amounts,
                               withdrawal_month_amounts=withdrawal_month_amounts,

                               # charts
                               deposits_per_bank=deposits_per_bank,
                               withdrawals_per_bank=withdrawals_per_bank,
                               chart_labels=chart_labels,
                               chart_deposit_amounts=chart_deposit_amounts,
                               chart_deposit_counts=chart_deposit_counts,
                               chart_withdrawal_labels=chart_withdrawal_labels,
                               chart_withdrawal_amounts=chart_withdrawal_amounts
                               )


@app.route("/deposit", methods=["GET", "POST"])
def deposit():
    with SessionLocal() as session:
        # --- fetch setting ---
        setting = session.query(Setting).first()
        if not setting and request.method == "POST":
            try:
                setting = Setting(enable_delete=False)
                session.add(setting)
                session.commit()
            except Exception:
                session.rollback()
                setting = session.query(Setting).first()

        # --- Pagination ---
        page = request.args.get("page", default=1, type=int)
        per_page = request.args.get("per_page", default=20, type=int)

        # --- Handle file upload ---
        if request.method == "POST":
            file = request.files.get("excel_file")
            if file:
                try:
                    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
                    os.close(fd)

                    if file.filename.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(file, engine="openpyxl")
                        df.to_csv(tmp_path, index=False)
                    else:
                        file.save(tmp_path)

                    chunksize = 10_000
                    inserted = 0
                    skipped = 0

                    for chunk in pd.read_csv(tmp_path, chunksize=chunksize, dtype=str):
                        # normalize columns
                        chunk.columns = chunk.columns.str.strip().str.lower().str.replace(" ", "_")
                        chunk = chunk.where(pd.notnull(chunk), None)

                        # normalize txnid
                        chunk["txnid"] = chunk["txnid"].astype(str).str.strip()
                        chunk = chunk[chunk["txnid"] != ""]

                        # --- Normalize account columns ---
                        def split_account(account_str):
                            if not account_str:
                                return None, None
                            parts = account_str.split(" - ", 1)
                            return parts[0].strip(), parts[1].strip() if len(parts) == 2 else (None, account_str.strip())

                        chunk['agent_number'], chunk['shop_name'] = zip(*chunk['to_account'].apply(split_account))

                        # Optional: normalize numbers only (strip letters etc.)
                        def normalize_number(x):
                            if not x:
                                return None
                            import re
                            match = re.match(r'(\d+)', str(x))
                            return match.group(1) if match else None

                        chunk['from_account'] = chunk['from_account'].apply(normalize_number)
                        chunk['to_account'] = chunk['to_account'].apply(normalize_number)

                        # required columns
                        required_columns = ["deposit_type", "merchant_code", "rebalance",
                                            "original_amount", "txnid", "to_account"]
                        missing = [c for c in required_columns if c not in chunk.columns]
                        if missing:
                            flash(f"Missing columns: {', '.join(missing)}", "error")
                            return redirect(url_for("deposit"))

                        # remove duplicates
                        existing_txnids = set(
                            r[0] for r in session.query(Deposit.txnid)
                            .filter(Deposit.txnid.in_(chunk["txnid"].tolist())).all()
                        )
                        new_rows = chunk[~chunk["txnid"].isin(existing_txnids)]
                        skipped += len(chunk) - len(new_rows)
                        if new_rows.empty:
                            continue

                        # Prepare records for bulk insert
                        records = []
                        for r in new_rows.to_dict(orient="records"):
                            records.append({
                                "deposit_type": r.get("deposit_type"),
                                "merchant_code": r.get("merchant_code"),
                                "customer": r.get("customer"),
                                "txnid": r.get("txnid"),
                                "currency": r.get("currency"),
                                "bank": r.get("bank"),
                                "from_account": r.get("from_account"),
                                "to_account": r.get("to_account"),
                                "agent_number": r.get("agent_number"),
                                "shop_name": r.get("shop_name"),
                                "amount": parse_float(r.get("amount")),
                                "original_amount": parse_float(r.get("original_amount")),
                                "rebalance": parse_float(r.get("rebalance")),
                                "fee": parse_float(r.get("fee")),
                                "status": r.get("status"),
                                "audit": r.get("audit"),
                                "note_message": r.get("note_message"),
                                "refcode": r.get("refcode"),
                                "approve_by": r.get("approve_by"),
                                "matched_by": r.get("matched_by"),
                                "confirm_by": r.get("confirm_by"),
                                "created_time": parse_excel_datetime(r.get("created_time")) or datetime.utcnow(),
                                "updated_time": parse_excel_datetime(r.get("updated_time")),
                                "transfer_time": parse_excel_datetime(r.get("transfer_time")),
                                "imported_at": datetime.now(timezone.utc),
                                "deleted": False
                            })

                        session.bulk_insert_mappings(Deposit, records)
                        session.commit()
                        inserted += len(records)

                    os.remove(tmp_path)
                    flash(f"✅ Uploaded! Inserted: {inserted}, Skipped (duplicates): {skipped}", "success")
                    return redirect(url_for("deposit", page=1, per_page=20))

                except Exception as e:
                    session.rollback()
                    flash(f"Upload failed: {str(e)}", "error")
                    return redirect(url_for("deposit"))

        # --- GET filters + pagination ---
        query = session.query(Deposit).filter(Deposit.deleted == False)

        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")
        merchant_code_filter = request.args.get("merchant_code")
        bank_filter = request.args.get("bank")
        status_filter = request.args.get("status")

        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_date = (datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc)
            query = query.filter(Deposit.created_time >= start_date, Deposit.created_time < end_date)
        else:
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = today + timedelta(days=1)
            query = query.filter(Deposit.created_time >= today, Deposit.created_time < tomorrow)

        if merchant_code_filter:
            query = query.filter(Deposit.merchant_code == merchant_code_filter)
        if bank_filter:
            query = query.filter(Deposit.bank == bank_filter)
        if status_filter:
            query = query.filter(Deposit.status == status_filter)

        total_count = query.count()
        total_pages = (total_count + per_page - 1) // per_page

        deposits = query.order_by(Deposit.created_time.desc()) \
                        .offset((page - 1) * per_page) \
                        .limit(per_page) \
                        .all()

        merchant_codes = [r[0] for r in session.query(distinct(Deposit.merchant_code)).all()]
        banks = [r[0] for r in session.query(distinct(Deposit.bank)).all()]
        statuses = [r[0] for r in session.query(distinct(Deposit.status)).all()]

        table_html = render_template("deposit_table.html", deposits=deposits)

        return render_template(
            "deposit.html",
            deposits=deposits,
            table_html=table_html,
            merchant_codes=sorted([m or "" for m in merchant_codes]),
            banks=sorted([b or "" for b in banks]),
            statuses=sorted([s or "" for s in statuses]),
            selected_merchant=merchant_code_filter,
            selected_bank=bank_filter,
            selected_status=status_filter,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            settings=setting
        )




@app.route("/deposits/cleanup", methods=["POST"])
def cleanup_deposits():
    try:
        with SessionLocal() as session:  # <-- create session here
            session.query(Deposit).filter(Deposit.status != "Successful").delete(synchronize_session=False)
            session.commit()
            flash("All non-successful deposits have been removed.", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")

    return redirect(url_for("deposit"))


@app.route('/deleted_deposits')
@with_session
def deleted_deposits(session):
    deleted_rows = session.query(Deposit).filter(Deposit.deleted == True).all()
    return render_template('deleted.html', deleted_rows=deleted_rows)


@app.route('/deposit/delete/<int:deposit_id>', methods=['POST'])
@with_session
def delete_deposit(session, deposit_id):
    deposit = session.get(Deposit, deposit_id)
    if deposit:
        deposit.deleted = True
        session.commit()
    return redirect(request.referrer or url_for('deposit'))


@app.route('/deposit/bin')
@with_session
def deposit_bin(session):
    deleted_rows = session.query(Deposit).filter(Deposit.deleted == True).all()
    return render_template('deposit_bin.html', deleted_rows=deleted_rows)


@app.route('/deposit/restore/<int:deposit_id>', methods=['POST'])
@with_session
def restore_deposit(session, deposit_id):
    deposit = session.get(Deposit, deposit_id)
    if deposit:
        deposit.deleted = False
        session.commit()
    return redirect(url_for('deposit_bin'))


@app.route('/deposit/permanent_delete/<int:deposit_id>', methods=['POST'])
@with_session
def permanent_delete(session, deposit_id):
    deposit = session.get(Deposit, deposit_id)
    if deposit:
        session.delete(deposit)
        session.commit()
    return redirect(url_for('deposit_bin'))


@app.route("/deposit/duplicates")
def deposit_duplicates():
    with SessionLocal() as session:
        duplicates = (
            session.query(Deposit.txnid, func.count(Deposit.id).label("count"))
            .group_by(Deposit.txnid)
            .having(func.count(Deposit.id) > 1)
            .all()
        )

        duplicate_rows = []
        for txnid, _ in duplicates:
            rows = session.query(Deposit).filter(Deposit.txnid == txnid).all()
            duplicate_rows.extend(rows)

    return render_template("deposit_duplicates.html", duplicate_rows=duplicate_rows)





@app.route("/withdrawal", methods=["GET", "POST"])
def withdrawal():
    with SessionLocal() as session:
        setting = session.query(Setting).first()

        # Pagination
        page = request.args.get("page", default=1, type=int)
        per_page = request.args.get("per_page", default=20, type=int)

        if request.method == "POST":
            file = request.files.get("excel_file")
            if file:
                try:
                    # Save file temporarily
                    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
                    os.close(fd)

                    # Convert Excel → CSV if needed
                    if file.filename.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(file, engine="openpyxl")
                        df.to_csv(tmp_path, index=False)
                    else:
                        file.save(tmp_path)

                    chunksize = 10_000
                    inserted = 0
                    skipped = 0

                    # Split account string into agent_number and shop_name
                    def split_account(account_str):
                        if not account_str:
                            return None, None
                        parts = str(account_str).split(" - ", 1)
                        if len(parts) == 2:
                            return parts[0].strip(), parts[1].strip()
                        return parts[0].strip(), None

                    # Normalize numeric account (remove letters, etc.)
                    def normalize_account(x):
                        if not x:
                            return None
                        import re
                        match = re.match(r"(\d+)", str(x))
                        return match.group(1) if match else None

                    for chunk in pd.read_csv(tmp_path, chunksize=chunksize, dtype=str):
                        chunk.columns = chunk.columns.str.strip().str.lower().str.replace(" ", "_")
                        chunk = chunk.where(pd.notnull(chunk), None)

                        # Required columns
                        required_columns = ["merchant", "spid", "transaction_note", "approve_batches_by", "r.bal"]
                        missing = [c for c in required_columns if c not in chunk.columns]
                        if missing:
                            flash(f"Missing required columns: {', '.join(missing)}", "error")
                            return redirect(url_for("withdrawal"))

                        # Normalize txnid
                        chunk["txnid"] = chunk["txnid"].astype(str).str.strip()
                        chunk = chunk[chunk["txnid"] != ""]
                        if chunk.empty:
                            continue

                        # Parse numeric fields
                        for col in ["amount", "fee", "r.bal"]:
                            if col in chunk.columns:
                                chunk[col] = pd.to_numeric(
                                    chunk[col].astype(str).str.replace(",", "").str.strip(),
                                    errors="coerce"
                                ).fillna(0.0)

                        # Normalize account columns
                        chunk['from_account'] = chunk['from_account'].apply(lambda x: x.strip() if x else "")
                        chunk['to_account'] = chunk['to_account'].apply(lambda x: x.strip() if x else "")

                        # Check duplicates
                        existing_txnids = set(
                            r[0] for r in session.query(Withdrawal.txnid)
                            .filter(Withdrawal.txnid.in_(chunk["txnid"].tolist())).all()
                        )
                        new_rows = chunk[~chunk["txnid"].isin(existing_txnids)]
                        skipped += len(chunk) - len(new_rows)
                        if new_rows.empty:
                            continue

                        # Prepare Withdrawal objects
                        records = []
                        for _, row in new_rows.iterrows():
                            from_account_str = str(row.get("from_account") or "")
                            agent_number, shop_name = split_account(from_account_str)

                            records.append({
                                "merchant_code": str(row.get("merchant") or ""),
                                "spid": str(row.get("spid") or ""),
                                "customer": str(row.get("customer") or ""),
                                "txnid": str(row.get("txnid") or ""),
                                "currency": str(row.get("currency") or ""),
                                "bank": str(row.get("bank") or ""),
                                "from_account": from_account_str,
                                "to_account": str(row.get("to_account") or ""),
                                "agent_number": agent_number,
                                "shop_name": shop_name,
                                "amount": parse_float(row.get("amount")),
                                "fee": parse_float(row.get("fee")),
                                "r_bal": parse_float(row.get("r.bal")),
                                "ref_code": str(row.get("refcode") or ""),
                                "created_time": parse_excel_datetime(row.get("created_time")) or datetime.utcnow(),
                                "updated_time": parse_excel_datetime(row.get("updated_time")),
                                "transfered_time": parse_excel_datetime(row.get("transfered_time")),
                                "status": str(row.get("status") or ""),
                                "audit": str(row.get("audit") or ""),
                                "note_message": str(row.get("note_message") or ""),
                                "approve_batches_by": str(row.get("approve_batches_by") or ""),
                                "transaction_note": str(row.get("transaction_note") or ""),
                                "approve_by": str(row.get("approve_by") or ""),
                                "matched_by": str(row.get("matched_by") or ""),
                                "confirm_by": str(row.get("confirm_by") or ""),
                                "imported_at": datetime.utcnow()
                            })

                        session.bulk_insert_mappings(Withdrawal, records)
                        session.commit()
                        inserted += len(records)

                    os.remove(tmp_path)
                    flash(f"✅ Uploaded! Inserted: {inserted}, Skipped duplicates: {skipped}", "success")
                    return redirect(url_for("withdrawal", page=1, per_page=20))

                except Exception as e:
                    flash(f"Upload failed: {str(e)}", "error")
                    return redirect(url_for("withdrawal"))

        # --- GET filters & pagination ---
        query = session.query(Withdrawal)
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")
        merchant_code = request.args.get("merchant_code")
        bank = request.args.get("bank")
        status = request.args.get("status")

        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(Withdrawal.created_time >= start_date,
                                 Withdrawal.created_time < end_date)
        else:
            today = datetime.now(timezone.utc).date()
            tomorrow = today + timedelta(days=1)
            query = query.filter(Withdrawal.created_time >= today,
                                 Withdrawal.created_time < tomorrow)

        if merchant_code:
            query = query.filter(Withdrawal.merchant_code == merchant_code)
        if bank:
            query = query.filter(Withdrawal.bank == bank)
        if status:
            query = query.filter(Withdrawal.status == status)

        total_count = query.count()
        total_pages = (total_count + per_page - 1) // per_page

        withdrawals = query.order_by(Withdrawal.created_time.desc()) \
                           .offset((page - 1) * per_page) \
                           .limit(per_page) \
                           .all()

        merchant_codes = [m[0] for m in session.query(Withdrawal.merchant_code).distinct()]
        banks = [b[0] for b in session.query(Withdrawal.bank).distinct()]
        statuses = [s[0] for s in session.query(Withdrawal.status).distinct()]

        table_html = render_template("partials/withdrawal_table.html", withdrawals=withdrawals)

        return render_template(
            "withdrawal.html",
            table_html=table_html,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            merchant_codes=merchant_codes,
            banks=banks,
            statuses=statuses,
            selected_merchant=merchant_code,
            selected_bank=bank,
            selected_status=status,
            active_page="withdrawal",
            settings=setting
        )







@app.route("/withdrawals/cleanup", methods=["POST"])
def cleanup_withdrawals():
    try:
        with SessionLocal() as session:
            session.query(Withdrawal).filter(Withdrawal.status != "Successful").delete(synchronize_session=False)
            session.commit()
            flash("All non-successful withdrawals have been removed.", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")

    return redirect(url_for("withdrawal"))



@app.route("/accounts")
def accounts():
    with SessionLocal() as session:   # ✅ Open session
        deposit_stats = session.query(
            Deposit.to_account.label("agent"),
            func.count(Deposit.id).label("deposit_count"),
            func.sum(Deposit.amount).label("total_deposit")
        ).filter(Deposit.status == "Successful") \
         .group_by(Deposit.to_account) \
         .order_by(func.sum(Deposit.amount).desc()) \
         .limit(10).all()

        withdrawal_stats = session.query(
            Withdrawal.from_account.label("agent"),
            func.count(Withdrawal.id).label("withdrawal_count"),
            func.sum(Withdrawal.amount).label("total_withdrawal")
        ).filter(Withdrawal.status == "Successful") \
         .group_by(Withdrawal.from_account) \
         .order_by(func.sum(Withdrawal.amount).desc()) \
         .limit(10).all()

    return render_template(
        "accounts/accounts.html",
        deposit_stats=deposit_stats,
        withdrawal_stats=withdrawal_stats
    )



@app.route("/monitoring")
def monitoring():
    def normalize_key(value: str) -> str:
        """Normalize shop/account names to a consistent key."""
        if not value:
            return ""
        key = str(value).replace("\xa0", " ").strip()
        parts = key.split(" - ")
        return parts[-1].strip().upper()

    with SessionLocal() as session:   # ✅ Open session

        # --- Only successful deposits ---
        deposits = (
            session.query(
                Deposit.to_account.label("agent"),
                func.coalesce(func.sum(Deposit.amount), 0.0).label("total_deposit")
            )
            .filter(Deposit.status == "Successful")
            .group_by(Deposit.to_account)
            .all()
        )
        deposit_map = {normalize_key(d.agent): float(d.total_deposit) for d in deposits}

        # --- Only successful withdrawals ---
        withdrawals = (
            session.query(
                Withdrawal.from_account.label("agent"),
                func.coalesce(func.sum(Withdrawal.amount), 0.0).label("total_withdrawal")
            )
            .filter(Withdrawal.status == "Successful")
            .group_by(Withdrawal.from_account)
            .all()
        )
        withdrawal_map = {normalize_key(w.agent): float(w.total_withdrawal) for w in withdrawals}

        # --- Prepare agents_data dict ---
        agents_data = {}

        # Collect all unique agent keys
        all_agents = set(list(deposit_map.keys()) + list(withdrawal_map.keys()))

        # Initialize agents_data
        for agent_key in all_agents:
            agents_data[agent_key] = {
                "deposit": deposit_map.get(agent_key, 0.0),
                "withdrawal": withdrawal_map.get(agent_key, 0.0),
                "adjustment": 0.0,
                "settlement": 0.0,
                "group_code": "",
            }

        # --- Lookup group code from Limit table ---
        limits = (
            session.query(Limit.account, Limit.group)
            .all()
        )
        for limit in limits:
            key = normalize_key(limit.account)
            if key in agents_data:
                agents_data[key]["group_code"] = limit.group

        # --- Compute balances ---
        for agent, data in agents_data.items():
            data["balance"] = (
                data["deposit"]
                - data["withdrawal"]
                + data["adjustment"]
                - data["settlement"]
            )

    return render_template("accounts/monitoring.html", show_accounts_header=True, agents_data=agents_data)


@app.route("/accounts/overview")
def overview():
    today = date.today()
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())
    with SessionLocal() as session:
        settlements = (
            session.query(Settlement)
            .filter(Settlement.date >= start_of_day, Settlement.date <= end_of_day)
            .all()
        )
        # --- Fetch agents ---
        agents = session.query(Agent).all()
        opening_balances = {a.id: a.opening_balance or 0 for a in agents}

        # --- Fetch SDP for each shop ---
        sdp_map = {s.shop: s.sdp or 0 for s in session.query(Sdp).all()}

        # --- Aggregate deposits ---
        deposit_totals = (
            session.query(
                Deposit.to_account,
                func.coalesce(func.sum(Deposit.amount), 0.0).label("total")
            )
            .filter(
                Deposit.status == "Successful",
                Deposit.created_time >= start_of_day,
                Deposit.created_time <= end_of_day
            )
            .group_by(Deposit.to_account)
            .all()
        )
        deposit_map = {row.to_account: float(row.total) for row in deposit_totals if row.to_account}

        # --- Aggregate withdrawals ---
        withdrawal_totals = (
            session.query(
                Withdrawal.from_account,
                func.coalesce(func.sum(Withdrawal.amount), 0.0).label("total")
            )
            .filter(
                Withdrawal.status == "Successful",
                Withdrawal.created_time >= start_of_day,
                Withdrawal.created_time <= end_of_day
            )
            .group_by(Withdrawal.from_account)
            .all()
        )
        withdrawal_map = {row.from_account: float(row.total) for row in withdrawal_totals if row.from_account}

        # --- Limits / Group Code / Status ---
        limits = session.query(Limit).all()
        limit_map = {l.shop_name: l.group for l in limits}
        status_map = {l.shop_name: l.status for l in limits}

        # --- Build agents overview ---
        agents_overview = []
        for agent in agents:
            shop = agent.shop
            opening_balance = opening_balances.get(agent.id, 0.0)
            deposit = deposit_map.get(shop, 0.0)
            withdrawal = withdrawal_map.get(shop, 0.0)
            balance = opening_balance + deposit - withdrawal
            sdp_val = sdp_map.get(shop, 0.0)
            group_code = limit_map.get(shop, "")

            # Detect DP or WD in group code (case-insensitive)
            has_dp = "DP" in group_code.upper()
            has_wd = "WD" in group_code.upper()

            agents_overview.append({
                "shop": shop,
                "opening_balance": opening_balance,
                "deposit": deposit,
                "withdrawal": withdrawal,
                "balance": balance,
                "sdp": sdp_val,
                "group_code": group_code,
                "status": status_map.get(shop, "Locked"),
                "has_dp": has_dp,
                "has_wd": has_wd
            })

        # --- Apply filters ---
        filter1 = [a for a in agents_overview if a["balance"] > a["sdp"] and a["has_dp"]]
        filter2 = [a for a in agents_overview if a["balance"] < 30000 and not a["has_dp"]]
        filter3 = [a for a in agents_overview if a["balance"] <= 10000 and a["has_wd"]]

        # --- All unique group codes for exclusion filter ---
        all_group_codes = [l[0] for l in session.query(distinct(Limit.group)).all() if l[0]]

    return render_template(
        "accounts/overview.html", show_accounts_header=True,
        agents_overview=agents_overview,
        settlements=settlements,
        filter1=filter1,
        filter2=filter2,
        filter3=filter3,
        all_group_codes=all_group_codes
    )




@app.route("/accounts/wallet_limit", methods=["GET"])
def wallet_limit():
    with SessionLocal() as session:
        # --- Wallet stats ---
        total_wallets = session.query(Wallet).count()
        wallet_by_status = session.query(Wallet.status, func.count(Wallet.id)).group_by(Wallet.status).all()
        wallet_by_group = session.query(Wallet.group_code, func.count(Wallet.id)).group_by(Wallet.group_code).all()
        wallet_by_bank = session.query(Wallet.bank_code, func.count(Wallet.id)).group_by(Wallet.bank_code).all()

        # --- Limit stats ---
        total_limits = session.query(Limit).count()
        limit_by_status = session.query(Limit.status, func.count(Limit.id)).group_by(Limit.status).all()
        limit_by_group = session.query(Limit.group, func.count(Limit.id)).group_by(Limit.group).all()
        limit_by_bank = session.query(Limit.bank, func.count(Limit.id)).group_by(Limit.bank).all()

        return render_template(
            "accounts/wallet_limit.html", show_accounts_header=True,

            total_wallets=total_wallets,
            wallet_by_status=wallet_by_status,
            wallet_by_group=wallet_by_group,
            wallet_by_bank=wallet_by_bank,
            total_limits=total_limits,
            limit_by_status=limit_by_status,
            limit_by_group=limit_by_group,
            limit_by_bank=limit_by_bank,
        )

# --- Wallet Route ---
@app.route("/accounts/wallet", methods=["GET", "POST"])
def wallet():
    with SessionLocal() as session:
        if request.method == "POST":
            excel_file = request.files.get("excel_file")

            if not excel_file:
                flash("No file selected!", "danger")
                return redirect(url_for("wallet"))

            df = pd.read_excel(excel_file, engine="openpyxl")

            # Clear old wallet data
            session.query(Wallet).delete()
            session.commit()

            for _, row in df.iterrows():
                wallet = Wallet(
                    currency_name=str(row.get("Currency Name") or ""),
                    deposit_channels=str(row.get("Deposit Channels") or ""),
                    merchant_code=str(row.get("Merchant Code") or ""),
                    group_code=str(row.get("Group Code") or ""),
                    bank_code=str(row.get("Bank Code") or ""),
                    account_name=str(row.get("Account Name") or ""),
                    short_name=str(row.get("Short Name") or ""),
                    account_number=str(row.get("Account Number") or ""),
                    phone=str(row.get("Phone") or ""),
                    status=str(row.get("Status") or ""),
                    timestamp=datetime.now()
                )
                session.add(wallet)
            session.commit()

            flash("⚠️ Wallet data uploaded successfully! Old data cleared.", "success")
            return redirect(url_for("wallet"))

        # --- GET: Show Wallet Data ---
        wallet_data = session.query(Wallet).all()
        return render_template("accounts/wallet.html", show_accounts_header=True, wallet_data=wallet_data)


# --- Limit Route ---
@app.route("/accounts/limit", methods=["GET", "POST"])
def limit():
    with SessionLocal() as session:
        if request.method == "POST":
            excel_file = request.files.get("excel_file")

            if not excel_file:
                flash("No file selected!", "danger")
                return redirect(url_for("limit"))

            df = pd.read_excel(excel_file, engine="openpyxl")

            # Clear old limit data
            session.query(Limit).delete()
            session.commit()

            for _, row in df.iterrows():
                account_str = str(row.get("Account") or "")
                agent_number = shop_name = None
                if " - " in account_str:
                    agent_number, shop_name = account_str.split(" - ", 1)
                    agent_number = agent_number.strip()
                    shop_name = shop_name.strip()
                else:
                    shop_name = account_str.strip()

                limit = Limit(
                    bank=str(row.get("Bank") or ""),
                    channel=str(row.get("Channel") or ""),
                    group=str(row.get("Group") or ""),
                    account=account_str,
                    agent_number=agent_number,  # ✅ new
                    shop_name=shop_name,        # ✅ new
                    balance=str(row.get("Balance") or ""),
                    balance_limit=str(row.get("Balance Limit") or ""),
                    dp_limit=str(row.get("DP Limit") or ""),
                    total_dp=str(row.get("Total DP") or ""),
                    wd_limit=str(row.get("WD Limit") or ""),
                    total_wd=str(row.get("Total WD") or ""),
                    update_time=row.get("Update Time") if row.get("Update Time") else None,
                    login=str(row.get("Login") or ""),
                    status=str(row.get("Status") or ""),
                    timestamp=datetime.now()
                )
                session.add(limit)
            session.commit()

            flash("⚠️ Limit data uploaded successfully! Old data cleared.", "success")
            return redirect(url_for("limit"))

        # --- GET: Show Limit Data ---
        limit_data = session.query(Limit).all()
        return render_template("accounts/limit.html", show_accounts_header=True, limit_data=limit_data)




@app.route("/accounts/upload_wallet", methods=["POST"])
def upload_wallet():
    file = request.files.get("wallet_file")
    if not file:
        flash("No file selected!", "danger")
        return redirect(url_for("wallet_limit"))

    filename = file.filename.lower()

    try:
        # Read Excel or CSV
        if filename.endswith(".xlsx"):
            df = pd.read_excel(file, engine="openpyxl")
        elif filename.endswith(".xls"):
            # Ensure xlrd==2.0.1 is installed
            df = pd.read_excel(file, engine="xlrd")
        elif filename.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            flash("Unsupported file format. Upload .xlsx, .xls, or .csv", "danger")
            return redirect(url_for("wallet_limit"))
    except Exception as e:
        flash(f"Error reading file: {e}", "danger")
        return redirect(url_for("wallet_limit"))

    if df.empty:
        flash("The uploaded file is empty.", "warning")
        return redirect(url_for("wallet_limit"))

    try:
        with SessionLocal() as session:
            # Clear existing data
            session.query(Wallet).delete()
            session.commit()

            # Insert new data
            for _, row in df.iterrows():
                wallet = Wallet(
                    id=Column(Integer, primary_key=True, autoincrement=True),

                    currency_name=str(row.get("Currency Name") or ""),
                    deposit_channels=str(row.get("Deposit Channels") or ""),
                    merchant_code=str(row.get("Merchant Code") or ""),
                    group_code=str(row.get("Group Code") or ""),
                    bank_code=str(row.get("Bank Code") or ""),
                    account_name=str(row.get("Account Name") or ""),
                    short_name=str(row.get("Short Name") or ""),
                    account_number=str(row.get("Account Number") or ""),
                    phone=str(row.get("Phone") or ""),
                    status=str(row.get("Status") or ""),
                    timestamp=datetime.now()
                )
                session.add(wallet)
            session.commit()

    except Exception as e:
        flash(f"Database error: {e}", "danger")
        return redirect(url_for("wallet_limit"))

    flash("⚠️ Wallet data uploaded successfully! Old data cleared.", "success")
    return redirect(url_for("wallet_limit", show_accounts_header=True, active_tab="wallet-tab"))


@app.route("/accounts/upload_limit", methods=["POST"])
def upload_limit():
    file = request.files.get("limit_file")
    if not file:
        flash("No file selected!", "danger")
        return redirect(url_for("wallet_limit"))

    filename = file.filename.lower()
    if filename.endswith(".xlsx"):
        df = pd.read_excel(file, engine="openpyxl")
    elif filename.endswith(".xls"):
        df = pd.read_excel(file, engine="xlrd")
    elif filename.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        flash("Unsupported file format. Upload .xlsx, .xls, or .csv", "danger")
        return redirect(url_for("wallet_limit"))

    with SessionLocal() as session:
        session.query(Limit).delete()
        session.commit()

        for _, row in df.iterrows():
            limit = Limit(
                bank=str(row.get("Bank") or ""),
                channel=str(row.get("Channel") or ""),
                group=str(row.get("Group") or ""),
                account=str(row.get("Account") or ""),
                balance=safe_float(row.get("Balance")),
                balance_limit=safe_float(row.get("Balance Limit")),
                dp_limit=safe_float(row.get("DP Limit")),
                total_dp=safe_float(row.get("Total DP")),
                wd_limit=safe_float(row.get("WD Limit")),
                total_wd=safe_float(row.get("Total WD")),
                update_time=parse_datetime(row.get("Update Time")),  # ✅ FIXED
                login=str(row.get("Login") or ""),
                status=str(row.get("Status") or ""),
                timestamp=datetime.now()
            )

            session.add(limit)

        session.commit()
        flash("⚠️ Limit data uploaded successfully!", "success")
    return redirect(url_for("wallet_limit", show_accounts_header=True, active_tab="limit-tab"))



@app.route("/accounts/inactive")
def inactive():
    return render_template(
        "accounts/inactive.html",
        show_accounts_header=True,
        active_page="inactive"
    )





# Hardcode headers so they always match HTML
HEADERS = [
    "SDP", "ACCOUNTS", "OPENING", "DP", "WD", "ADJUSTMENT", "BALANCE",
    "GROUP CODE", "STATUS", "AVAILABLE LIMIT", "SYSTEM BALANCE",
    "TG LINKS", "CHAT ID", "WALLET", "REMARKS"
]

@app.route("/gsheet")
def gsheet_view():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    SERVICE_ACCOUNT_FILE = "service_account.json"
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
    client = gspread.authorize(creds)

    ws = client.open("SSG AGENT BALANCE MONITORING 2025").worksheet("MANAGING1")

    # --- fetch raw values (all cells) ---
    data = ws.get_all_values()

    # --- pad each row so it always matches HEADERS length ---
    rows = []
    for row in data[2:]:  # skip first 2 rows if those are metadata
        padded_row = row + [""] * (len(HEADERS) - len(row))
        row_dict = dict(zip(HEADERS, padded_row))
        rows.append(row_dict)

    return render_template("accounts/gsheet.html", show_accounts_header=True, rows=rows, headers=HEADERS)



@app.route("/daily_balance")
def daily_balance():
    today = date.today()
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())

    # --- Pagination ---
    page = int(request.args.get("page", 1))
    per_page = 100

    # --- Sorting ---
    sort_column = request.args.get("sort", "shop_name")
    sort_order = request.args.get("order", "asc")

    # --- Search filters ---
    shop_search = request.args.get("shop_search", "").strip().upper()
    group_search = request.args.get("group_search", "").strip().upper()

    def normalize_key(value: str) -> str:
        """Normalize shop names to a consistent key."""
        if not value:
            return ""
        # Remove non-breaking spaces, trim, split by "-", take last part, uppercase
        key = str(value).replace("\xa0", " ").strip()
        parts = key.split(" - ")
        return parts[-1].strip().upper()

    with SessionLocal() as session:
        # --- Fetch agents ---
        agents = session.query(Agent).all()
        opening_balances = {agent.id: safe_float(agent.opening_balance) for agent in agents}

        # --- Aggregate deposits ---
        deposit_totals = (
            session.query(
                Deposit.shop_name.label("shop"),
                func.coalesce(func.sum(Deposit.amount), 0.0).label("total")
            )
            .filter(
                Deposit.status == "Successful",
                Deposit.created_time >= start_of_day,
                Deposit.created_time <= end_of_day
            )
            .group_by(Deposit.shop_name)
            .all()
        )
        deposit_map = {normalize_key(row.shop): safe_float(row.total) for row in deposit_totals if row.shop}

        # --- Aggregate withdrawals ---
        withdrawal_totals = (
            session.query(
                Withdrawal.shop_name.label("shop"),
                func.coalesce(func.sum(Withdrawal.amount), 0.0).label("total")
            )
            .filter(
                Withdrawal.status == "Successful",
                Withdrawal.created_time >= start_of_day,
                Withdrawal.created_time <= end_of_day
            )
            .group_by(Withdrawal.shop_name)
            .all()
        )
        withdrawal_map = {normalize_key(row.shop): safe_float(row.total) for row in withdrawal_totals if row.shop}

        # --- Limits & status ---
        limits = session.query(Limit).all()
        limit_map = {normalize_key(l.shop_name): l.group for l in limits}
        status_map = {normalize_key(l.shop_name): l.status for l in limits}

        # --- SDP (for both sdp and original group code) ---
        sdps = session.query(Sdp).all()
        sdp_map = {normalize_key(s.shop): safe_float(s.sdp) for s in sdps}
        sdp_group_map = {normalize_key(s.shop): s.group_code for s in sdps}

        # --- Map today's adjustments (TopUps) ---
        topups_today = session.query(TopUp).filter(TopUp.date >= start_of_day, TopUp.date <= end_of_day).all()
        adjustment_map = {}
        for t in topups_today:
            to_key = normalize_key(t.to_agent)
            if to_key:
                adjustment_map[to_key] = adjustment_map.get(to_key, 0.0) + safe_float(t.amount_process)
            from_key = normalize_key(t.from_agent)
            if from_key:
                adjustment_map[from_key] = adjustment_map.get(from_key, 0.0) - safe_float(t.amount_process)

        # --- Map today's settlements ---
        settlements_today = session.query(Settlement).filter(Settlement.date >= start_of_day, Settlement.date <= end_of_day).all()
        settlement_map = {}
        for s in settlements_today:
            key = normalize_key(s.agent)
            total_amount = safe_float(s.amount) + safe_float(s.fee)
            settlement_map[key] = settlement_map.get(key, 0.0) + total_amount

        # --- Build agents data ---
        agents_data = []
        for agent in agents:
            k = normalize_key(agent.shop)
            if shop_search and shop_search not in k:
                continue
            if group_search and group_search not in (limit_map.get(k, "").upper()):
                continue

            deposits = deposit_map.get(k, 0.0)
            withdrawals = withdrawal_map.get(k, 0.0)
            opening_balance = opening_balances.get(agent.id, 0.0)
            adjustment_value = adjustment_map.get(k, 0.0)
            settlement_value = settlement_map.get(k, 0.0)
            sdp_value = sdp_map.get(k, 0.0)

            balance = opening_balance + deposits - withdrawals + adjustment_value + settlement_value

            agents_data.append({
                "shop_name": agent.shop,
                "opening_balance": opening_balance,
                "deposit": deposits,
                "withdrawal": withdrawals,
                "adjustment": adjustment_value,
                "settlement": settlement_value,
                "balance": balance,
                "sdp": sdp_value,
                "group_code": limit_map.get(k, ""),
                "original_group_code": sdp_group_map.get(k, ""),
                "status": status_map.get(k, "Locked")
            })

        # --- Sorting ---
        reverse = sort_order == "desc"
        agents_data.sort(key=lambda x: x.get(sort_column, ""), reverse=reverse)

        # --- Pagination ---
        total_pages = (len(agents_data) + per_page - 1) // per_page
        start = (page - 1) * per_page
        end = start + per_page
        paginated_agents = agents_data[start:end]

        # --- Prepare pagination args ---
        pagination_args = request.args.to_dict()
        pagination_args.pop("page", None)

    return render_template(
        "accounts/daily_balance.html",
        show_accounts_header=True,
        agents_data=paginated_agents,
        today=today,
        page=page,
        total_pages=total_pages,
        pagination_args=pagination_args,
        sort_column=sort_column,
        sort_order=sort_order,
        shop_search=shop_search,
        group_search=group_search
    )




@app.route("/daily_balance/export_csv")
def daily_balance_export_csv():
    # Get query params
    sort_column = request.args.get("sort", "shop_name")
    sort_order = request.args.get("order", "asc")
    shop_search = request.args.get("shop_search", "").strip().upper()
    group_search = request.args.get("group_search", "").strip().upper()

    today = date.today()
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())

    with SessionLocal() as session:
        agents = session.query(Agent).all()

        def normalize_key(value: str) -> str:
            return str(value).split(" - ")[-1].strip().upper() if value else ""

        deposit_totals = (
            session.query(
                Deposit.to_account,
                func.coalesce(func.sum(Deposit.amount), 0.0).label("total")
            )
            .filter(
                Deposit.status == "Successful",
                Deposit.created_time >= start_of_day,
                Deposit.created_time <= end_of_day
            )
            .group_by(Deposit.to_account)
            .all()
        )
        deposit_map = {normalize_key(r.to_account): float(r.total or 0.0) for r in deposit_totals if r.to_account}

        withdrawal_totals = (
            session.query(
                Withdrawal.from_account,
                func.coalesce(func.sum(Withdrawal.amount), 0.0).label("total")
            )
            .filter(
                Withdrawal.status == "Successful",
                Withdrawal.created_time >= start_of_day,
                Withdrawal.created_time <= end_of_day
            )
            .group_by(Withdrawal.from_account)
            .all()
        )
        withdrawal_map = {normalize_key(r.from_account): float(r.total or 0.0) for r in withdrawal_totals if r.from_account}

        limits = session.query(Limit).all()
        limit_map = {normalize_key(l.shop_name): l.group for l in limits}
        status_map = {normalize_key(l.shop_name): l.status for l in limits}

        agents_data = []
        for agent in agents:
            k = normalize_key(agent.shop)
            deposits = deposit_map.get(k, 0.0)
            withdrawals = withdrawal_map.get(k, 0.0)
            opening_balance = agent.opening_balance or 0.0
            balance = opening_balance + deposits - withdrawals
            group_code = limit_map.get(k, "")
            status = status_map.get(k, "Locked")

            # Apply search filters
            if shop_search and shop_search not in k:
                continue
            if group_search and group_search not in group_code.upper():
                continue

            agents_data.append({
                "shop_name": agent.shop,
                "opening_balance": opening_balance,
                "deposit": deposits,
                "withdrawal": withdrawals,
                "balance": balance,
                "adjustment": 0.0,
                 "settlement": settlement_map.get(k, 0.0),
                "group_code": group_code,
                "status": status
            })

        # Sort
        reverse = True if sort_order == "desc" else False
        agents_data.sort(key=lambda x: x.get(sort_column, ""), reverse=reverse)

    # Generate CSV
    def generate_csv():
        header = ["Shop", "Opening Balance", "Deposit", "Withdrawal", "Adjustment",
                  "Settlement", "Closing Balance", "Group Code", "Status"]
        yield ",".join(header) + "\n"
        for row in agents_data:
            yield ",".join([
                str(row['shop_name']),
                f"{row['opening_balance']:.2f}",
                f"{row['deposit']:.2f}",
                f"{row['withdrawal']:.2f}",
                f"{row['adjustment']:.2f}",
                f"{row['settlement']:.2f}",
                f"{row['balance']:.2f}",
                str(row['group_code']),
                str(row['status'])
            ]) + "\n"

    return Response(generate_csv(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=daily_balance.csv"})


# Example in-memory store (for demo; replace with DB/Sheet)
opening_balance_store = {}  # key: shop_name, value: {'running_balance': float, 'status': str}

@app.route("/update_opening_balance", methods=["GET", "POST"])
def update_opening_balance():
    session = SessionLocal()
    try:
        if request.method == "POST":
            ob_data = request.form.get("opening_balance_data", "")
            sdp_data = request.form.get("sdp_data", "")

            # --- Save Opening Balance ---
            if ob_data:
                for row in ob_data.strip().splitlines():
                    cols = row.split("\t")
                    if len(cols) >= 3:
                        shop = cols[0].strip()
                        balance_str = cols[1].strip() or "0"

                        if balance_str in ["-", "--", "N/A", ""]:
                            balance_str = "0"
                        balance_str = balance_str.replace(",", "")
                        if balance_str.startswith("(") and balance_str.endswith(")"):
                            balance_str = "-" + balance_str[1:-1]

                        try:
                            balance = float(balance_str)
                        except ValueError:
                            balance = 0

                        status = cols[2].strip()

                        ob = session.query(Agent).filter_by(shop=shop).first()
                        if not ob:
                            ob = Agent(shop=shop)
                            session.add(ob)
                        ob.opening_balance = balance
                        ob.status = status if status else None

            # --- Save SDP (in Sdp table) ---
            if sdp_data:
                for row in sdp_data.strip().splitlines():
                    cols = row.split("\t")
                    if len(cols) >= 2:
                        shop = cols[0].strip()
                        try:
                            sdp_val = float(cols[1].strip())
                        except ValueError:
                            sdp_val = 0

                        sdp = session.query(Sdp).filter_by(shop=shop).first()
                        if not sdp:
                            sdp = Sdp(shop=shop)
                            session.add(sdp)
                        sdp.sdp = sdp_val

            session.commit()
            flash("Opening Balance and SDP updated successfully!", "success")
            return redirect(url_for("update_opening_balance"))

        # --- GET ---
        opening_balances = session.query(Agent).all()
        sdps = session.query(Sdp).all()

        # Agents without matching SDP entry
        sdp_shops = [s.shop for s in sdps]
        agents_without_sdp = session.query(Agent).filter(~Agent.shop.in_(sdp_shops)).all()

        return render_template(
            "Accounts/update_opening_balance.html",
            show_accounts_header=True,
            opening_balances=opening_balances,
            sdps=sdps,
            agents_without_sdp=agents_without_sdp
        )

    finally:
        session.close()



# --- Clear Opening Balance ---
@app.route("/clear_opening_balance", methods=["POST"])
def clear_opening_balance():
    with SessionLocal() as session:
        try:
            session.query(Agent).update(
                {Agent.opening_balance: 0, Agent.status: None},
                synchronize_session=False
            )
            session.commit()
            flash("All Opening Balance data cleared!", "success")
        except Exception as e:
            session.rollback()
            flash(f"Error clearing OB: {e}", "danger")
    return redirect(url_for("update_opening_balance"))



# --- Clear SDP ---
@app.route("/clear_sdp", methods=["POST"])
def clear_sdp():
    with SessionLocal() as session:
        try:
            session.query(Sdp).delete()
            session.commit()
            flash("All SDP data cleared!", "success")
        except Exception as e:
            session.rollback()
            flash(f"Error clearing SDP: {e}", "danger")
    return redirect(url_for("update_opening_balance"))




@app.route("/settings", methods=["GET", "POST"])
def settings():
    with SessionLocal() as session:
        setting = session.query(Setting).first()

        if request.method == "POST":
            enable_delete = "enable_delete" in request.form

            if not setting:
                # Create new row if it doesn’t exist
                setting = Setting(enable_delete=enable_delete)
                session.add(setting)
            else:
                # Update existing row
                setting.enable_delete = enable_delete

            session.commit()
            flash("Settings updated successfully!", "success")
            return redirect(url_for("settings"))

        # Make sure template always has a setting row
        if not setting:
            setting = Setting(enable_delete=False)
            session.add(setting)
            session.commit()

        return render_template("settings.html", settings=setting)




@app.route("/deposits/delete", methods=["POST"])
def delete_deposits():
    delete_type = request.form.get("delete_type")
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")

    with SessionLocal() as session:
        if delete_type == "all":
            session.query(Deposit).delete()
        elif delete_type == "filter" and start_date and end_date:
            try:
                # Convert string "YYYY-MM-DD" to datetime
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                # End date should include the whole day (23:59:59)
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                end_dt = end_dt.replace(hour=23, minute=59, second=59)

                session.query(Deposit).filter(
                    Deposit.created_time.between(start_dt, end_dt)
                ).delete(synchronize_session=False)
            except Exception as e:
                flash(f"Error deleting by date: {e}", "danger")

        session.commit()

    flash("Deposits deleted successfully.", "success")
    return redirect(url_for("deposit"))


@app.route("/update_settings", methods=["POST"])
def update_settings():
    enable_delete = request.form.get("enable_delete") == "on"

    with SessionLocal() as session:
        setting = session.query(Setting).first()

        if not setting:
            # create default row if not exist
            setting = Setting(enable_delete=enable_delete)
            session.add(setting)
        else:
            setting.enable_delete = enable_delete

        session.commit()

    flash("Settings updated!", "success")
    return redirect(url_for("settings"))

@app.route("/withdrawals/delete", methods=["POST"])
def delete_withdrawals():
    delete_type = request.form.get("delete_type")
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")

    with SessionLocal() as session:
        if delete_type == "all":
            session.query(Withdrawal).delete()
            session.commit()
            flash("All withdrawals deleted successfully.", "success")

        elif delete_type == "filter":
            if start_date and end_date:
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

                    session.query(Withdrawal).filter(
                        Withdrawal.created_time >= start_dt,
                        Withdrawal.created_time < end_dt
                    ).delete(synchronize_session=False)

                    session.commit()
                    flash(f"Withdrawals from {start_date} to {end_date} deleted.", "success")
                except ValueError:
                    flash("Invalid date format.", "danger")
            else:
                flash("Start and end dates required for filter delete.", "warning")

    return redirect(url_for("withdrawal"))


@app.route("/adjustments")
def adjustments():
    with SessionLocal() as session:
        # Adjustments & Settlements
        adjustments_data = session.query(Adjustment).all()

        # --- Topups filters ---
        date_from = request.args.get("date_from")
        date_to = request.args.get("date_to")
        brand = request.args.get("brand")
        wallet = request.args.get("wallet")
        from_agent = request.args.get("from_agent")
        to_agent = request.args.get("to_agent")
        tx_type = request.args.get("type")
        page = int(request.args.get("page", 1))
        per_page = 20

        query = session.query(TopUp)
        if date_from:
            query = query.filter(TopUp.date >= date_from)
        if date_to:
            query = query.filter(TopUp.date <= date_to)
        if brand:
            query = query.filter(TopUp.brand == brand)
        if wallet:
            query = query.filter(TopUp.wallet == wallet)
        if from_agent:
            query = query.filter(TopUp.from_agent == from_agent)
        if to_agent:
            query = query.filter(TopUp.to_agent == to_agent)
        if tx_type:
            query = query.filter(TopUp.type == tx_type)

        total = query.count()
        topups_data = (
            query.order_by(TopUp.date.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )

        # --- Settlements filters ---
        s_date_from = request.args.get("s_date_from")
        s_date_to = request.args.get("s_date_to")
        s_agent = request.args.get("s_agent")
        s_brand = request.args.get("s_brand")
        s_wallet = request.args.get("s_wallet")
        s_purpose = request.args.get("s_purpose")
        s_page = int(request.args.get("s_page", 1))
        s_per_page = 20

        s_query = session.query(Settlement)
        if s_date_from:
            s_query = s_query.filter(Settlement.date >= s_date_from)
        if s_date_to:
            s_query = s_query.filter(Settlement.date <= s_date_to)
        if s_agent:
            s_query = s_query.filter(Settlement.agent == s_agent)
        if s_brand:
            s_query = s_query.filter(Settlement.brand == s_brand)
        if s_wallet:
            s_query = s_query.filter(Settlement.wallet == s_wallet)
        if s_purpose:
            s_query = s_query.filter(Settlement.purpose == s_purpose)

        s_total = s_query.count()
        settlements_data = (
            s_query.order_by(Settlement.date.desc())
            .offset((s_page - 1) * s_per_page)
            .limit(s_per_page)
            .all()
        )

        # --- Dropdowns ---
        brands = [r[0] for r in session.query(TopUp.brand).distinct() if r[0]]
        wallets = [r[0] for r in session.query(TopUp.wallet).distinct() if r[0]]
        from_agents = [r[0] for r in session.query(TopUp.from_agent).distinct() if r[0]]
        to_agents = [r[0] for r in session.query(TopUp.to_agent).distinct() if r[0]]
        types = [r[0] for r in session.query(TopUp.type).distinct() if r[0]]

        s_agents = [r[0] for r in session.query(Settlement.agent).distinct() if r[0]]
        s_brands = [r[0] for r in session.query(Settlement.brand).distinct() if r[0]]
        s_wallets = [r[0] for r in session.query(Settlement.wallet).distinct() if r[0]]
        s_purposes = [r[0] for r in session.query(Settlement.purpose).distinct() if r[0]]

    settlement_dashboard = get_settlement_dashboard(session)
    topup_dashboard = get_topup_dashboard(session)

    return render_template(
        "adjustments/adjustments.html",
        adjustments_data=adjustments_data,
        settlements_data=settlements_data,
        topups_data=topups_data,

        # Dashboards
        settlement_dashboard=settlement_dashboard,
        topup_dashboard=topup_dashboard,
        show_dashboard=True,


        # TopUp dropdowns
        brands=brands, wallets=wallets,
        from_agents=from_agents, to_agents=to_agents, types=types,

        # Settlement dropdowns
        s_agents=s_agents, s_brands=s_brands, s_wallets=s_wallets, s_purposes=s_purposes,

        # Filters
        filters={
            "date_from": date_from or "",
            "date_to": date_to or "",
            "brand": brand or "",
            "wallet": wallet or "",
            "from_agent": from_agent or "",
            "to_agent": to_agent or "",
            "type": tx_type or "",
        },
        s_filters={
            "date_from": s_date_from or "",
            "date_to": s_date_to or "",
            "agent": s_agent or "",
            "brand": s_brand or "",
            "wallet": s_wallet or "",
            "purpose": s_purpose or "",
        },

        # Pagination
        pagination={
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": (total + per_page - 1) // per_page
        },
        s_pagination={
            "page": s_page,
            "per_page": s_per_page,
            "total": s_total,
            "pages": (s_total + s_per_page - 1) // s_per_page
        }
    )

@app.route("/settlements")
def settlements():
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")
    agent = request.args.get("agent")
    brand = request.args.get("brand")
    wallet = request.args.get("wallet")

    with SessionLocal() as session:
        query = session.query(Settlement)

        if date_from:
            query = query.filter(Settlement.date >= date_from)
        if date_to:
            query = query.filter(Settlement.date <= date_to)
        if agent:
            query = query.filter(Settlement.agent == agent)
        if brand:
            query = query.filter(Settlement.brand == brand)
        if wallet:
            query = query.filter(Settlement.wallet == wallet)

        settlements_data = query.order_by(Settlement.date.desc()).all()

        # Fetch distinct values for filters
        agents = [a[0] for a in session.query(Settlement.agent).distinct().all() if a[0]]
        brands = [b[0] for b in session.query(Settlement.brand).distinct().all() if b[0]]
        wallets = [w[0] for w in session.query(Settlement.wallet).distinct().all() if w[0]]

    return render_template(
        "adjustments/settlements.html",
        settlements_data=settlements_data,
        filters={
            "date_from": date_from or "",
            "date_to": date_to or "",
            "agent": agent or "",
            "brand": brand or "",
            "wallet": wallet or "",
        },
        agents=agents,
        brands=brands,
        wallets=wallets
    )

@app.route("/settlement/save", methods=["POST"])
def save_settlements():
    data = request.json.get("data", [])
    with SessionLocal() as session:
        try:
            for row in data:
                # --- Parse date as YYYY-MM-DD only ---
                date_val = None
                if row.get("date"):
                    try:
                        date_val = datetime.strptime(row["date"], "%Y-%m-%d")
                    except ValueError:
                        try:
                            # fallback if time included
                            date_val = datetime.strptime(row["date"], "%Y-%m-%d %H:%M")
                        except ValueError:
                            date_val = None

                # --- Convert numeric fields safely (remove commas) ---
                def parse_float(v):
                    if not v:
                        return 0.0
                    if isinstance(v, (int, float)):
                        return float(v)
                    return float(str(v).replace(",", ""))

                s = Settlement(
                    agent=row.get("agent"),
                    brand=row.get("brand"),
                    amount=parse_float(row.get("amount")),
                    fee=parse_float(row.get("fee")),
                    remarks=row.get("remarks"),
                    mc=row.get("mc"),
                    purpose=row.get("purpose"),
                    wallet=row.get("wallet"),
                    date=date_val
                )
                session.add(s)
            session.commit()
            return jsonify({"success": True})
        except Exception as e:
            session.rollback()
            return jsonify({"success": False, "error": str(e)})


@app.route("/settlement/update/<int:id>", methods=["POST"])
def update_settlement(id):
    data = request.json
    with SessionLocal() as session:
        try:
            s = session.query(Settlement).filter_by(id=id).first()
            if not s:
                return jsonify({"success": False, "error": "Not found"})

            s.agent = data.get("agent")
            s.brand = data.get("brand")
            s.amount = parse_float(data.get("amount"))
            s.fee = parse_float(data.get("fee"))
            s.remarks = data.get("remarks")
            s.mc = data.get("mc")
            s.purpose = data.get("purpose")
            s.wallet = data.get("wallet")

            if data.get("date"):
                try:
                    s.date = datetime.strptime(data["date"], "%Y-%m-%d")
                except ValueError:
                    s.date = None

            session.commit()
            return jsonify({"success": True})
        except Exception as e:
            session.rollback()
            return jsonify({"success": False, "error": str(e)})

@app.route("/settlement/delete/<int:id>", methods=["POST"])
def delete_settlement(id):
    with SessionLocal() as session:
        try:
            s = session.query(Settlement).filter_by(id=id).first()
            if not s: return jsonify({"success": False, "error":"Not found"})
            session.delete(s)
            session.commit()
            return jsonify({"success": True})
        except Exception as e:
            session.rollback()
            return jsonify({"success": False, "error": str(e)})


@app.route("/settlement/import", methods=["POST"])
def import_settlements():
    file = request.files["file"]
    df = pd.read_excel(file) if file.filename.endswith(".xlsx") else pd.read_csv(file)
    with SessionLocal() as session:
        for _, row in df.iterrows():
            s = Settlement(
                agent=row.get("Agent"),
                brand=row.get("Brand"),
                amount=parse_float(row.get("Amount")),
                fee=parse_float(row.get("Fee")),
                remarks=row.get("Remarks"),
                mc=row.get("MC"),
                purpose=row.get("Purpose"),
                wallet=row.get("Wallet"),
                date=parse_excel_datetime(row.get("Date"))  # safe parsing
            )
            session.add(s)
        session.commit()
    return redirect(url_for("adjustments") + "#settlementsTab")

@app.route("/settlement/export")
def export_settlements():
    with SessionLocal() as session:
        data = session.query(Settlement).all()
    df = pd.DataFrame([{
        "Agent": s.agent,
        "Brand": s.brand,
        "Amount": s.amount,
        "Fee": s.fee,
        "Remarks": s.remarks,
        "MC": s.mc,
        "Purpose": s.purpose,
        "Wallet": s.wallet,
        "Date": s.date
    } for s in data])
    file_path = "settlements_export.xlsx"
    df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)


@app.route("/topups")
def topups():
    # Filters
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")
    brand = request.args.get("brand")
    wallet = request.args.get("wallet")
    from_agent = request.args.get("from_agent")
    to_agent = request.args.get("to_agent")
    tx_type = request.args.get("type")

    # Pagination
    page = int(request.args.get("page", 1))
    per_page = 20

    with SessionLocal() as session:
        query = session.query(TopUp)

        if date_from:
            query = query.filter(TopUp.date >= date_from)
        if date_to:
            query = query.filter(TopUp.date <= date_to)
        if brand:
            query = query.filter(TopUp.brand == brand)
        if wallet:
            query = query.filter(TopUp.wallet == wallet)
        if from_agent:
            query = query.filter(TopUp.from_agent == from_agent)
        if to_agent:
            query = query.filter(TopUp.to_agent == to_agent)
        if tx_type:
            query = query.filter(TopUp.type == tx_type)

        total = query.count()
        topups_data = query.order_by(TopUp.date.desc()).offset((page - 1) * per_page).limit(per_page).all()

        # Simple pagination object
        class SimplePagination:
            def __init__(self, page, per_page, total):
                self.page = page
                self.per_page = per_page
                self.total = total
                self.pages = (total + per_page - 1) // per_page

        pagination = SimplePagination(page, per_page, total)

        # Dropdown values
        brands = [b[0] for b in session.query(TopUp.brand).distinct().all() if b[0]]
        wallets = [w[0] for w in session.query(TopUp.wallet).distinct().all() if w[0]]
        from_agents = [fa[0] for fa in session.query(TopUp.from_agent).distinct().all() if fa[0]]
        to_agents = [ta[0] for ta in session.query(TopUp.to_agent).distinct().all() if ta[0]]
        types = [t[0] for t in session.query(TopUp.type).distinct().all() if t[0]]

    return render_template(
        "adjustments/topups.html",
        topups_data=topups_data,
        filters={
            "date_from": date_from or "",
            "date_to": date_to or "",
            "brand": brand or "",
            "wallet": wallet or "",
            "from_agent": from_agent or "",
            "to_agent": to_agent or "",
            "type": tx_type or "",
        },
        brands=brands,
        wallets=wallets,
        from_agents=from_agents,
        to_agents=to_agents,
        types=types,
        pagination=pagination
    )



@app.route("/topup/save", methods=["POST"])
def save_topup():
    try:
        data = request.json.get("data", [])
        if not data:
            return jsonify({"success": False, "error": "No data provided"})

        with SessionLocal() as session:
            for row in data:
                topup = TopUp(
                    date=row.get("date"),
                    brand=row.get("brand"),
                    from_agent=row.get("from_agent"),
                    remarks_d=row.get("remarks_d"),
                    mc=row.get("mc"),
                    type=row.get("type"),
                    to_agent=row.get("to_agent"),
                    brand_to=row.get("brand_to"),
                    wallet=row.get("wallet"),
                    amount_process=safe_float(row.get("amount_process")),
                    fee=safe_float(row.get("fee")),
                    remarks_l=row.get("remarks_l"),
                    updated_by=row.get("updated_by"),
                    status=row.get("status"),
                    checker=row.get("checker"),
                )
                session.add(topup)
            session.commit()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})



@app.route("/topup/delete/<int:id>", methods=["POST"])
def delete_topup(id):
    with SessionLocal() as session:
        topup = session.get(TopUp, id)
        if topup:
            session.delete(topup)
            session.commit()
            return jsonify({"success": True})
    return jsonify({"success": False, "error": "TopUp not found"})


@app.route("/topup/update/<int:id>", methods=["POST"])
def update_topup(id):
    data = request.json
    with SessionLocal() as session:
        topup = session.get(TopUp, id)
        if not topup:
            return jsonify({"success": False, "error": "TopUp not found"})

        try:
            # Update fields safely
            topup.date = data.get("date", topup.date)
            topup.brand = data.get("brand", topup.brand)
            topup.from_agent = data.get("from_agent", topup.from_agent)
            topup.remarks_d = data.get("remarks_d", topup.remarks_d)
            topup.mc = data.get("mc", topup.mc)
            topup.type = data.get("type", topup.type)
            topup.to_agent = data.get("to_agent", topup.to_agent)
            topup.brand_to = data.get("brand_to", topup.brand_to)
            topup.wallet = data.get("wallet", topup.wallet)
            topup.amount_process = safe_float(data.get("amount_process", topup.amount_process))
            topup.fee = safe_float(data.get("fee", topup.fee))
            topup.remarks_l = data.get("remarks_l", topup.remarks_l)
            topup.updated_by = data.get("updated_by", topup.updated_by)
            topup.status = data.get("status", topup.status)
            topup.checker = data.get("checker", topup.checker)

            session.commit()
            return jsonify({"success": True})
        except Exception as e:
            session.rollback()
            return jsonify({"success": False, "error": str(e)})

@app.route("/dashboard_new")
def dashboard_new():
    with SessionLocal() as session:
        topup_db = get_topup_dashboard(session)
        settlement_db = get_settlement_dashboard(session)
    return render_template("adjustments/dashboard_new.html",
                           topup_dashboard=topup_db,
                           settlement_dashboard=settlement_db)


settlement_dashboard = {
    'today': [5, 12345.67],
    'this_week': [30, 56789.01],
    'this_month': [120, 234567.89],
    'per_brand': [("Brand A", 10, 1000.00), ("Brand B", 20, 2000.00)],
    'per_agent': [("Agent X", 15, 1500.00), ("Agent Y", 15, 1500.00)]
}

def get_settlement_dashboard(session):
    today = datetime.today().date()
    start_week = today - timedelta(days=today.weekday())
    start_month = today.replace(day=1)

    dashboard = {}

    # Per brand
    dashboard['per_brand'] = session.query(
        Settlement.brand,
        func.count(Settlement.id),
        func.sum(Settlement.amount)
    ).group_by(Settlement.brand).all()

    # Per agent
    dashboard['per_agent'] = session.query(
        Settlement.agent,
        func.count(Settlement.id),
        func.sum(Settlement.amount)
    ).group_by(Settlement.agent).all()

    # Per remarks
    dashboard['per_remarks'] = session.query(
        Settlement.remarks,
        func.count(Settlement.id),
        func.sum(Settlement.amount)
    ).group_by(Settlement.remarks).all()

    # Per wallet
    dashboard['per_wallet'] = session.query(
        Settlement.wallet,
        func.count(Settlement.id),
        func.sum(Settlement.amount)
    ).group_by(Settlement.wallet).all()

    # Per day
    dashboard['today'] = session.query(
        func.count(Settlement.id),
        func.sum(Settlement.amount)
    ).filter(func.date(Settlement.date) == today).first()

    # This week
    dashboard['this_week'] = session.query(
        func.count(Settlement.id),
        func.sum(Settlement.amount)
    ).filter(Settlement.date >= start_week).first()

    # This month
    dashboard['this_month'] = session.query(
        func.count(Settlement.id),
        func.sum(Settlement.amount)
    ).filter(Settlement.date >= start_month).first()

    return dashboard

def get_topup_dashboard(session):
    today = datetime.today().date()
    start_week = today - timedelta(days=today.weekday())
    start_month = today.replace(day=1)

    dashboard = {}

    # Per brand
    dashboard['per_brand'] = session.query(
        TopUp.brand,
        func.count(TopUp.id),
        func.sum(TopUp.amount_process)
    ).group_by(TopUp.brand).all()

    # Per agent
    dashboard['per_agent'] = session.query(
        TopUp.from_agent,
        func.count(TopUp.id),
        func.sum(TopUp.amount_process)
    ).group_by(TopUp.from_agent).all()

    # Per type
    dashboard['per_type'] = session.query(
        TopUp.type,
        func.count(TopUp.id),
        func.sum(TopUp.amount_process)
    ).group_by(TopUp.type).all()

    # Per MC
    dashboard['per_mc'] = session.query(
        TopUp.mc,
        func.count(TopUp.id),
        func.sum(TopUp.amount_process)
    ).group_by(TopUp.mc).all()

    # Per wallet
    dashboard['per_wallet'] = session.query(
        TopUp.wallet,
        func.count(TopUp.id),
        func.sum(TopUp.amount_process)
    ).group_by(TopUp.wallet).all()

    # In / Out totals
    dashboard['out_today'] = session.query(func.sum(TopUp.amount_process)).filter(
        func.date(TopUp.date) == today, TopUp.from_agent != None
    ).scalar() or 0

    dashboard['in_today'] = session.query(func.sum(TopUp.amount_process)).filter(
        func.date(TopUp.date) == today, TopUp.to_agent != None
    ).scalar() or 0

    dashboard['out_week'] = session.query(func.sum(TopUp.amount_process)).filter(
        TopUp.date >= start_week, TopUp.from_agent != None
    ).scalar() or 0

    dashboard['in_week'] = session.query(func.sum(TopUp.amount_process)).filter(
        TopUp.date >= start_week, TopUp.to_agent != None
    ).scalar() or 0

    dashboard['out_month'] = session.query(func.sum(TopUp.amount_process)).filter(
        TopUp.date >= start_month, TopUp.from_agent != None
    ).scalar() or 0

    dashboard['in_month'] = session.query(func.sum(TopUp.amount_process)).filter(
        TopUp.date >= start_month, TopUp.to_agent != None
    ).scalar() or 0

    return dashboard


@app.route("/accounts/management")
def accounts_management():
    session = SessionLocal()
    sdps = session.query(Sdp).all()
    session.close()
    return render_template("accounts/management.html",
                           sdps=sdps,
                           show_accounts_header=True,

    )


# --- Run ---
@app.route('/accounts/save_sdps', methods=['POST'])
def save_sdps():
    session = SessionLocal()
    try:
        data = request.get_json()
        if not data or 'sdps' not in data:
            return jsonify({"message": "No data received"}), 400

        sdps_list = data['sdps']

        if not sdps_list:
            # If empty array, delete all rows
            session.query(Sdp).delete()
            session.commit()
            return jsonify({"message": "All SDPs cleared successfully"})

        # Otherwise, update existing or add new
        for item in sdps_list:
            shop = item.get('shop', '').strip()
            if not shop:
                continue

            sdp_obj = session.query(Sdp).filter(Sdp.shop == shop).first()
            if sdp_obj:
                sdp_obj.sdp = item.get('sdp', '')
                sdp_obj.group_code = item.get('group_code', '')
                sdp_obj.chat_id = item.get('chat_id', '')
                sdp_obj.tg_link = item.get('tg_link', '')
                sdp_obj.remarks = item.get('remarks', '')
            else:
                sdp_obj = Sdp(
                    shop=shop,
                    sdp=item.get('sdp', ''),
                    group_code=item.get('group_code', ''),
                    chat_id=item.get('chat_id', ''),
                    tg_link=item.get('tg_link', ''),
                    remarks=item.get('remarks', '')
                )
                session.add(sdp_obj)
        session.commit()
        return jsonify({"message": "Accounts saved successfully"})
    except Exception as e:
        session.rollback()
        return jsonify({"message": f"Error saving data: {str(e)}"}), 500
    finally:
        session.close()

@app.route("/ramborghini")
def ramborghini():
    return render_template("ramborghini.html", show_accounts_header=False, active_page="ramborghini", current_year=datetime.now().year)

@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # ✅ OpenAI Whisper API
        with open(tmp_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        transcript = result.text
        return jsonify({"transcript": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass




@app.route("/api/ai", methods=["POST"])
def ai_reply():
    data = request.get_json()
    history = data.get("history", [])
    latest = data.get("latest_transcript", "")

    # Convert to OpenAI chat format
    messages = []
    for m in history:
        role = "assistant" if m.get("role") == "assistant" else "user"
        messages.append({"role": role, "content": m.get("text", "")})
    messages.append({"role": "user", "content": latest})

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        reply = completion.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


port = int(os.environ.get("PORT", 8080))  # default 8080 if PORT not set

if os.environ.get("FLASK_ENV") == "production":
    serve(app, host="0.0.0.0", port=port)
else:
    app.run(debug=True, host="0.0.0.0", port=port)

