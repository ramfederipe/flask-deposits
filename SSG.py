import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for,jsonify


from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, func, or_
from sqlalchemy.orm import sessionmaker, scoped_session
from models import Base, Deposit  # make sure Deposit model has "deleted = Column(Boolean, default=False)"
from flask import Flask
from models import Base


engine = create_engine(
    "mysql+pymysql://flaskuser:flaskpass@localhost:3306/flaskdb",
    echo=True
)

Session = sessionmaker(bind=engine)
session = Session()

Base.metadata.create_all(engine)


# --- Flask setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



def parse_datetime(dt_str):
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return None


def parse_float(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))  # remove commas
    except ValueError:
        return 0.0

def format_header(name):
    # special cases
    special = {
         "id": "ID",
        "deposit_type": "Deposit Type",
        "merchant_code": "Merchant Code",
        "customer": "Customer",
        "txnid": "TXNID",
        "currency": "Currency",
        "bank": "Bank",
        "from_account": "From Account",
        "to_account": "To Account",
        "amount": "Amount",
        "original_amount": "Original Amount",
        "rebalance": "Rebalance",
        "fee": "Fee",
        "created_time": "Created Time",
        "updated_time": "Updated Time",
        "transfer_time": "Transfer Time",
        "status": "Status",
        "audit": "Audit",
        "note_message": "Note Message",
        "refcode": "RefCode",
        "approve_by": "Approve By",
        "matched_by": "Matched By",
        "confirm_by": "Confirm By",  # fixed typo
        "last_updated": "Last Updated"

    }
    if name in special:
        return special[name]

    # general formatting: replace "_" with space and capitalize
    return name.replace("_", " ").title()

def fmt_dt(v):
    if isinstance(v, (datetime, datetime.date)):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return "" if v is None else v


def df_to_html_with_class(df):
    html = '<table class="excel-table"><thead><tr>'
    for col in df.columns:
        if col != 'row_class':
            html += f'<th>{col}</th>'
    html += '<th>Action</th></tr></thead><tbody>'

    for _, row in df.iterrows():
        row_class = row.get('row_class', '')
        html += f'<tr class="{row_class}">'
        for col in df.columns:
            if col != 'row_class':
                cell_value = '' if row[col] is None else row[col]  # ✅ blank if None
                html += f'<td>{cell_value}</td>'
        html += f'''
        <td>
            <form method="POST" action="{url_for('delete_deposit', id=row['ID'])}">
                <button type="submit" class="delete-btn">Delete</button>
            </form>
        </td>
        '''
        html += '</tr>'
    html += '</tbody></table>'
    return html

#For filter and search
def get_filtered_deposits(params):
    query = session.query(Deposit).filter(Deposit.deleted == False)

    # 🔍 Search across multiple text fields
    if params.get("search"):
        search_term = f"%{params['search']}%"
        query = query.filter(
            (Deposit.txnid.ilike(search_term)) |
            (Deposit.refcode.ilike(search_term)) |
            (Deposit.customer.ilike(search_term))
        )

    # 🎯 Apply filters
    if params.get("status"):
        query = query.filter(Deposit.status == params["status"])
    if params.get("merchant_code"):
        query = query.filter(Deposit.merchant_code == params["merchant_code"])
    if params.get("bank"):
        query = query.filter(Deposit.bank == params["bank"])
    if params.get("deposit_type"):
        query = query.filter(Deposit.deposit_type == params["deposit_type"])

    return query.all()



@app.route('/deposit', methods=['GET', 'POST'])
def deposit():
    # --- Handle Excel upload first ---
    if request.method == 'POST':
        file = request.files.get('excel_file')
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_excel(filepath, engine='openpyxl').replace({np.nan: ''})

            for _, row in df.iterrows():
                txnid = str(row.get('TXNID', '')).strip()
                status = str(row.get('Status', '')).strip()

                # Skip rows without TXNID
                if not txnid:
                    continue

                # 🔍 Check for duplicate (TXNID + Status)
                exists = session.query(Deposit).filter(
                    Deposit.txnid == txnid,
                    Deposit.status == status
                ).first()

                if exists:
                    continue  # Skip duplicate

                deposit_row = Deposit(
                    deposit_type=row.get('Deposit Type', ''),
                    merchant_code=row.get('Merchant Code', ''),
                    txnid=txnid,
                    customer=row.get('Customer', ''),
                    currency=row.get('Currency', ''),
                    bank=row.get('Bank', ''),
                    from_account=row.get('From Account', ''),
                    to_account=row.get('To Account', ''),
                    amount=parse_float(row.get('Amount')),
                    original_amount=parse_float(row.get('Original Amount')),
                    rebalance=parse_float(row.get('ReBalance')),
                    fee=parse_float(row.get('Fee')),
                    created_time=parse_datetime(str(row.get('Created Time'))) or datetime.utcnow(),
                    updated_time=parse_datetime(str(row.get('Updated Time'))),
                    transfer_time=parse_datetime(str(row.get('Transfer Time'))),
                    status=status,
                    audit=row.get('Audit', ''),
                    note_message=row.get('Note Message', ''),
                    refcode=row.get('RefCode', ''),
                    approve_by=row.get('Approve By', ''),
                    matched_by=row.get('Matched By', ''),
                    confirm_by=row.get('Confirm By', ''),
                    last_updated=datetime.utcnow(),
                    deleted=False
                )
                session.add(deposit_row)

                try:
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print("Error inserting deposits:", e)

    # --- Build filter params ---
    params = {
        "date": request.values.get("date", "").strip(),
        "search": request.values.get("search", "").strip(),
        "status": request.values.get("status", "").strip(),
        "merchant_code": request.values.get("merchant_code", "").strip(),
        "bank": request.values.get("bank", "").strip(),
        "deposit_type": request.values.get("deposit_type", "").strip()
    }
    params = {k: v for k, v in params.items() if v}

    # --- Base query ---
    query = session.query(Deposit).filter(Deposit.deleted == False)

    query_date = request.args.get("date")
    search = request.args.get("search", "").strip()

    if query_date:  # user selected a date
        try:
            parsed_date = datetime.strptime(query_date, "%Y-%m-%d").date()
            query = query.filter(func.date(Deposit.created_time) == parsed_date)
        except ValueError:
            pass
    else:
        # default to today if no date provided
        today = datetime.now().date()
        query = query.filter(func.date(Deposit.created_time) == today)

    # 🔍 Search filter
    if search:
        term = f"%{params['search']}%"
        query = query.filter(
            (Deposit.txnid.ilike(term)) |
            (Deposit.refcode.ilike(term)) |
            (Deposit.customer.ilike(term))
        )

    # 📌 Dropdown filters
    if "status" in params:
        query = query.filter(Deposit.status == params["status"])
    if "merchant_code" in params:
        query = query.filter(Deposit.merchant_code == params["merchant_code"])
    if "bank" in params:
        query = query.filter(Deposit.bank == params["bank"])
    if "deposit_type" in params:
        query = query.filter(Deposit.deposit_type == params["deposit_type"])

    # --- Final result ---
    deposits = query.order_by(Deposit.created_time.desc()).all()

    # --- Generate HTML table ---
    columns = [c.name for c in Deposit.__table__.columns if c.name != "deleted"]
    table_html = '<table id="depositTable" class="excel-table"><thead><tr>'
    for col in columns:
        table_html += f"<th>{format_header(col)}</th>"
    table_html += "<th>Actions</th></tr></thead><tbody>"

    for d in deposits:
        table_html += "<tr>"
        for col in columns:
            table_html += f"<td>{getattr(d, col) if getattr(d, col) is not None else ''}</td>"
        table_html += (
            "<td>"
            f"<form method='POST' action='{url_for('delete_deposit', deposit_id=d.id)}' style='display:inline;'>"
            "<button type='submit' class='delete-btn'>Delete</button>"
            "</form>"
            "</td>"
        )
        table_html += "</tr>"
    table_html += "</tbody></table>"

    # --- Dropdown values ---
    merchant_codes = [m[0] for m in session.query(Deposit.merchant_code).distinct().all() if m[0]]
    banks = [b[0] for b in session.query(Deposit.bank).distinct().all() if b[0]]
    statuses = [s[0] for s in session.query(Deposit.status).distinct().all() if s[0]]
    deposit_types = [d[0] for d in session.query(Deposit.deposit_type).distinct().all() if d[0]]

    return render_template(
        "deposit.html",
        table_html=table_html,
        params=params,
        merchant_codes=merchant_codes,
        banks=banks,
        statuses=statuses,
        deposit_types=deposit_types,
        columns=columns,
        selected_date=params.get("date", ""),
        format_header=format_header
    )



@app.route('/deleted_deposits')
def deleted_deposits():
    deleted_rows = session.query(Deposit).filter(Deposit.deleted == True).all()
    data = []
    for d in deleted_rows:
        data.append({
            "id": d.id,
            "deposit_type": d.deposit_type,
            "merchant_code": d.merchant_code,
            "customer": d.customer,
            "txnid": d.txnid,
            "currency": d.currency,
            "bank": d.bank,
            "from_account": d.from_account,
            "to_account": d.to_account,
            "amount": d.amount,
            "original_amount": d.original_amount,
            "rebalance": d.rebalance,
            "fee": d.fee,
            "created_time": d.created_time.strftime("%Y-%m-%d %H:%M:%S") if d.created_time else '',
            "updated_time": d.updated_time.strftime("%Y-%m-%d %H:%M:%S") if d.updated_time else '',
            "transfer_time": d.transfer_time.strftime("%Y-%m-%d %H:%M:%S") if d.transfer_time else '',
            "status": d.status,
            "audit": d.audit,
            "note_message": d.note_message,
            "refcode": d.refcode,
            "approve_by": d.approve_by,
            "matched_by": d.matched_by,
            "confirm_by": d.confirm_by,
            "last_updated": d.last_updated.strftime("%Y-%m-%d %H:%M:%S") if d.last_updated else ''
        })
    return jsonify(data)


# ---------- DELETE DEPOSIT ----------
@app.route('/deposit/delete/<int:deposit_id>', methods=['POST'])
def delete_deposit(deposit_id):
    deposit = session.get(Deposit, deposit_id)
    if deposit:
        deposit.deleted = True
        session.commit()

    # Redirect back to the deposit page with the same filters
    redirect_url = request.referrer or url_for('deposit')
    return redirect(redirect_url)


# ---------- DEPOSIT BIN ----------
@app.route('/deposit/bin')
def deposit_bin():
    deposits = session.query(Deposit).filter_by(deleted=True).all()
    columns = [c.name for c in Deposit.__table__.columns if c.name != "deleted"]

    table_html = '<table id="binTable" class="display"><thead><tr>'
    for col in columns:
        table_html += f"<th>{col.replace('_',' ').title()}</th>"
    table_html += "<th>Actions</th></tr></thead><tbody>"

    for d in deposits:
        table_html += "<tr>"
        for col in columns:
            val = getattr(d, col)
            if col in ("created_time", "updated_time", "transfer_time", "last_updated"):
                val = fmt_dt(val)
            else:
                val = "" if val is None else val
            table_html += f"<td>{val if val is not None else ''}</td>"
        table_html += f"""
        <td>
            <a href='{url_for('restore_deposit', deposit_id=d.id)}' title='Restore'>
                <i class='fa fa-undo'></i>
            </a>
            &nbsp;
            <a href='{url_for('permanent_delete', deposit_id=d.id)}' title='Delete Permanently'>
                <i class='fa fa-trash'></i>
            </a>
        </td>
        """

        table_html += "</tr>"
    table_html += "</tbody></table>"

    return render_template("deposit_bin.html", table_html=table_html)


@app.route('/deposit/restore/<int:deposit_id>')
def restore_deposit(deposit_id):
    deposit = session.query(Deposit).get(deposit_id)
    if deposit:
        deposit.deleted = False
        session.commit()
    return redirect(url_for('deposit_bin'))


@app.route('/deposit/permanent_delete/<int:deposit_id>')
def permanent_delete(deposit_id):
    deposit = session.query(Deposit).get(deposit_id)
    if deposit:
        session.delete(deposit)
        session.commit()
    return redirect(url_for('deposit_bin'))



# --- Withdrawal Upload ---
@app.route('/withdrawal', methods=['GET', 'POST'])
def withdrawal():
    table_html = ''
    if request.method == 'POST':
        file = request.files.get('excel_file')
        if file:
            filename = secure_filename(file.filename)
            unique_filename = f"{int(time.time())}_{filename}"
            filepath = os.path.join('uploads', unique_filename)
            file.save(filepath)

            df = pd.read_excel(filepath, engine='openpyxl')
            df = df.replace({np.nan: ''})

            def balance_class(val):
                if val == 0 or val == '' or val is None:
                    return 'grey'
                elif val < 500:
                    return 'low'
                elif val < 2000:
                    return 'medium'
                else:
                    return 'high'

            if 'Balance' in df.columns:
                df['row_class'] = df['Balance'].apply(balance_class)
            else:
                df['row_class'] = ''

            table_html = df.to_html(classes='excel-table', index=False, escape=False)

    return render_template('withdrawal.html', table_html=table_html)


# --- Dashboard & Static Pages ---
@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/accounts')
def accounts():
    return render_template('accounts.html')


@app.route('/settings')
def settings():
    return render_template('settings.html')


if __name__ == '__main__':
    app.run(debug=True)




