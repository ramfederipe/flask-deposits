from models import SessionLocal, Deposit, Withdrawal  # <-- real models.py

def split_account(account_str):
    if not account_str:
        return None, None
    parts = account_str.split(" - ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None, account_str.strip()

# Open session
session = SessionLocal()

# Process Withdrawals
for wd in session.query(Withdrawal).all():
    if wd.from_account:
        wd.agent_number, wd.shop_name = split_account(wd.from_account)

# Process Deposits
for dp in session.query(Deposit).all():
    if dp.from_account:
        dp.agent_number, dp.shop_name = split_account(dp.from_account)

# Save changes
session.commit()
session.close()

print("✅ Agent Number & Shop Name filled for withdrawals and deposits")
