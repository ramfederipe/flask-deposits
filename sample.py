df = pd.read_excel(file)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.where(pd.notnull(df), None)

# Drop ID column
df = df.drop(columns=["id"], errors="ignore")

# Compute agent_number & shop_name
df["agent_number"], df["shop_name"] = zip(*df["to_account"].apply(split_account))

# Convert numeric columns
float_cols = ["amount", "original_amount", "rebalance", "fee"]
for col in float_cols:
    df[col] = df[col].apply(safe_float)

# Convert datetime columns
dt_cols = ["created_time", "updated_time", "transfer_time"]
for col in dt_cols:
    df[col] = df[col].apply(parse_excel_datetime)

df["imported_at"] = datetime.utcnow()

# Drop rows missing required fields
df.dropna(subset=["txnid", "to_account", "original_amount"], inplace=True)

# Prepare dicts for bulk insert
records = df.to_dict(orient="records")

# Bulk insert safely
with SessionLocal() as session:
    bulk_insert_deposits(session, records)