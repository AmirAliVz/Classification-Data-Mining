
import pandas as pd

# Part D. Data Preparation
# =========================
# Step 1: Load Data
df = pd.read_csv("data/medical_clean.csv")
df_clean = df.copy()

# Step 2: Drop Irrelevant Columns
df_clean = df_clean.drop(columns=[
    'CaseOrder', 'Customer_id', 'Interaction', 'UID',
    'City', 'State', 'County', 'Zip', 'Lat', 'Lng', 'Population', 'TimeZone', 'Job'
])

# Step 3: Handle Missing Values
df_clean = df_clean.dropna()

# Step 4: Encode Target Variable
df_clean['ReAdmis'] = df_clean['ReAdmis'].map({'Yes': 1, 'No': 0})

# Step 5: Encode Binary Variables
binary_cols = [
    'HighBlood', 'Stroke', 'Overweight', 'Arthritis', 'Diabetes',
    'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis',
    'Reflux_esophagitis', 'Asthma', 'Soft_drink'
]

for col in binary_cols:
    df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})

# Step 6: One-Hot Encode Categorical Variables
df_clean = pd.get_dummies(df_clean, columns=[
    'Gender', 'Marital', 'Area',
    'Initial_admin', 'Complication_risk', 'Services'
], drop_first=True)

# Step 7: Convert boolean columns to 0 and 1 for consistency
bool_cols = df_clean.select_dtypes(include='bool').columns
df_clean[bool_cols] = df_clean[bool_cols].astype(int)

# Step 8: Saving the Cleaned Dataset
df_clean.to_csv("data/medical_clean_processed.csv", index=False)