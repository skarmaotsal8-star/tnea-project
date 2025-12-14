import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
BASE_DIR = os.path.dirname(__file__)  # folder where model_backend.py is
file_list = glob.glob(os.path.join(BASE_DIR, "dataset", "tnea*.csv"))

if not file_list:
    print("Error: No data files found. Please ensure 'tnea2022.csv' is in the folder.")
    df = pd.DataFrame()
else:
    df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)
    print(f"Data Loaded: {len(df)} rows found.")

rf_model = None
le_branch = None
le_community = None
df_final = None

if not df.empty:
    cutoff_cols = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']

    # Drop completely unused columns if present
    df_clean = df.drop(columns=['MBCDNC', 'MBCV'], errors='ignore')
    for col in cutoff_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Long format
    df_melted = df_clean.melt(
        id_vars=['College Code', 'College Name', 'Branch Code', 'Branch Name'],
        value_vars=cutoff_cols,
        var_name='Community',
        value_name='Cutoff'
    )

    # Remove missing cutoffs
    df_final = df_melted.dropna(subset=['Cutoff'])

    # ENCODING & TRAINING
    le_branch = LabelEncoder()
    le_community = LabelEncoder()

    df_train = df_final.copy()
    df_train['Branch_Encoded'] = le_branch.fit_transform(df_train['Branch Code'])
    df_train['Community_Encoded'] = le_community.fit_transform(df_train['Community'])

    X = df_train[['College Code', 'Branch_Encoded', 'Community_Encoded']]
    y = df_train['Cutoff']

    print("Training Random Forest Model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    print("Training Complete!")


def recommend_colleges(user_cutoff, user_community, user_branches):
    """
    Takes user input and returns three DataFrames:
      dream_df, ambitious_df, safe_df
    If something is wrong, returns (None, None, None, error_message)
    """
    if df_final is None or rf_model is None:
        return None, None, None, "Model or data not loaded."

    unique_colleges = df_final[['College Code', 'College Name']].drop_duplicates()
    results = []

    # Encode community
    try:
        comm_enc = le_community.transform([user_community])[0]
    except Exception:
        return None, None, None, f"Community '{user_community}' not found in training data."

    # Loop through each preferred branch
    for branch in user_branches:
        try:
            branch_enc = le_branch.transform([branch])[0]
        except Exception:
            # Skip invalid branch codes
            continue

        temp_df = unique_colleges.copy()
        temp_df['Branch_Encoded'] = branch_enc
        temp_df['Community_Encoded'] = comm_enc
        temp_df['Branch'] = branch

        preds = rf_model.predict(
            temp_df[['College Code', 'Branch_Encoded', 'Community_Encoded']]
        )
        temp_df['Predicted_Cutoff'] = preds

        def get_category(row):
            pred = row['Predicted_Cutoff']
            diff = user_cutoff - pred

            if diff >= 3:
                return "🟢 SAFE"
            elif -5 <= diff < 3:
                return "🟠 AMBITIOUS"
            else:
                return "🔴 DREAM"

        temp_df['Category'] = temp_df.apply(get_category, axis=1)
        results.append(temp_df)

    if not results:
        return None, None, None, "No valid branches found or no results."

    final_df = pd.concat(results).sort_values(
        by='Predicted_Cutoff', ascending=False
    )

    dream_df = final_df[final_df['Category'] == '🔴 DREAM'][
        ['College Name', 'Branch', 'Predicted_Cutoff']
    ]
    ambitious_df = final_df[final_df['Category'] == '🟠 AMBITIOUS'][
        ['College Name', 'Branch', 'Predicted_Cutoff']
    ]
    safe_df = final_df[final_df['Category'] == '🟢 SAFE'][
        ['College Name', 'Branch', 'Predicted_Cutoff']
    ]

    return dream_df, ambitious_df, safe_df, None
