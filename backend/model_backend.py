import os
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ===================== LOAD DATA =====================
BASE_DIR = os.path.dirname(__file__)
files = glob.glob(os.path.join(BASE_DIR, "dataset", "tnea*.csv"))

df_final = None
rf_model = None
le_branch = LabelEncoder()
le_community = LabelEncoder()

if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    cutoff_cols = ['OC', 'BC', 'BCM', 'MBC', 'SC', 'SCA', 'ST']
    df = df.drop(columns=['MBCDNC', 'MBCV'], errors='ignore')

    for col in cutoff_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df_melted = df.melt(
        id_vars=['College Code', 'College Name', 'Branch Code', 'Branch Name'],
        value_vars=cutoff_cols,
        var_name='Community',
        value_name='Cutoff'
    )

    df_final = df_melted.dropna(subset=['Cutoff'])

    # ===================== TRAIN MODEL =====================
    df_train = df_final.copy()
    df_train['Branch_Encoded'] = le_branch.fit_transform(df_train['Branch Code'])
    df_train['Community_Encoded'] = le_community.fit_transform(df_train['Community'])

    X = df_train[['College Code', 'Branch_Encoded', 'Community_Encoded']]
    y = df_train['Cutoff']

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X, y)


# ===================== RECOMMENDATION FUNCTION =====================
def recommend_colleges(user_cutoff, user_community, user_branch_names):

    if df_final is None or rf_model is None:
        return None, None, None, "Model or data not loaded."

    # Branch Name → Branch Code (optional)
    if user_branch_names:
        branch_codes = (
            df_final[df_final["Branch Name"].isin(user_branch_names)]
            ["Branch Code"].unique().tolist()
        )
    else:
        branch_codes = df_final["Branch Code"].unique().tolist()

    try:
        comm_enc = le_community.transform([user_community])[0]
    except:
        return None, None, None, "Invalid community selected."

    colleges = df_final[
        ['College Code', 'College Name', 'Branch Code', 'Branch Name']
    ].drop_duplicates()

    results = []

    for code in branch_codes:
        try:
            branch_enc = le_branch.transform([code])[0]
        except:
            continue

        temp = colleges[colleges["Branch Code"] == code].copy()
        temp["Branch_Encoded"] = branch_enc
        temp["Community_Encoded"] = comm_enc

        preds = rf_model.predict(
            temp[['College Code', 'Branch_Encoded', 'Community_Encoded']]
        )
        temp["Predicted_Cutoff"] = preds

        def categorize(pred):
            diff = user_cutoff - pred
            if diff >= 3:
                return "🟢 SAFE"
            elif -5 <= diff < 3:
                return "🟠 AMBITIOUS"
            else:
                return "🔴 DREAM"

        temp["Category"] = temp["Predicted_Cutoff"].apply(categorize)
        results.append(temp)

    final_df = pd.concat(results).sort_values(
        by="Predicted_Cutoff",
        ascending=False
    )

    dream_df = final_df[final_df["Category"] == "🔴 DREAM"]
    ambitious_df = final_df[final_df["Category"] == "🟠 AMBITIOUS"]
    safe_df = final_df[final_df["Category"] == "🟢 SAFE"]

    cols = ["College Name", "Branch Name", "Predicted_Cutoff"]

    return (
        dream_df[cols].reset_index(drop=True),
        ambitious_df[cols].reset_index(drop=True),
        safe_df[cols].reset_index(drop=True),
        None
    )
