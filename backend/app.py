import streamlit as st
import pandas as pd

from model_backend import recommend_colleges, df_final, le_community, le_branch

st.set_page_config(page_title="TNEA College Recommendation", layout="wide")

st.title("🎓 TNEA College Recommendation System")

if df_final is None:
    st.error("Data not loaded. Make sure tnea*.csv files are in this folder.")
    st.stop()

st.markdown("Enter your **cutoff**, select your **community**, and choose preferred **departments (branches)**.")

# ---------- INPUTS ----------
# Cutoff
user_cutoff = st.number_input(
    "Cutoff (out of 200)",
    min_value=0.0,
    max_value=200.0,
    step=0.25,
    value=180.0,
)

# Community options from the data (e.g., OC, BC, MBC, etc.)
community_options = sorted(df_final["Community"].unique().tolist())
user_community = st.selectbox("Community", community_options)

# Department options – use Branch Code (CS, IT, EC, etc.)
branch_options = sorted(df_final["Branch Code"].unique().tolist())
default_branches = ["CS"] if "CS" in branch_options else []
user_branches = st.multiselect(
    "Preferred Departments (Branch Codes)",
    branch_options,
    default=default_branches,
    help="Example: CS, IT, EC, ME, CE..."
)

st.write("---")

# ---------- SUBMIT BUTTON ----------
if st.button("🔍 Get Recommendations"):
    if not user_branches:
        st.warning("Please select at least one department.")
    else:
        dream_df, ambitious_df, safe_df, error_msg = recommend_colleges(
            user_cutoff=user_cutoff,
            user_community=user_community,
            user_branches=user_branches
        )

        if error_msg:
            st.error(error_msg)
        else:
            # Show results in three columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("🔴 Dream Colleges")
                if dream_df is None or dream_df.empty:
                    st.info("No dream colleges found for the given input.")
                else:
                    st.dataframe(
                        dream_df.head(20).reset_index(drop=True),
                        use_container_width=True
                    )

            with col2:
                st.subheader("🟠 Ambitious / Likely")
                if ambitious_df is None or ambitious_df.empty:
                    st.info("No ambitious colleges found for the given input.")
                else:
                    st.dataframe(
                        ambitious_df.head(20).reset_index(drop=True),
                        use_container_width=True
                    )

            with col3:
                st.subheader("🟢 Safe Colleges")
                if safe_df is None or safe_df.empty:
                    st.info("No safe colleges found for the given input.")
                else:
                    st.dataframe(
                        safe_df.head(20).reset_index(drop=True),
                        use_container_width=True
                    )
