import streamlit as st
import pandas as pd
import altair as alt

from model_backend import recommend_colleges, df_final

st.set_page_config(page_title="TNEA College Recommendation", layout="wide")

st.title("🎓 TNEA College Recommendation System")

st.markdown(
    "This system predicts **expected TNEA cutoffs** using historical data and "
    "categorizes colleges into **Dream, Ambitious, and Safe** options."
)

if df_final is None or df_final.empty:
    st.error("Dataset not loaded.")
    st.stop()

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("Student Details")

    user_cutoff = st.number_input(
        "Cutoff (out of 200)",
        min_value=0.0,
        max_value=200.0,
        value=180.0,
        step=0.25
    )

    community = st.selectbox(
        "Community",
        sorted(df_final["Community"].unique())
    )

    branches = st.multiselect(
        "Preferred Branches (Optional)",
        sorted(df_final["Branch Name"].unique())
    )

    submit = st.button("🔍 Get Recommendations")

# ===================== RESULTS =====================
if submit:
    dream, ambitious, safe, error = recommend_colleges(
        user_cutoff, community, branches
    )

    if error:
        st.error(error)
    else:
        with st.expander("ℹ️ How are categories decided?"):
            st.write(
                """
                • **Dream** – Colleges where predicted cutoff is higher than your score  
                • **Ambitious** – Colleges close to your cutoff  
                • **Safe** – Colleges where your cutoff is much higher  
                """
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🔴 Dream Colleges")
            if dream.empty:
                st.info(
                    "No dream colleges found. "
                    "Your cutoff is high compared to predicted cutoffs."
                )
            else:
                st.dataframe(dream, use_container_width=True)

        with col2:
            st.subheader("🟠 Ambitious Colleges")
            if ambitious.empty:
                st.info("No ambitious colleges for this cutoff.")
            else:
                st.dataframe(ambitious, use_container_width=True)

        with col3:
            st.subheader("🟢 Safe Colleges")
            if safe.empty:
                st.info("No safe colleges found.")
            else:
                st.dataframe(safe, use_container_width=True)

        # ===================== GRAPH =====================
        st.subheader("📊 Predicted Cutoff Comparison")

        graph_df = pd.concat([
            dream.assign(Category="Dream"),
            ambitious.assign(Category="Ambitious"),
            safe.assign(Category="Safe")
        ])

        if not graph_df.empty:
            chart = alt.Chart(graph_df).mark_bar().encode(
                x=alt.X("College Name:N", sort="-y"),
                y="Predicted_Cutoff:Q",
                color="Category:N",
                tooltip=["College Name", "Branch Name", "Predicted_Cutoff"]
            ).properties(
                width=900,
                height=400
            )

            st.altair_chart(chart, use_container_width=True)
