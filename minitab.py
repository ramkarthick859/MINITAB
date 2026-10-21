import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Minitab Web App & RSM", page_icon="📊", layout="wide"
)

st.title("📊 Mini-Tab: Statistical Analysis & Response Surface Methodology")
st.write(
    "Upload your dataset or use the sample Central Composite Design (CCD) data to perform statistical analysis and optimization using RSM."
)

# --- SIDEBAR: DATA INPUT ---
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file", type=["csv", "xlsx", "xls"]
)

# Load data logic
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
        df = None
else:
    # Default Sample RSM Dataset (Central Composite Design style)
    st.sidebar.info(
        "No file uploaded. Using default Central Composite Design (CCD) sample data."
    )
    df = pd.DataFrame(
        {
            "Run": range(1, 12),
            "Factor_A": [-1, 1, -1, 1, 0, 0, 0, -1.414, 1.414, 0, 0],
            "Factor_B": [-1, -1, 1, 1, 0, 0, 0, 0, 0, -1.414, 1.414],
            "Response_Y": [45.2, 55.6, 50.1, 60.4, 72.1, 70.0, 71.5, 42.0, 58.3, 48.9, 52.1],
        }
    )

if df is not None:
    # --- NAVIGATION TABS ---
    tab1, tab2, tab3 = st.tabs(
        [
            "📁 Data Preview & Summary",
            "📈 Basic Statistics & Plots",
            "🚀 Response Surface Methodology (RSM)",
        ]
    )

    # --- TAB 1: DATA PREVIEW ---
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    # --- TAB 2: BASIC STATS & PLOTS ---
    with tab2:
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax_corr
            )
            st.pyplot(fig_corr)
        else:
            st.warning("No numeric columns available for correlation.")

    # --- TAB 3: RESPONSE SURFACE METHODOLOGY ---
    with tab3:
        st.subheader("Response Surface Methodology (RSM) Optimization")
        st.write(
            "Fit a full quadratic model ($Y = \\beta_0 + \\beta_1 A + \\beta_2 B + \\beta_{11} A^2 + \\beta_{22} B^2 + \\beta_{12} AB$) to analyze factor effects and interactions."
        )

        numeric_columns = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if len(numeric_columns) >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                factor_a = st.selectbox(
                    "Select Factor A (Numeric)",
                    numeric_columns,
                    index=1 if len(numeric_columns) > 1 else 0,
                )
            with col2:
                factor_b = st.selectbox(
                    "Select Factor B (Numeric)",
                    numeric_columns,
                    index=2 if len(numeric_columns) > 2 else 0,
                )
            with col3:
                response_y = st.selectbox(
                    "Select Response Variable (Y)",
                    numeric_columns,
                    index=len(numeric_columns) - 1,
                )

            if factor_a != factor_b and factor_a != response_y and factor_b != response_y:
                # Prepare data for modeling
                model_df = df[[factor_a, factor_b, response_y]].dropna()
                model_df.columns = ["A", "B", "Y"]

                # Fit OLS Quadratic Model
                formula = "Y ~ A + B + I(A**2) + I(B**2) + A:B"
                try:
                    rsm_model = smf.ols(formula, data=model_df).fit()

                    st.markdown("### Model Summary & Coefficients")
                    # Display statsmodels summary as text
                    st.text(rsm_model.summary())

                    st.markdown("### Response Surface 3D Plot")
                    fig_3d = plt.figure(figsize=(9, 6))
                    ax_3d = fig_3d.add_subplot(projection="3d")

                    # Generate grid for predictions
                    a_vals = np.linspace(model_df["A"].min(), model_df["A"].max(), 40)
                    b_vals = np.linspace(model_df["B"].min(), model_df["B"].max(), 40)
                    A_grid, B_grid = np.meshgrid(a_vals, b_vals)
                    grid_data = pd.DataFrame(
                        {"A": A_grid.ravel(), "B": B_grid.ravel()}
                    )
                    Y_pred = rsm_model.predict(grid_data).values.reshape(
                        A_grid.shape
                    )

                    # Plot surface
                    surf = ax_3d.plot_surface(
                        A_grid,
                        B_grid,
                        Y_pred,
                        cmap="viridis",
                        edgecolor="none",
                        alpha=0.85,
                    )
                    ax_3d.set_xlabel(factor_a)
                    ax_3d.set_ylabel(factor_b)
                    ax_3d.set_zlabel(response_y)
                    fig_3d.colorbar(surf, shrink=0.5, aspect=10)

                    st.pyplot(fig_3d)

                except Exception as ex:
                    st.error(
                        f"Could not fit the quadratic model. Ensure your data has enough variation. Details: {ex}"
                    )
            else:
                st.warning(
                    "Please select three distinct columns for Factor A, Factor B, and Response Y."
                )
        else:
            st.warning(
                "Your dataset needs at least 3 numeric columns to run RSM analysis."
            )
