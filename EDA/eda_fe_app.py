import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

st.set_page_config(page_title="EDA App", layout="wide")
st.title("📊 Exploratory Data Analysis App (with Advanced Plots & Feature Engineering)")

@st.cache_data(show_spinner=False)
def load_csv_flex(file):
    raw = file.read()
    sample = raw[:4096].decode("utf-8", errors="ignore")
    import csv
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except Exception:
        sep = ","
    file.seek(0)
    return pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    return (series < lower) | (series > upper)

def download_plotly_fig(fig, filename="figure.png"):
    import plotly.io as pio
    buf = io.BytesIO()
    pio.write_image(fig, buf, format="png")
    buf.seek(0)
    st.download_button("Download chart as PNG", data=buf, file_name=filename, mime="image/png")

def get_sample(df, n=3000):
    if len(df) > n:
        return df.sample(n=n, random_state=42)
    else:
        return df

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = load_csv_flex(uploaded_file)
    st.sidebar.success(f"Loaded data with {df.shape[0]:,} rows and {df.shape[1]:,} columns.")

    st.sidebar.markdown("### Variable Type Selection")
    all_columns = df.columns.tolist()

    date_cols = st.sidebar.multiselect("Select date columns", all_columns)
    continuous_cols = st.sidebar.multiselect(
        "Select continuous (numeric) columns", [c for c in all_columns if c not in date_cols]
    )
    categorical_cols = st.sidebar.multiselect(
        "Select categorical columns", [c for c in all_columns if c not in date_cols + continuous_cols]
    )

    # Convert user-chosen date columns
    for dcol in date_cols:
        try:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        except Exception:
            st.warning(f"Could not convert {dcol} to datetime.")

    st.sidebar.write(f"Date columns: {date_cols}")
    st.sidebar.write(f"Continuous: {continuous_cols}")
    st.sidebar.write(f"Categorical: {categorical_cols}")

    if not (continuous_cols or categorical_cols or date_cols):
        st.warning("Please select at least one variable type in the sidebar.")
    else:
        st.write("First 10 rows:")
        st.dataframe(df.head(10))

        # Feature engineering stateful copy (reset if file changes)
        if "fe_df" not in st.session_state or st.session_state["fe_df_source"] != uploaded_file.name:
            st.session_state["fe_df"] = df.copy()
            st.session_state["fe_df_source"] = uploaded_file.name
        fe_df = st.session_state["fe_df"]

        tabs = [
            "Overview", "Univariate", "Bivariate", "Missing", "Outliers",
            "Feature Engineering", "Advanced", "Export"
        ]
        tab_overview, tab_univariate, tab_bivariate, tab_missing, tab_outliers, tab_fe, tab_advanced, tab_export = st.tabs(tabs)

        # ================= Overview Tab =================
        with tab_overview:
            st.header("Data Overview")
            desc = df.describe(include='all').transpose()
            if any(df.dtypes[col] == 'datetime64[ns]' for col in date_cols):
                for col in date_cols:
                    desc.loc[col, 'min'] = df[col].min()
                    desc.loc[col, 'max'] = df[col].max()
            st.dataframe(desc)
            st.markdown(f"**Continuous columns:** {', '.join(continuous_cols) if continuous_cols else 'None'}")
            st.markdown(f"**Categorical columns:** {', '.join(categorical_cols) if categorical_cols else 'None'}")
            st.markdown(f"**Datetime columns:** {', '.join(date_cols) if date_cols else 'None'}")

        # ================= Univariate Tab =================
        with tab_univariate:
            st.header("Univariate Analysis")
            univ_col = st.selectbox("Choose a variable to plot", options=continuous_cols + categorical_cols + date_cols)
            plot_type = st.selectbox(
                "Select plot type",
                ["Histogram", "Bar (for categories)", "Violin (cat vs cont)", "Stacked Bar (two categoricals)"]
            )
            if plot_type == "Histogram" and univ_col in continuous_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[univ_col].dropna(), kde=st.checkbox("Show KDE", value=True), ax=ax)
                st.pyplot(fig)
                plt.close(fig)
                st.write(df[univ_col].describe())
            elif plot_type == "Bar (for categories)" and univ_col in categorical_cols:
                top_n = st.slider("Show top N categories", 2, 30, 10)
                counts = df[univ_col].value_counts(dropna=False).head(top_n)
                fig, ax = plt.subplots()
                sns.barplot(x=counts.values, y=counts.index, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
                st.write(counts)
            elif plot_type == "Violin (cat vs cont)":
                cat = st.selectbox("Categorical for x-axis", options=categorical_cols)
                cont = st.selectbox("Continuous for y-axis", options=continuous_cols)
                fig, ax = plt.subplots()
                sns.violinplot(x=df[cat], y=df[cont], ax=ax, inner="quartile")
                st.pyplot(fig)
                plt.close(fig)
            elif plot_type == "Stacked Bar (two categoricals)":
                cat1 = st.selectbox("First categorical", options=categorical_cols)
                cat2 = st.selectbox("Second categorical", options=[c for c in categorical_cols if c != cat1])
                data = pd.crosstab(df[cat1], df[cat2])
                fig = px.bar(data, barmode='stack')
                st.plotly_chart(fig)
                #Bug Fix 090525: disable download because of compatibility of issues streamlit cloud and plotly-kaliedo
                #download_plotly_fig(fig, "stacked_bar.png")
                st.write(data)
            elif univ_col in date_cols and plot_type == "Histogram":
                freq = st.selectbox("Date histogram frequency", options=["D", "W", "M"], format_func=lambda x: dict(D="Day",W="Week",M="Month")[x])
                df_date = df.copy()
                df_date = df_date[df_date[univ_col].notna()]
                df_date["_date"] = df_date[univ_col].dt.to_period(freq).dt.to_timestamp()
                st.bar_chart(df_date["_date"].value_counts().sort_index())

        # ================= Bivariate Tab =================
        with tab_bivariate:
            st.header("Bivariate Analysis")
            # Pairplot
            st.subheader("Pairplot (max 4 continuous variables)")
            continuous_vars = st.multiselect("Select up to 4 continuous variables for pairplot",
                                             options=continuous_cols, max_selections=4)
            if continuous_vars:
                df_small = get_sample(df[continuous_vars])
                st.info("Showing pairplot for sampled data (max 3,000 rows for speed).")
                grid = sns.pairplot(df_small)
                st.pyplot(grid.figure)
                plt.close(grid.figure)

            # Correlation Matrix
            st.subheader("Correlation Matrix")
            if len(continuous_cols) >= 2:
                corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
                corr = df[continuous_cols].corr(method=corr_method)
                fig_corr = px.imshow(corr, text_auto=False, aspect="auto",
                                     color_continuous_scale="RdBu", zmin=-1, zmax=1)
                st.plotly_chart(fig_corr, use_container_width=True)
                download_plotly_fig(fig_corr, "correlation_matrix.png")
            else:
                st.info("Need at least two continuous columns for correlation.")

            # Cramér's V
            st.subheader("Cramér's V (Categorical Association)")
            if len(categorical_cols) >= 2:
                results = []
                for i, c1 in enumerate(categorical_cols):
                    for c2 in categorical_cols[i+1:]:
                        v = cramers_v(df[c1], df[c2])
                        results.append({"Var1": c1, "Var2": c2, "Cramér's V": v})
                cramers_df = pd.DataFrame(results).sort_values("Cramér's V", ascending=False)
                st.dataframe(cramers_df)
            else:
                st.info("Need at least two categorical columns for Cramér's V.")

            # Scatter Plot with Facet & Color
            st.subheader("Scatter Plot (with Facet & Color)")
            if len(continuous_cols) >= 2:
                x_scatter = st.selectbox("X-axis (continuous)", options=continuous_cols, key="x_scatter")
                y_scatter = st.selectbox("Y-axis (continuous)", options=[c for c in continuous_cols if c != x_scatter], key="y_scatter")
                color_scatter = st.selectbox("Color by (categorical, optional)", options=["None"] + categorical_cols, key="color_scatter")
                facet_col = st.selectbox("Facet by (categorical, optional)", options=["None"] + categorical_cols, key="facet_scatter")
                cols_needed = [x_scatter, y_scatter]
                if color_scatter != "None":
                    cols_needed.append(color_scatter)
                if facet_col != "None" and facet_col not in cols_needed:
                    cols_needed.append(facet_col)
                plot_df = df[cols_needed].dropna()
                if color_scatter != "None":
                    plot_df[color_scatter] = plot_df[color_scatter].astype(str)
                if facet_col != "None":
                    plot_df[facet_col] = plot_df[facet_col].astype(str)
                fig_scatter = px.scatter(
                    plot_df, x=x_scatter, y=y_scatter,
                    color=None if color_scatter == "None" else color_scatter,
                    facet_col=None if facet_col == "None" else facet_col,
                    opacity=0.7
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                download_plotly_fig(fig_scatter, "scatter.png")
            else:
                st.info("Need at least two continuous columns for scatter plot.")

            # Violin Plot with Facet
            st.subheader("Violin Plot (continuous by categorical, with Facet)")
            if continuous_cols and categorical_cols:
                violin_y = st.selectbox("Continuous variable", options=continuous_cols, key="violin_y")
                violin_x = st.selectbox("Categorical variable", options=categorical_cols, key="violin_x")
                facet_violin = st.selectbox("Facet by (categorical, optional)", options=["None"] + [c for c in categorical_cols if c != violin_x], key="facet_violin")
                cols_needed = [violin_x, violin_y]
                if facet_violin != "None":
                    cols_needed.append(facet_violin)
                plot_df = df[cols_needed].dropna()
                plot_df[violin_x] = plot_df[violin_x].astype(str)
                if facet_violin != "None":
                    plot_df[facet_violin] = plot_df[facet_violin].astype(str)
                fig_violin = px.violin(
                    plot_df, x=violin_x, y=violin_y, box=True, points="all",
                    facet_col=None if facet_violin == "None" else facet_violin
                )
                st.plotly_chart(fig_violin, use_container_width=True)
                download_plotly_fig(fig_violin, "violin.png")
            else:
                st.info("Need at least one continuous and one categorical variable for violin plot.")

        # ================= Missing Tab =================
        with tab_missing:
            st.header("Missing Values")
            miss = df.isna().sum().sort_values(ascending=False)
            miss = miss[miss > 0]
            st.bar_chart(miss)
            st.dataframe(miss.to_frame("N Missing"))
            if st.checkbox("Show rows with missing values in selected columns?"):
                cols = st.multiselect("Columns to check for missing", options=continuous_cols + categorical_cols + date_cols)
                if cols:
                    st.dataframe(df[df[cols].isna().any(axis=1)])
            csv_miss = df[df.isna().any(axis=1)]
            csv_bytes = csv_miss.to_csv(index=False).encode("utf-8")
            st.download_button("Download rows with missing values (CSV)", data=csv_bytes, file_name="missing_rows.csv", mime="text/csv")

        # ================= Outliers Tab =================
        with tab_outliers:
            st.header("Outlier Detection (IQR method)")
            outlier_cols = st.multiselect("Choose continuous columns to check for outliers", options=continuous_cols, default=continuous_cols)
            if outlier_cols:
                mask = np.zeros(len(df), dtype=bool)
                for col in outlier_cols:
                    mask = mask | iqr_outliers(df[col])
                outlier_rows = df[mask]
                st.write(f"Rows flagged as outliers in selected columns: {len(outlier_rows)}")
                st.dataframe(outlier_rows.head(100))
                st.download_button("Download outlier rows (CSV)", data=outlier_rows.to_csv(index=False).encode("utf-8"), file_name="outliers.csv", mime="text/csv")
                cleaned = df[~mask]
                st.download_button("Download dataset with outliers removed", data=cleaned.to_csv(index=False).encode("utf-8"), file_name="dataset_no_outliers.csv", mime="text/csv")

        # ================= Feature Engineering Tab =================
        with tab_fe:
            st.header("Feature Engineering (Queue System - Apply All at Once)")
            # Set up session state for queue and output
            if "fe_queue" not in st.session_state or st.session_state.get("fe_queue_reset") != uploaded_file.name:
                st.session_state.fe_queue = []
                st.session_state.fe_queue_reset = uploaded_file.name

            queue = st.session_state.fe_queue

            col = st.selectbox("Select column to engineer", df.columns)
            dtype = df[col].dtype

            # Dynamically show options based on dtype
            op_type = None
            if pd.api.types.is_numeric_dtype(dtype):
                op_type = st.selectbox(
                    "Feature Engineering Operation",
                    ["Impute (Mean)", "Impute (Median)", "Impute (Mode)", "Impute (Fill Value)",
                    "Log Transform", "Sqrt Transform", "Standardize", "Min-Max Scale", "Binning"]
                )
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                op_type = st.selectbox(
                    "Feature Engineering Operation",
                    ["Impute (Fill Value)", "Extract Year", "Extract Month", "Extract Day", "Extract Weekday"]
                )
            else:  # categorical/other
                op_type = st.selectbox(
                    "Feature Engineering Operation",
                    ["Impute (Mode)", "Impute (Fill Value)", "Label Encoding", "One-Hot Encoding"]
                )

            # Additional config for some options
            config = {}
            if "Fill Value" in op_type:
                config['fill_value'] = st.text_input("Value to fill missing values with")
            if op_type == "Binning":
                config['bins'] = st.number_input("Number of bins", min_value=2, max_value=20, value=5)
                config['bin_label'] = st.text_input("Name for binned column", f"{col}_bin")

            # Add to queue
            if st.button("Add to Queue"):
                queue.append({'col': col, 'dtype': str(dtype), 'op_type': op_type, 'config': config.copy()})
                st.success(f"Added to queue: {col} - {op_type}")

            # Display the queue
            st.write("**Feature Engineering Queue:**")
            if queue:
                for i, item in enumerate(queue):
                    cdesc = f"{item['col']} ({item['dtype']}): {item['op_type']}"
                    if item['config']:
                        cdesc += f" | Params: {item['config']}"
                    col1, col2 = st.columns([6,1])
                    col1.write(f"{i+1}. {cdesc}")
                    if col2.button("Delete", key=f"del_{i}"):
                        queue.pop(i)
                        st.rerun()
            else:
                st.info("No operations in queue yet.")

            # Apply all
            if st.button("Apply All Operations"):
                fe_df = df.copy()
                for step in queue:
                    col = step['col']
                    op = step['op_type']
                    config = step['config']
                    if op == "Impute (Mean)":
                        fe_df[f"{col}_impute_mean"] = fe_df[col].fillna(fe_df[col].mean())
                    elif op == "Impute (Median)":
                        fe_df[f"{col}_impute_median"] = fe_df[col].fillna(fe_df[col].median())
                    elif op == "Impute (Mode)":
                        fe_df[f"{col}_impute_mode"] = fe_df[col].fillna(fe_df[col].mode().iloc[0])
                    elif op == "Impute (Fill Value)":
                        fe_df[f"{col}_impute_fill"] = fe_df[col].fillna(config.get('fill_value', ''))
                    elif op == "Log Transform":
                        fe_df[f"{col}_log"] = np.log1p(fe_df[col])
                    elif op == "Sqrt Transform":
                        fe_df[f"{col}_sqrt"] = np.sqrt(fe_df[col])
                    elif op == "Standardize":
                        scaler = StandardScaler()
                        fe_df[f"{col}_std"] = scaler.fit_transform(fe_df[[col]])
                    elif op == "Min-Max Scale":
                        scaler = MinMaxScaler()
                        fe_df[f"{col}_minmax"] = scaler.fit_transform(fe_df[[col]])
                    elif op == "Binning":
                        fe_df[config.get('bin_label', f"{col}_bin")] = pd.cut(fe_df[col], bins=int(config.get('bins',5)), labels=False)
                    elif op == "Label Encoding":
                        le = LabelEncoder()
                        fe_df[f"{col}_label_enc"] = le.fit_transform(fe_df[col].astype(str))
                    elif op == "One-Hot Encoding":
                        dummies = pd.get_dummies(fe_df[col], prefix=col)
                        for dcol in dummies.columns:
                            fe_df[dcol] = dummies[dcol]
                    elif op == "Extract Year":
                        fe_df[f"{col}_year"] = fe_df[col].dt.year
                    elif op == "Extract Month":
                        fe_df[f"{col}_month"] = fe_df[col].dt.month
                    elif op == "Extract Day":
                        fe_df[f"{col}_day"] = fe_df[col].dt.day
                    elif op == "Extract Weekday":
                        fe_df[f"{col}_weekday"] = fe_df[col].dt.weekday
                st.session_state["fe_applied_df"] = fe_df
                st.success("All feature engineering steps applied! Preview below.")

            # Reset
            if st.button("Reset Queue"):
                st.session_state.fe_queue = []
                if "fe_applied_df" in st.session_state:
                    del st.session_state["fe_applied_df"]
                st.experimental_rerun()

            # Preview
            fe_applied_df = st.session_state.get("fe_applied_df", df)
            st.write("**Preview of feature-engineered data:**")
            st.dataframe(fe_applied_df.head(10))
            st.download_button("Download feature-engineered data", data=fe_applied_df.to_csv(index=False).encode("utf-8"), file_name="feature_engineered.csv", mime="text/csv")


        # ================= Advanced Tab =================
        with tab_advanced:
            st.header("Advanced Visualizations")
            adv_plot = st.selectbox(
                "Select advanced plot type",
                [
                    "Ridge (Joy) Plot", "Parallel Coordinates", "Hexbin Density Scatter",
                    "Mosaic Plot", "Missingno Heatmap", "Missingno Matrix",
                    "Line Plot (Time Series)", "QQ Plot", "Autocorrelation Plot",
                    "Swarm Plot", "Strip Plot", "Clustermap", "Dendrogram",
                    "Sunburst", "Treemap", "Geospatial Map", "3D Scatter"
                ]
            )

            try:
                if adv_plot == "Ridge (Joy) Plot":
                    cat = st.selectbox("Category for ridges", options=categorical_cols)
                    cont = st.selectbox("Continuous for distribution", options=continuous_cols)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.violinplot(x=cont, y=cat, data=df, scale="width", inner=None, cut=0)
                    st.pyplot(fig)
                    plt.close(fig)

                elif adv_plot == "Parallel Coordinates":
                    from pandas.plotting import parallel_coordinates
                    target_col = st.selectbox("Target/categorical to color by", options=categorical_cols)
                    plot_cols = st.multiselect("Columns to include", continuous_cols, default=continuous_cols[:5])
                    fig, ax = plt.subplots(figsize=(8, 6))
                    parallel_coordinates(df.dropna(subset=plot_cols + [target_col]), class_column=target_col, cols=plot_cols, ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)

                elif adv_plot == "Hexbin Density Scatter":
                    x = st.selectbox("X (continuous)", options=continuous_cols)
                    y = st.selectbox("Y (continuous)", options=[c for c in continuous_cols if c != x])
                    fig, ax = plt.subplots()
                    df2 = df[[x, y]].dropna()
                    ax.hexbin(df2[x], df2[y], gridsize=30, cmap='Blues')
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    st.pyplot(fig)
                    plt.close(fig)

                elif adv_plot == "Mosaic Plot":
                    from statsmodels.graphics.mosaicplot import mosaic
                    cat1 = st.selectbox("First categorical", options=categorical_cols)
                    cat2 = st.selectbox("Second categorical", options=[c for c in categorical_cols if c != cat1])
                    fig, _ = mosaic(df, [cat1, cat2], gap=0.01)
                    st.pyplot(fig)
                    plt.close(fig)

                elif adv_plot == "Missingno Heatmap":
                    import missingno as msno
                    fig = msno.heatmap(df)
                    st.pyplot(fig.figure)
                    plt.close(fig.figure)

                elif adv_plot == "Missingno Matrix":
                    import missingno as msno
                    fig = msno.matrix(df)
                    st.pyplot(fig.figure)
                    plt.close(fig.figure)

                elif adv_plot == "Line Plot (Time Series)":
                    date = st.selectbox("Date column", options=date_cols)
                    cont = st.selectbox("Continuous value", options=continuous_cols)
                    fig = px.line(df.sort_values(date), x=date, y=cont)
                    st.plotly_chart(fig, use_container_width=True)
                    download_plotly_fig(fig, "line_plot.png")

                elif adv_plot == "QQ Plot":
                    import scipy.stats as stats
                    col = st.selectbox("Continuous column for QQ plot", options=continuous_cols)
                    fig, ax = plt.subplots()
                    stats.probplot(df[col].dropna(), dist="norm", plot=ax)
                    st.pyplot(fig)
                    plt.close(fig)

                elif adv_plot == "Autocorrelation Plot":
                    col = st.selectbox("Continuous column", options=continuous_cols)
                    import pandas.plotting as pd_plot
                    fig, ax = plt.subplots()
                    pd_plot.autocorrelation_plot(df[col].dropna(), ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)

                elif adv_plot == "Swarm Plot":
                    cat = st.selectbox("Categorical for x-axis", options=categorical_cols)
                    cont = st.selectbox("Continuous for y-axis", options=continuous_cols)
                    fig, ax = plt.subplots()
                    sns.swarmplot(x=cat, y=cont, data=df)
                    st.pyplot(fig)
                    plt.close(fig)

                elif adv_plot == "Strip Plot":
                    cat = st.selectbox("Categorical for x-axis", options=categorical_cols)
                    cont = st.selectbox("Continuous for y-axis", options=continuous_cols)
                    fig, ax = plt.subplots()
                    sns.stripplot(x=cat, y=cont, data=df)
                    st.pyplot(fig)
                    plt.close(fig)

                elif adv_plot == "Clustermap":
                    if len(continuous_cols) < 2:
                        st.warning("Need at least 2 continuous columns for clustermap.")
                    else:
                        cm_df = df[continuous_cols].dropna()
                        g = sns.clustermap(cm_df, figsize=(8,8))
                        st.pyplot(g.fig)
                        plt.close(g.fig)

                elif adv_plot == "Dendrogram":
                    from scipy.cluster.hierarchy import dendrogram, linkage
                    if len(continuous_cols) < 2:
                        st.warning("Need at least 2 continuous columns for dendrogram.")
                    else:
                        Z = linkage(df[continuous_cols].dropna(), 'ward')
                        fig, ax = plt.subplots(figsize=(8, 4))
                        dendrogram(Z, ax=ax)
                        st.pyplot(fig)
                        plt.close(fig)

                elif adv_plot == "Sunburst":
                    if len(categorical_cols) < 2:
                        st.warning("Need at least 2 categorical columns for sunburst.")
                    else:
                        cats = st.multiselect("Levels for sunburst (hierarchy)", categorical_cols, default=categorical_cols[:2])
                        val = st.selectbox("Size/count column (optional)", options=["None"] + continuous_cols)
                        fig = px.sunburst(df, path=cats, values=None if val=="None" else val)
                        st.plotly_chart(fig, use_container_width=True)
                        download_plotly_fig(fig, "sunburst.png")

                elif adv_plot == "Treemap":
                    if len(categorical_cols) < 2:
                        st.warning("Need at least 2 categorical columns for treemap.")
                    else:
                        cats = st.multiselect("Levels for treemap (hierarchy)", categorical_cols, default=categorical_cols[:2])
                        val = st.selectbox("Size/count column (optional)", options=["None"] + continuous_cols)
                        fig = px.treemap(df, path=cats, values=None if val=="None" else val)
                        st.plotly_chart(fig, use_container_width=True)
                        download_plotly_fig(fig, "treemap.png")

                elif adv_plot == "Geospatial Map":
                    geo_cols = st.multiselect("Select lat/lon columns", options=all_columns, default=None)
                    if len(geo_cols) == 2:
                        fig = px.scatter_mapbox(df, lat=geo_cols[0], lon=geo_cols[1], zoom=2)
                        fig.update_layout(mapbox_style="carto-positron")
                        st.plotly_chart(fig, use_container_width=True)
                        download_plotly_fig(fig, "map.png")
                    else:
                        st.info("Select two columns (lat, lon) for mapping.")

                elif adv_plot == "3D Scatter":
                    if len(continuous_cols) < 3:
                        st.warning("Need at least 3 continuous columns for 3D scatter.")
                    else:
                        x = st.selectbox("X", options=continuous_cols, key="3dx")
                        y = st.selectbox("Y", options=[c for c in continuous_cols if c != x], key="3dy")
                        z = st.selectbox("Z", options=[c for c in continuous_cols if c not in [x, y]], key="3dz")
                        color = st.selectbox("Color by (categorical)", options=["None"] + categorical_cols, key="3dcolor")
                        plot_df = df[[x, y, z] + ([color] if color != "None" else [])].dropna()
                        if color != "None":
                            plot_df[color] = plot_df[color].astype(str)
                        fig = px.scatter_3d(
                            plot_df, x=x, y=y, z=z,
                            color=None if color == "None" else color
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        download_plotly_fig(fig, "3dscatter.png")
            except Exception as e:
                st.warning(f"Plot failed: {e}")

        # ================= Export Tab =================
        with tab_export:
            st.header("Download Data")
            out_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download current dataset (CSV)", data=out_csv, file_name="dataset_cleaned.csv", mime="text/csv")

else:
    st.info("Upload a CSV file to begin.")

st.markdown("---")
