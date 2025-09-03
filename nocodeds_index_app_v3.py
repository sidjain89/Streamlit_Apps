import streamlit as st

# Font Awesome CDN for pro icons
st.markdown(
    """
    <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css"
        crossorigin="anonymous" referrerpolicy="no-referrer" />
    """,
    unsafe_allow_html=True,
)

# ========== SETUP YOUR APPS BELOW ==========
apps = {
    "EDA/FE": {
        "fa_icon": "fa-chart-bar",
        "apps": [
            {"name": "Data Profiler and Feature Explorer", "url": "https://edafeatgen.streamlit.app/", "fa_icon": "fa-magnifying-glass-chart", "desc": "Automated EDA and Feature Engineering."},
        ],
    },
    "Regression": {
        "fa_icon": "fa-chart-line",
        "apps": [
            {"name": "Tree Regressor", "url": "https://decisiontreeclassification.streamlit.app/", "fa_icon": "fa-tree", "desc": "Train/test decision trees and ensembles."},
        ],
    },
    "Classification": {
        "fa_icon": "fa-square-poll-vertical",
        "apps": [
            {"name": "Tree Classifier", "url": "https://example.com/class1", "fa_icon": "fa-tree", "desc": "Train/test decision trees and ensembles."},
            {"name": "Logistic Regression", "url": "https://example.com/class2", "fa_icon": "fa-balance-scale", "desc": "Play with logistic regression and ROC curves."},
        ],
    },
    "Clustering": {
        "fa_icon": "fa-shapes",
        "apps": [
            {"name": "K-Means Cluster", "url": "https://kmeanswithviz.streamlit.app/", "fa_icon": "fa-bullseye", "desc": "Cluster data and visualize results using K-Means."},
            {"name": "DB Scan", "url": "https://dbscanclustering.streamlit.app/", "fa_icon": "fa-bullseye", "desc": "Cluster data and visualize results using K-Means."},
        ],
    },
    "Media Optimization": {
        "fa_icon": "fa-photo-film",
        "apps": [
            {"name": "Media Mix Model (WIP)", "url": "https://nocodeds.streamlit.app/Media_Mix_Model", "fa_icon": "fa-bullhorn", "desc": "Optimize media spends with MMM."},
        ],
    },
}
# ========== END SETUP ==========

st.set_page_config(page_title="ML Apps Landing Page", layout="wide", page_icon="✨")

# ---- CSS FOR CARD STYLING ----
st.markdown(
    """
    <style>
    .card {
        background: linear-gradient(135deg, #f7fafc 70%, #e5faff 100%);
        border-radius: 1.2em;
        box-shadow: 0 3px 20px #0d99ff12, 0 1.5px 4px #0002;
        padding: 1.4em 1.1em 1.1em 1.1em;
        min-width: 210px;
        max-width: 310px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 1.3em;
        position: relative;
    }
    .card:hover {
        box-shadow: 0 8px 32px #0d99ff30, 0 2px 6px #00ffae22;
    }
    .fa-app-main {
        font-size: 2.1em;
        margin-bottom: 0.17em;
        color: #0d99ff;
        filter: drop-shadow(0 0 1px #00b9ae22);
    }
    .app-title {
        font-size: 1.12em;
        font-weight: 700;
        margin-bottom: 0.11em;
        color: #183f5c;
    }
    .app-desc {
        font-size: 0.95em;
        color: #444d55;
        margin-bottom: 0.66em;
        min-height: 2.2em;
    }
    .launch-btn {
        display: inline-block;
        padding: 0.49em 1.13em;
        background: linear-gradient(90deg, #0d99ff 50%, #00ffae 110%);
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 1.6em;
        font-size: 0.99em;
        cursor: pointer;
        text-decoration: none;
        text-align: center;
        margin-top: auto;
        box-shadow: 0 1.5px 8px #0d99ff33;
        transition: background 0.18s;
    }
    .launch-btn:hover {
        background: linear-gradient(90deg, #00ffae 30%, #0d99ff 130%);
        color: #ffffffcc !important;
    }
    .section-header {
        font-size: 1.22em;
        font-weight: 700;
        margin: 0.4em 0 0.1em 0;
        display: flex;
        align-items: center;
        gap: 0.53em;
        color: #0d99ff;
        justify-content: center;
    }
    .fa-section {
        font-size: 1.16em;
        color: #0d99ff;
        margin-right: 0.25em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- HEADER / INTRO ----
st.title("✨ ML Apps Hub")

st.markdown(
    """
    Welcome to your one-stop portal for ML & Data Science tools at Walmart!  
    *(Edit this intro to suit your needs, e.g., add usage tips, links, or a company blurb.)*
    """
)
st.markdown("---")

# ---- SECTION COLUMNS AS GRID ----
section_names = list(apps.keys())
n_sections = len(section_names)
cols = st.columns(n_sections)

for i, section in enumerate(section_names):
    with cols[i]:
        content = apps[section]
        st.markdown(
            f'<div class="section-header"><i class="fa-solid {content["fa_icon"]} fa-section"></i> {section}</div>',
            unsafe_allow_html=True
        )
        for app in content["apps"]:
            st.markdown(
                f'''
                <div class="card" title="{app["desc"]}">
                    <div>
                        <i class="fa-solid {app["fa_icon"]} fa-app-main"></i>
                        <div class="app-title">{app["name"]}</div>
                        <div class="app-desc">{app["desc"]}</div>
                    </div>
                    <a href="{app["url"]}" class="launch-btn" target="_blank" rel="noopener noreferrer">
                        <i class="fa-solid fa-arrow-up-right-from-square" style="margin-right:0.47em"></i>Launch
                    </a>
                </div>
                ''',
                unsafe_allow_html=True
            )

st.markdown("---")
st.info("💡 Tip: Hover over a card for more info. Contact admin to add new apps!")

st.caption("Built with ❤️ using Streamlit. | [Find me on GitHub](https://github.com/sidjain89/) | [Linkedin](https://www.linkedin.com/in/siddharth-jain-45256619/)")
