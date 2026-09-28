import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.datasets import fetch_openml

st.set_page_config(page_title="PC3 Defect Dashboard", page_icon="🟢", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #F4FBF4; }
[data-testid="stMetric"] { background-color: #2E8B57; padding: 15px; border-radius: 10px; color: #FFD700; }
[data-testid="stMetricLabel"] { color: #E8F5E9 !important; }
h1, h2, h3 { color: #1E5631; }
</style>
""", unsafe_allow_html=True)

st.title("🟢🟡 PC3 Software Defect Prediction Dashboard")
st.caption("NASA PROMISE dataset — predicting defective modules from code complexity metrics")

@st.cache_data
def load_data():
    # Pulled directly from OpenML by dataset name — no local file or manual download needed.
    try:
        bunch = fetch_openml(name="pc3", version=1, as_frame=True, parser="auto")
        df = bunch.frame
    except Exception as e:
        st.error(f"Couldn't fetch the PC3 dataset from OpenML (needs an internet connection): {e}")
        st.stop()
    df.columns = [c.upper() for c in df.columns]
    target_col = "C" if "C" in df.columns else "DEFECTS"
    if df[target_col].dtype == object or str(df[target_col].dtype).startswith("category"):
        df[target_col] = df[target_col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0}).astype(int)
    else:
        df[target_col] = df[target_col].astype(int)
    df = df.rename(columns={target_col: 'C'})
    return df

df = load_data()

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['C'])
    y = df['C']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    logreg.fit(X_train_s, y_train)
    acc_lr = accuracy_score(y_test, logreg.predict(X_test_s))
    auc_lr = roc_auc_score(y_test, logreg.predict_proba(X_test_s)[:, 1])

    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    auc_rf = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    return scaler, logreg, rf, acc_lr, acc_rf, auc_lr, auc_rf, X.columns.tolist()

scaler, logreg, rf, acc_lr, acc_rf, auc_lr, auc_rf, feature_cols = train_models(df)

def col_or_median(name, default_frac=0.3):
    return name if name in feature_cols else feature_cols[int(len(feature_cols) * default_frac)]

LOC_COL = col_or_median('LOC_TOTAL')
HV_COL = col_or_median('HALSTEAD_VOLUME')
HC_COL = col_or_median('HALSTEAD_CONTENT')
UO_COL = col_or_median('NUM_UNIQUE_OPERANDS')
BLANK_COL = col_or_median('LOC_BLANK')
LINES_COL = col_or_median('NUMBER_OF_LINES')

tab1, tab2, tab3 = st.tabs(["📊 Past — EDA", "🎯 Present — Model Performance", "🔮 Future — Module Risk Assessor"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Code Modules", f"{len(df):,}")
    col2.metric("Defect Rate", f"{df['C'].mean()*100:.1f}%")
    col3.metric("Avg Lines of Code", f"{df[LOC_COL].mean():.0f}" if LOC_COL in df else "n/a")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df['C'].value_counts().plot(kind='bar', ax=axes[0], color=['#2E8B57', '#FFD700'])
    axes[0].set_xticklabels(['Clean (0)', 'Defective (1)'], rotation=0)
    axes[0].set_title('Module Defect Distribution')
    corr = df.corr(numeric_only=True)['C'].drop('C').sort_values(key=abs, ascending=False).head(10)
    corr.plot(kind='barh', ax=axes[1], color='#2E8B57')
    axes[1].set_title('Top 10 Metrics Correlated with Defects')
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression", f"Acc {acc_lr*100:.1f}% | AUC {auc_lr:.3f}")
    col2.metric("Random Forest", f"Acc {acc_rf*100:.1f}% | AUC {auc_rf:.3f}")
    st.info("Only a small share of modules are defective, so ROC-AUC matters more than raw accuracy here.")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
    importance.sort_values().plot(kind='barh', ax=ax2, color='#2E8B57')
    ax2.set_title('Top 10 Feature Importances (Random Forest)')
    plt.tight_layout()
    st.pyplot(fig2)

    # --- unique feature: review-hours cost simulator ---
    st.subheader("🟡 Estimated Review Cost If Defect Risk Goes Unchecked")
    n_modules = st.slider("Modules in this release", 10, 1000, 200)
    hrs_per_defect = st.slider("Avg debugging hours per missed defect", 1.0, 20.0, 6.0)
    expected_defects = n_modules * df['C'].mean()
    st.metric("Expected Defective Modules", f"{expected_defects:.0f}")
    st.metric("Estimated Debug Hours If Unreviewed", f"{expected_defects * hrs_per_defect:.0f} hrs")

with tab3:
    st.subheader("Module Defect Risk Assessment")
    col1, col2 = st.columns(2)
    with col1:
        loc = st.slider("Lines of Code", 1, 500, 45)
        halstead_vol = st.slider("Halstead Volume", 0.0, 3000.0, 200.0)
        halstead_content = st.slider("Halstead Content", 0.0, 100.0, 10.0)
    with col2:
        num_operands = st.slider("Number of Unique Operands", 0, 300, 30)
        loc_blank = st.slider("Blank Lines", 0, 100, 5)
        num_lines = st.slider("Number of Lines", 1, 500, 60)

    if st.button("Assess Module Risk"):
        row = {c: df[c].median() for c in feature_cols}
        row.update({LOC_COL: loc, HV_COL: halstead_vol, HC_COL: halstead_content,
                    UO_COL: num_operands, BLANK_COL: loc_blank, LINES_COL: num_lines})
        input_df = pd.DataFrame([row])[feature_cols]
        proba = rf.predict_proba(input_df)[0][1]
        if proba > 0.3:
            st.error(f"🟡⚠️ Elevated Defect Risk: {proba*100:.0f}% — recommend code review before deployment")
        else:
            st.success(f"🟢✨ Low Defect Risk: {proba*100:.0f}%")