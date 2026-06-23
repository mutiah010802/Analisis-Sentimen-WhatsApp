import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WhatsApp Sentimen Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  PASTEL COLOR PALETTE
# ─────────────────────────────────────────────
C_BG_1      = "#F4F1FB"   # lavender pastel
C_BG_2      = "#EAF6F0"   # mint pastel
C_CARD      = "#FFFFFF"
C_BORDER    = "#E7E1F7"
C_TEXT      = "#3D3D56"
C_SUBTEXT   = "#8A86A8"
C_GREEN     = "#5FD1A0"   # pastel WhatsApp green (positif)
C_GREEN_BG  = "#E3F8EF"
C_PINK      = "#FF9AAE"   # pastel coral/pink (negatif)
C_PINK_BG   = "#FFEAEF"
C_BLUE      = "#8FC6F2"   # pastel blue (akurasi)
C_BLUE_BG   = "#E9F3FC"
C_PURPLE    = "#C6AEEC"   # pastel purple (f1 / extra)
C_PURPLE_BG = "#F2EBFB"
C_YELLOW    = "#FFD180"   # pastel amber (recall)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: linear-gradient(160deg, {C_BG_1} 0%, {C_BG_2} 100%);
    color: {C_TEXT};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: #FFFFFF !important;
    border-right: 1px solid {C_BORDER};
}}

[data-testid="stSidebar"] .stRadio label {{
    color: {C_SUBTEXT} !important;
    font-size: 0.9rem;
    padding: 7px 0;
    transition: color 0.2s;
}}

[data-testid="stSidebar"] .stRadio label:hover {{
    color: #2BBE7C !important;
}}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {{
    color: #2BBE7C !important;
    font-weight: 600;
}}

/* ── Metric cards ── */
.metric-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 20px;
    padding: 22px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 6px 18px rgba(140, 120, 200, 0.08);
    transition: transform 0.2s, box-shadow 0.2s;
}}

.metric-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 26px rgba(140, 120, 200, 0.16);
}}

.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 6px;
    background: var(--accent, {C_GREEN});
    border-radius: 20px 20px 0 0;
}}

.metric-icon-wrap {{
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: var(--accent-bg, {C_GREEN_BG});
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px auto;
    font-size: 1.6rem;
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent, {C_GREEN});
    line-height: 1;
    margin-bottom: 4px;
}}

.metric-label {{
    font-size: 0.74rem;
    font-weight: 600;
    color: {C_SUBTEXT};
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}

/* ── Section headers ── */
.section-title {{
    font-size: 1.3rem;
    font-weight: 700;
    color: {C_TEXT};
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.section-title::after {{
    content: '';
    flex: 1;
    height: 2px;
    background: {C_BORDER};
    border-radius: 2px;
    margin-left: 12px;
}}

/* ── Page header ── */
.page-header {{
    padding: 22px 28px;
    border-radius: 22px;
    margin-bottom: 26px;
    background: linear-gradient(120deg, #FFFFFF 0%, {C_GREEN_BG} 100%);
    border: 1px solid {C_BORDER};
    box-shadow: 0 8px 24px rgba(140, 120, 200, 0.10);
    display: flex;
    align-items: center;
    gap: 16px;
}}

.page-header .ph-logo {{
    font-size: 2.3rem;
    background: #FFFFFF;
    border-radius: 16px;
    width: 58px;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(140,120,200,0.15);
    flex-shrink: 0;
}}

.page-header h1 {{
    font-size: 1.6rem;
    font-weight: 800;
    color: {C_TEXT};
    margin: 0;
}}

.page-header p {{
    color: {C_SUBTEXT};
    font-size: 0.88rem;
    margin: 4px 0 0 0;
}}

/* ── Insight cards ── */
.insight-box {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-left: 5px solid {C_GREEN};
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    color: {C_SUBTEXT};
    line-height: 1.6;
    box-shadow: 0 4px 14px rgba(140,120,200,0.06);
}}

.insight-box strong {{
    color: {C_TEXT};
}}

/* ── Prediction box ── */
.pred-positive {{
    background: linear-gradient(135deg, {C_GREEN_BG} 0%, #FFFFFF 100%);
    border: 1px solid {C_GREEN};
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(95, 209, 160, 0.18);
}}

.pred-negative {{
    background: linear-gradient(135deg, {C_PINK_BG} 0%, #FFFFFF 100%);
    border: 1px solid {C_PINK};
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 8px 22px rgba(255, 154, 174, 0.18);
}}

.pred-emoji {{
    font-size: 3.5rem;
    margin-bottom: 8px;
}}

.pred-label-pos {{
    font-size: 1.4rem;
    font-weight: 800;
    color: #2BBE7C;
}}

.pred-label-neg {{
    font-size: 1.4rem;
    font-weight: 800;
    color: #F2607C;
}}

/* ── Tag badge ── */
.badge {{
    display: inline-block;
    background: {C_GREEN_BG};
    color: #2BBE7C;
    border: 1px solid {C_GREEN};
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 2px;
}}

.badge-red {{
    background: {C_PINK_BG};
    color: #F2607C;
    border-color: {C_PINK};
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid {C_BORDER};
}}

/* ── Text area ── */
.stTextArea textarea {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    color: {C_TEXT} !important;
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
}}

.stTextArea textarea:focus {{
    border-color: {C_GREEN} !important;
    box-shadow: 0 0 0 3px {C_GREEN_BG} !important;
}}

.stTextInput input {{
    background: {C_CARD} !important;
    border: 1px solid {C_BORDER} !important;
    color: {C_TEXT} !important;
    border-radius: 14px !important;
}}

/* ── Button ── */
.stButton > button {{
    background: linear-gradient(120deg, {C_GREEN} 0%, #4FC79A 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 12px 32px !important;
    font-size: 0.9rem !important;
    box-shadow: 0 6px 16px rgba(95, 209, 160, 0.30) !important;
    transition: transform 0.15s, box-shadow 0.2s !important;
}}

.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 22px rgba(95, 209, 160, 0.40) !important;
}}

/* ── Selectbox ── */
.stSelectbox > div > div {{
    background: {C_CARD} !important;
    border-color: {C_BORDER} !important;
    color: {C_TEXT} !important;
    border-radius: 14px !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}

.stTabs [data-baseweb="tab"] {{
    background: {C_CARD};
    border-radius: 14px 14px 0 0;
    padding: 10px 20px;
    color: {C_SUBTEXT};
    font-weight: 600;
    border: 1px solid {C_BORDER};
}}

.stTabs [aria-selected="true"] {{
    color: #2BBE7C !important;
    background: {C_GREEN_BG} !important;
}}

/* ── About cards ── */
.about-card {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 20px;
    padding: 26px;
    height: 100%;
    box-shadow: 0 6px 18px rgba(140,120,200,0.08);
}}

.about-card h3 {{
    color: #2BBE7C;
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin: 0 0 12px 0;
}}

.about-card p, .about-card li {{
    color: {C_SUBTEXT};
    font-size: 0.89rem;
    line-height: 1.7;
}}

.about-card ul {{
    padding-left: 16px;
    margin: 0;
}}

/* ── Progress bar ── */
.stProgress > div > div > div {{
    background: linear-gradient(120deg, {C_GREEN} 0%, #4FC79A 100%) !important;
    border-radius: 6px !important;
}}

/* ── Sidebar logo block ── */
.sidebar-logo {{
    text-align:center;
    padding: 22px 0 26px 0;
}}
.sidebar-logo .logo-circle {{
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: linear-gradient(135deg, {C_GREEN} 0%, #4FC79A 100%);
    display:flex;
    align-items:center;
    justify-content:center;
    margin: 0 auto;
    font-size: 1.9rem;
    box-shadow: 0 8px 18px rgba(95,209,160,0.35);
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME (pastel / light)
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#FFFFFF",
    "axes.facecolor":    "#FFFFFF",
    "axes.edgecolor":    "#E7E1F7",
    "axes.labelcolor":   "#8A86A8",
    "xtick.color":       "#8A86A8",
    "ytick.color":       "#8A86A8",
    "text.color":        "#3D3D56",
    "grid.color":        "#EFEAFB",
    "grid.alpha":        0.8,
})

PASTEL_GREEN  = "#5FD1A0"
PASTEL_PINK   = "#FF9AAE"
PASTEL_BLUE   = "#8FC6F2"
PASTEL_PURPLE = "#C6AEEC"

# ─────────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("../data/DataSetWhatsApp1.csv")

@st.cache_resource
def load_model():
    model = joblib.load("../naive_bayes_model.pkl")
    tfidf = joblib.load("../tfidf.pkl")
    return model, tfidf

df = load_data()
model, tfidf = load_model()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-circle">💬</div>
        <div style="font-weight:800; font-size:1.05rem; color:#3D3D56; margin-top:10px;">WhatsApp Analytics</div>
        <div style="font-size:0.74rem; color:#8A86A8; margin-top:2px;">Sentimen Analysis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["🏠  Dashboard", "☁️  WordCloud", "🎯  Evaluasi Model", "🤖  Prediksi", "📄  Dataset", "📚  Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="margin-top:24px; padding: 18px 14px; border-radius:16px; background:#EAF6F0; text-align:center;">
        <div style="font-size:0.72rem; color:#5FA98A; font-weight:600;">
            🧠 Naive Bayes · TF-IDF<br>30K+ ulasan WhatsApp
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SHARED DATA
# ─────────────────────────────────────────────
total      = len(df)
positif    = int((df['sentimen'] == 1).sum())
negatif    = int((df['sentimen'] == 0).sum())
pct_pos    = positif / total * 100
pct_neg    = negatif / total * 100

CM = np.array([[1377, 208], [324, 4091]])

def page_header(emoji, title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <div class="ph-logo">{emoji}</div>
        <div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════
if "Dashboard" in menu:

    page_header("💬", "Dashboard Analisis Sentimen",
                 "Gambaran umum distribusi sentimen ulasan pengguna aplikasi WhatsApp")

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "📊", f"{total:,}", "Total Ulasan", PASTEL_BLUE, "#E9F3FC"),
        (c2, "😊", f"{positif:,}", "Sentimen Positif", PASTEL_GREEN, "#E3F8EF"),
        (c3, "😞", f"{negatif:,}", "Sentimen Negatif", PASTEL_PINK, "#FFEAEF"),
        (c4, "🎯", "91.13%", "Akurasi Model", PASTEL_PURPLE, "#F2EBFB"),
    ]

    for col, icon, value, label, color, color_bg in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}; --accent-bg:{color_bg}">
                <div class="metric-icon-wrap">{icon}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Distribusi Sentimen</div>', unsafe_allow_html=True)

    col_chart, col_info = st.columns([3, 2], gap="large")

    with col_chart:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Bar chart
        ax = axes[0]
        bars = ax.bar(
            ["Positif", "Negatif"],
            [positif, negatif],
            color=[PASTEL_GREEN, PASTEL_PINK],
            width=0.45,
            edgecolor="none",
            zorder=3
        )
        ax.set_ylabel("Jumlah Ulasan", fontsize=9)
        ax.set_title("Distribusi Kelas", fontsize=10, fontweight='700', pad=12)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"{int(bar.get_height()):,}",
                ha='center', va='bottom', fontsize=9, color='#3D3D56', fontweight='700'
            )

        # Pie chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            [positif, negatif],
            labels=["Positif", "Negatif"],
            colors=[PASTEL_GREEN, PASTEL_PINK],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="#FFFFFF", linewidth=3),
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_color("#FFFFFF")
            at.set_fontsize(9)
            at.set_fontweight('700')
        for t in texts:
            t.set_color("#3D3D56")
            t.set_fontsize(9)
        ax2.set_title("Proporsi Kelas", fontsize=10, fontweight='700', pad=12)

        fig.tight_layout(pad=2)
        st.pyplot(fig)

    with col_info:
        st.markdown("""
        <div class="insight-box">
            <strong>📌 Komposisi Dataset</strong><br>
            Dataset berisi ulasan pengguna WhatsApp dari Google Play Store yang sudah dilabeli secara manual ke dalam dua kelas sentimen.
        </div>
        """, unsafe_allow_html=True)

        pos_pct = f"{pct_pos:.1f}%"
        neg_pct = f"{pct_neg:.1f}%"

        st.markdown(f"""
        <div class="insight-box">
            <strong>😊 Positif — {pos_pct}</strong><br>
            Sebagian besar pengguna memberikan ulasan positif, menunjukkan tingkat kepuasan yang tinggi terhadap aplikasi WhatsApp.
        </div>
        <div class="insight-box" style="border-left-color:{PASTEL_PINK}">
            <strong>😞 Negatif — {neg_pct}</strong><br>
            Ulasan negatif umumnya menyoroti masalah koneksi, update aplikasi, notifikasi, dan bug antarmuka.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: WORDCLOUD
# ══════════════════════════════════════════════
elif "WordCloud" in menu:

    page_header("☁️", "Word Cloud Visualisasi",
                 "Kata-kata yang paling sering muncul dalam ulasan positif dan negatif")

    tab1, tab2 = st.tabs(["😊  Ulasan Positif", "😞  Ulasan Negatif"])

    def make_wordcloud(text, colormap, bg="#FFFFFF"):
        wc = WordCloud(
            width=1000,
            height=420,
            background_color=bg,
            colormap=colormap,
            max_words=120,
            prefer_horizontal=0.85,
            collocations=False,
            margin=8
        ).generate(text)
        return wc

    with tab1:
        positive_text = " ".join(df[df['sentimen'] == 1]['content'].dropna())
        wc = make_wordcloud(positive_text, "summer")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        words = positive_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">🌿 Top 10 Kata — Positif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color=PASTEL_GREEN, edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=9)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

    with tab2:
        negative_text = " ".join(df[df['sentimen'] == 0]['content'].dropna())
        wc = make_wordcloud(negative_text, "RdPu")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#FFFFFF")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        words = negative_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">🌸 Top 10 Kata — Negatif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color=PASTEL_PINK, edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=9)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

# ══════════════════════════════════════════════
#  PAGE: EVALUASI MODEL
# ══════════════════════════════════════════════
elif "Evaluasi" in menu:

    page_header("🎯", "Evaluasi Model", "Performa model Naive Bayes pada data uji")

    TP, FN = CM[1][1], CM[1][0]
    FP, TN = CM[0][1], CM[0][0]

    accuracy  = (TP + TN) / CM.sum()
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = 2 * precision * recall / (precision + recall)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, color, color_bg in [
        (m1, "Accuracy",  f"{accuracy*100:.2f}%",  PASTEL_BLUE, "#E9F3FC"),
        (m2, "Precision", f"{precision*100:.2f}%", PASTEL_GREEN, "#E3F8EF"),
        (m3, "Recall",    f"{recall*100:.2f}%",    "#F2B65B", "#FFF3E0"),
        (m4, "F1-Score",  f"{f1*100:.2f}%",        PASTEL_PURPLE, "#F2EBFB"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}; --accent-bg:{color_bg}">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧩 Confusion Matrix</div>', unsafe_allow_html=True)

    col_cm, col_report = st.columns([1, 1], gap="large")

    with col_cm:
        pastel_cmap = sns.light_palette("#5FD1A0", as_cmap=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            CM,
            annot=True,
            fmt='d',
            cmap=pastel_cmap,
            linewidths=2,
            linecolor="#FFFFFF",
            cbar=False,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold", "color": "#3D3D56"}
        )
        ax.set_xlabel("Predicted Label", fontsize=10, labelpad=8)
        ax.set_ylabel("Actual Label", fontsize=10, labelpad=8)
        ax.set_xticklabels(["Negatif", "Positif"], fontsize=9)
        ax.set_yticklabels(["Negatif", "Positif"], fontsize=9, rotation=0)
        fig.tight_layout()
        st.pyplot(fig)

    with col_report:
        st.markdown(f"""
        <div class="insight-box">
            <strong>True Positive (TP)</strong> — 4,091<br>
            Ulasan positif yang diprediksi benar sebagai positif.
        </div>
        <div class="insight-box">
            <strong>True Negative (TN)</strong> — 1,377<br>
            Ulasan negatif yang diprediksi benar sebagai negatif.
        </div>
        <div class="insight-box" style="border-left-color:{PASTEL_PINK}">
            <strong>False Positive (FP)</strong> — 208<br>
            Ulasan negatif yang keliru diprediksi sebagai positif.
        </div>
        <div class="insight-box" style="border-left-color:{PASTEL_PINK}">
            <strong>False Negative (FN)</strong> — 324<br>
            Ulasan positif yang keliru diprediksi sebagai negatif.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📋 Laporan Per Kelas</div>', unsafe_allow_html=True)

    report_data = {
        "Kelas":     ["Negatif (0)", "Positif (1)"],
        "Precision": [f"{TN/(TN+FN)*100:.1f}%", f"{precision*100:.1f}%"],
        "Recall":    [f"{TN/(TN+FP)*100:.1f}%",  f"{recall*100:.1f}%"],
        "F1-Score":  ["—", f"{f1*100:.1f}%"],
        "Support":   [TN+FP, TP+FN],
    }
    st.dataframe(pd.DataFrame(report_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  PAGE: PREDIKSI
# ══════════════════════════════════════════════
elif "Prediksi" in menu:

    page_header("🤖", "Prediksi Sentimen", "Masukkan ulasan untuk diklasifikasikan oleh model")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        review = st.text_area(
            "Tulis ulasan WhatsApp di sini…",
            placeholder="Contoh: Aplikasinya stabil, notifikasi cepat, dan mudah digunakan!",
            height=180
        )

        examples = [
            "Aplikasi cepat, notifikasi tepat waktu, sangat membantu",
            "Sering force close dan update terbaru bikin lemot",
            "Fitur stiker dan video call lancar tanpa lag",
        ]

        st.markdown("<div style='font-size:0.8rem; color:#8A86A8; margin:12px 0 6px 0;'>✨ Coba contoh ulasan:</div>", unsafe_allow_html=True)
        for ex in examples:
            if st.button(ex[:45] + "…" if len(ex) > 45 else ex, key=ex):
                review = ex

        predict_btn = st.button("🔍  Analisis Sentimen", use_container_width=True)

    with col_result:
        if predict_btn and review.strip():
            vector     = tfidf.transform([review])
            pred       = model.predict(vector)[0]
            prob       = model.predict_proba(vector)[0]
            confidence = max(prob) * 100

            if pred == 1:
                st.markdown(f"""
                <div class="pred-positive">
                    <div class="pred-emoji">😊</div>
                    <div class="pred-label-pos">Sentimen Positif</div>
                    <div style="color:#8A86A8; font-size:0.82rem; margin-top:6px;">
                        Model yakin ulasan ini mengandung sentimen positif
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-negative">
                    <div class="pred-emoji">😞</div>
                    <div class="pred-label-neg">Sentimen Negatif</div>
                    <div style="color:#8A86A8; font-size:0.82rem; margin-top:6px;">
                        Model yakin ulasan ini mengandung sentimen negatif
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:20px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span style="font-size:0.82rem; color:#8A86A8; font-weight:600;">Confidence Score</span>
                    <span style="font-size:0.9rem; color:#3D3D56; font-weight:800;">{confidence:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence / 100)

            st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 1.5))
            classes = ["Negatif", "Positif"]
            colors  = [PASTEL_PINK, PASTEL_GREEN]
            bars = ax.barh(classes, [prob[0]*100, prob[1]*100], color=colors, edgecolor="none", height=0.45)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilitas (%)", fontsize=8)
            ax.spines[['top','right','bottom']].set_visible(False)
            for bar, p in zip(bars, [prob[0]*100, prob[1]*100]):
                ax.text(p + 1, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", va='center', fontsize=8, color='#3D3D56')
            fig.tight_layout()
            st.pyplot(fig)

        elif predict_btn and not review.strip():
            st.warning("⚠️ Masukkan ulasan terlebih dahulu.")
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px; color:#8A86A8;">
                <div style="font-size:3rem; margin-bottom:12px;">💬</div>
                <div style="font-size:0.88rem;">Hasil prediksi akan muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: DATASET
# ══════════════════════════════════════════════
elif "Dataset" in menu:

    page_header("📄", "Dataset Explorer", "Jelajahi dan filter dataset ulasan WhatsApp")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search_term = st.text_input("🔍 Cari kata dalam ulasan", placeholder="Ketik kata kunci…")
    with c2:
        pilihan = st.selectbox("Filter Sentimen", ["Semua", "Positif 😊", "Negatif 😞"])
    with c3:
        n_rows = st.selectbox("Tampilkan", [50, 100, 200, 500, "Semua"], index=1)

    data = df.copy()
    if "Positif" in pilihan:
        data = data[data['sentimen'] == 1]
    elif "Negatif" in pilihan:
        data = data[data['sentimen'] == 0]

    if search_term:
        data = data[data['content'].str.contains(search_term, case=False, na=False)]

    if n_rows != "Semua":
        data = data.head(int(n_rows))

    st.markdown(f"""
    <div style="font-size:0.82rem; color:#8A86A8; margin-bottom:12px;">
        Menampilkan <strong style="color:#3D3D56">{len(data):,}</strong> baris
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        data.reset_index(drop=True),
        use_container_width=True,
        height=480
    )

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️  Download CSV",
        data=csv,
        file_name="whatsapp_sentiment_filtered.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════
#  PAGE: TENTANG
# ══════════════════════════════════════════════
elif "Tentang" in menu:

    page_header("📚", "Tentang Penelitian", "Informasi metodologi dan detail teknis proyek ini")

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown(f"""
        <div class="about-card">
            <h3>🎯 Tujuan Penelitian</h3>
            <p>
                Mengklasifikasikan sentimen ulasan pengguna aplikasi WhatsApp dari Google Play Store
                ke dalam dua kelas: <strong style="color:{PASTEL_GREEN}">positif</strong> dan
                <strong style="color:{PASTEL_PINK}">negatif</strong>, untuk membantu memahami
                persepsi pengguna secara otomatis menggunakan pendekatan machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="about-card">
            <h3>📊 Dataset</h3>
            <ul>
                <li>Sumber: Google Play Store</li>
                <li>Total: <strong style="color:#3D3D56">30.000 ulasan</strong></li>
                <li>Label: Positif & Negatif</li>
                <li>Metode labeling: Semi-otomatis + verifikasi manual</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown("""
        <div class="about-card">
            <h3>⚙️ Pipeline NLP</h3>
            <ul>
                <li>Preprocessing: case folding, stopword removal, stemming</li>
                <li>Ekstraksi fitur: <strong style="color:#3D3D56">TF-IDF Vectorizer</strong></li>
                <li>Classifier: <strong style="color:#3D3D56">Multinomial Naive Bayes</strong></li>
                <li>Validasi: train-test split 80:20</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="about-card">
            <h3>📈 Hasil Evaluasi</h3>
            <ul>
                <li>Accuracy: <strong style="color:{PASTEL_BLUE}">91.13%</strong></li>
                <li>Precision: <strong style="color:{PASTEL_GREEN}">95.2%</strong></li>
                <li>Recall: <strong style="color:#F2B65B">92.7%</strong></li>
                <li>F1-Score: <strong style="color:{PASTEL_PURPLE}">93.9%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#FFFFFF; border:1px solid {C_BORDER}; border-radius:18px; padding:20px 24px; text-align:center; box-shadow: 0 6px 18px rgba(140,120,200,0.08);">
        <span style="font-size:0.8rem; color:#8A86A8;">
            💬 Dibuat dengan ❤️ menggunakan Streamlit · Naive Bayes · TF-IDF
        </span>
    </div>
    """, unsafe_allow_html=True)