"""
app.py
------
UI only. Zero business logic lives here.
All analysis is delegated to predictor.py.
"""

import streamlit as st
from predictor import load_model, analyse

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background-color: #080A0F !important;
    font-family: 'Inter', sans-serif;
    color: #94A3B8;
}
.block-container {
    max-width: 580px !important;
    padding: 0 1.5rem 5rem !important;
    margin: 0 auto;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── Input ────────────────────────────────── */
.input-label {
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: #334155;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
[data-testid="stTextInput"] { margin-bottom: 0 !important; }
[data-testid="stTextInput"] label { display: none !important; }
[data-testid="stTextInput"] input {
    background: #0D111A !important;
    border: 1px solid #1A2332 !important;
    border-radius: 12px !important;
    color: #CBD5E1 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 0.9rem 1.1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    caret-color: #38BDF8;
}
[data-testid="stTextInput"] input::placeholder { color: #1E2D42 !important; }
[data-testid="stTextInput"] input:focus {
    border-color: #1E3D5C !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.06) !important;
    outline: none !important;
}

/* ── Button ───────────────────────────────── */
[data-testid="stButton"] > button {
    background: #0F172A !important;
    color: #38BDF8 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    width: 100% !important;
    padding: 0.75rem !important;
    margin-top: 0.6rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase;
}
[data-testid="stButton"] > button:hover {
    background: #0F2338 !important;
    border-color: #38BDF8 !important;
    color: #7DD3FC !important;
    box-shadow: 0 0 20px rgba(56,189,248,0.08) !important;
}

/* ── Footer ───────────────────────────────── */
.pg-footer {
    text-align: center !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #1A2332;
    letter-spacing: 0.5px;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()

model, scaler, ready = get_model()


# ─────────────────────────────────────────────
# RENDER HELPERS
# Use st.* components instead of raw HTML divs
# to avoid Streamlit's tag-stripping bugs
# ─────────────────────────────────────────────
def render_card(status: str, confidence: float, signals: list):

    cfg = {
        "trusted": {
            "icon":    "✓",
            "title":   "Trusted Domain",
            "badge":   "Verified organisation",
            "desc":    "This URL belongs to a well-known, verified organisation. No AI analysis needed.",
            "color":   "#7DD3FC",
            "bg":      "linear-gradient(160deg, #060C14, #040A10)",
            "border":  "#0A2040",
            "b_bg":    "#050D18",
            "b_color": "#38BDF8",
            "b_bdr":   "#0A2040",
        },
        "safe": {
            "icon":    "✓",
            "title":   "This URL looks safe",
            "badge":   f"{confidence:.0f}% confident",
            "desc":    "No phishing signals detected. This URL appears to follow normal, legitimate patterns.",
            "color":   "#6EE7B7",
            "bg":      "linear-gradient(160deg, #060F0B, #040D09)",
            "border":  "#0A2918",
            "b_bg":    "#040F08",
            "b_color": "#34D399",
            "b_bdr":   "#0A2918",
        },
        "phishing": {
            "icon":    "⚠",
            "title":   "This URL looks dangerous",
            "badge":   f"{confidence:.0f}% confident",
            "desc":    "Our model detected phishing signals in this link. Avoid entering any personal or financial information.",
            "color":   "#FCA5A5",
            "bg":      "linear-gradient(160deg, #110608, #0D0407)",
            "border":  "#2D0A0E",
            "b_bg":    "#1F0508",
            "b_color": "#F87171",
            "b_bdr":   "#3D0A10",
        },
    }[status]

    # Build signal pills
    if status == "trusted":
        pills = ""
    elif signals:
        pills = "".join(
            f'<span style="font-size:0.7rem;font-family:JetBrains Mono,monospace;'
            f'padding:4px 11px;border-radius:6px;white-space:nowrap;'
            f'background:#160407;color:#FDA4AF;border:1px solid #3D0A12;'
            f'margin:0.2rem;">{s}</span>'
            for s in signals
        )
    else:
        pills = (
            '<span style="font-size:0.7rem;font-family:JetBrains Mono,monospace;'
            'padding:4px 11px;border-radius:6px;'
            'background:#040C07;color:#6EE7B7;border:1px solid #0A2015;">'
            'No suspicious signals found</span>'
        )

    pills_row = (
        f'<p style="display:flex;flex-wrap:wrap;justify-content:center;'
        f'gap:0.4rem;margin-top:1.4rem;">{pills}</p>'
        if pills else ""
    )

    st.markdown(
        f"""
        <style>
          @keyframes rise {{
            from {{ opacity:0; transform:translateY(14px); }}
            to   {{ opacity:1; transform:translateY(0); }}
          }}
        </style>
        <p style="
            background:{cfg['bg']};
            border:1px solid {cfg['border']};
            border-radius:18px;
            padding:2.2rem 2rem;
            text-align:center;
            margin-top:2.5rem;
            animation:rise 0.4s cubic-bezier(0.16,1,0.3,1) both;
        ">
          <span style="font-size:2.6rem;display:block;margin-bottom:1rem;">{cfg['icon']}</span>
          <span style="display:block;font-size:1.15rem;font-weight:600;
                       letter-spacing:-0.2px;margin-bottom:0.5rem;
                       color:{cfg['color']};">{cfg['title']}</span>
          <span style="display:inline-block;font-family:JetBrains Mono,monospace;
                       font-size:0.7rem;padding:3px 12px;border-radius:20px;
                       margin-bottom:1.1rem;background:{cfg['b_bg']};
                       color:{cfg['b_color']};border:1px solid {cfg['b_bdr']};">{cfg['badge']}</span>
          <span style="display:block;font-size:0.82rem;color:#475569;
                       line-height:1.65;max-width:360px;
                       margin:0 auto;">{cfg['desc']}</span>
          {pills_row}
        </p>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# HERO  — use st.* to avoid HTML stripping
# ─────────────────────────────────────────────
st.markdown("<br><br><br>", unsafe_allow_html=True)

col = st.columns([1, 2, 1])[1]
with col:
    st.markdown(
        '<p style="text-align:center;font-size:2.2rem;'
        'filter:drop-shadow(0 0 18px rgba(56,189,248,0.2));'
        'margin-bottom:0.2rem;">🛡️</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align:center;font-size:1.5rem;font-weight:600;'
        'color:#E2E8F0;letter-spacing:-0.4px;margin:0;">Is this URL safe?</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align:center;font-size:0.78rem;color:#334155;'
        'font-family:JetBrains Mono,monospace;letter-spacing:0.8px;'
        'text-transform:uppercase;margin-top:0.3rem;">Powered by PhishGuard AI</p>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
if not ready:
    st.error("Model files missing. Run `train_final_model.py` first.", icon="⚠️")
    st.stop()

st.markdown('<p class="input-label">URL to check</p>', unsafe_allow_html=True)
url  = st.text_input("url", label_visibility="collapsed",
                     placeholder="https://example.ng/login",
                     key="url_field")
scan = st.button("Analyse", use_container_width=True)


# ─────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────
if scan:
    if not url.strip():
        st.warning("Paste a URL above and try again.")
    else:
        with st.spinner(""):
            result = analyse(url.strip(), model, scaler)
        render_card(result["status"], result["confidence"], result["signals"])


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    '<p class="pg-footer">PHISHGUARD · B.SC. THESIS · 2026 · 87.13% ACCURACY</p>',
    unsafe_allow_html=True
)