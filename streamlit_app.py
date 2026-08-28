import html
import unicodedata

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Decodr — Customer Intent Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# RESEARCH CONFIGURATION
# ============================================================

MODEL_ID = "MehakR/decodr-roberta-intent-classifier"

# Validation-selected operating threshold from the dissertation.
AUTO_THRESHOLD = 0.991

# Final sequence length used in the dissertation.
MAX_LENGTH = 32


INTENT_TO_CATEGORY = {
    "cancel_order": "ORDER",
    "change_order": "ORDER",
    "change_shipping_address": "SHIPPING",
    "check_cancellation_fee": "CANCEL",
    "check_invoice": "INVOICE",
    "check_payment_methods": "PAYMENT",
    "check_refund_policy": "REFUND",
    "complaint": "FEEDBACK",
    "contact_customer_service": "CONTACT",
    "contact_human_agent": "CONTACT",
    "create_account": "ACCOUNT",
    "delete_account": "ACCOUNT",
    "delivery_options": "DELIVERY",
    "delivery_period": "DELIVERY",
    "edit_account": "ACCOUNT",
    "get_invoice": "INVOICE",
    "get_refund": "REFUND",
    "newsletter_subscription": "SUBSCRIPTION",
    "payment_issue": "PAYMENT",
    "place_order": "ORDER",
    "recover_password": "ACCOUNT",
    "registration_problems": "ACCOUNT",
    "review": "FEEDBACK",
    "set_up_shipping_address": "SHIPPING",
    "switch_account": "ACCOUNT",
    "track_order": "ORDER",
    "track_refund": "REFUND",
}


EXAMPLE_QUERIES = {
    "Choose an example": "",
    "Cancel an order": "I placed an order yesterday but I need to cancel it.",
    "Payment problem": "My payment keeps failing when I try to complete the order.",
    "Track a refund": "Where is my refund? I still haven't received it.",
    "Change shipping address": "Can I change the delivery address for my order?",
    "Recover password": "I forgot my password and cannot access my account.",
    "Contact a human": "I need to speak to a human agent.",
    "Ambiguous account request": "I need to change something on my account but I am not sure what.",
}


# ============================================================
# SESSION STATE
# ============================================================

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.markdown("## ◆ Decodr")

    st.caption("Research prototype controls")

    st.divider()

    selected_example = st.selectbox(
        "Example query",
        list(EXAMPLE_QUERIES.keys()),
    )

    if st.button(
        "Load example",
        use_container_width=True,
    ):
        if selected_example != "Choose an example":
            st.session_state.query_text = EXAMPLE_QUERIES[selected_example]
            st.session_state.prediction_result = None
            st.rerun()

    st.divider()

    show_alternatives = st.toggle(
        "Show alternative intents",
        value=True,
    )

    show_probability_chart = st.toggle(
        "Show probability chart",
        value=True,
    )

    show_explanation = st.toggle(
        "Show decision explanation",
        value=True,
    )

    reduce_motion = st.toggle(
        "Reduce background motion",
        value=False,
    )

    st.divider()

    st.markdown("#### Research operating point")

    st.metric(
        "Routing threshold",
        "99.10%",
    )

    st.caption(
        "Selected on the validation set to achieve "
        "at least 99.9% selective accuracy."
    )

    st.divider()

    st.caption(
        "The threshold shown in this application is fixed to the "
        "validation-selected dissertation operating point."
    )


# ============================================================
# GLOBAL DESIGN SYSTEM
# ============================================================

st.markdown(
    """
<style>

:root {
    --bg: #060708;
    --surface: #0d0f12;
    --surface-2: #12151a;
    --surface-3: #171a20;
    --border: rgba(255,255,255,0.09);

    --text: #f7f7f8;
    --muted: #9499a3;
    --soft: #c9ccd2;

    --red: #ff4d57;
    --red-2: #ff646c;
    --red-soft: rgba(255,77,87,0.12);

    --green: #49d69c;
    --green-soft: rgba(73,214,156,0.10);

    --amber: #ffbf5b;
    --amber-soft: rgba(255,191,91,0.10);

    --blue: #6f8dff;
}

/* ---------------------------------------------------------
   APP
--------------------------------------------------------- */

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 8% 0%,
            rgba(255, 77, 87, 0.08),
            transparent 25%
        ),
        radial-gradient(
            circle at 100% 18%,
            rgba(111, 141, 255, 0.05),
            transparent 27%
        ),
        #060708;
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(6,7,8,0);
}

.block-container {
    max-width: 1180px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

[data-testid="stSidebar"] {
    background: #090a0c;
    border-right: 1px solid rgba(255,255,255,0.07);
}

[data-testid="stSidebar"] * {
    color: #e8e9eb;
}

/* ---------------------------------------------------------
   TEXT
--------------------------------------------------------- */

h1, h2, h3 {
    color: var(--text) !important;
}

p {
    color: var(--soft);
}

/* ---------------------------------------------------------
   TABS
--------------------------------------------------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(13,15,18,0.75);
    border: 1px solid var(--border);
    padding: 7px;
    border-radius: 16px;
    margin-top: 8px;
    margin-bottom: 30px;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    border-radius: 11px;
    padding: 0 22px;
    color: #979ca6;
    font-weight: 650;
    border: none;
}

.stTabs [aria-selected="true"] {
    color: white !important;
    background:
        linear-gradient(
            135deg,
            rgba(255,77,87,0.18),
            rgba(255,77,87,0.05)
        ) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* ---------------------------------------------------------
   INPUT
--------------------------------------------------------- */

.stTextArea textarea {
    background: #0e1013 !important;
    color: #f8f8f8 !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    font-size: 1.04rem !important;
    line-height: 1.65 !important;
    padding: 18px !important;
}

.stTextArea textarea:focus {
    border-color: rgba(255,77,87,0.65) !important;
    box-shadow:
        0 0 0 1px rgba(255,77,87,0.20),
        0 0 35px rgba(255,77,87,0.07) !important;
}

/* ---------------------------------------------------------
   BUTTON
--------------------------------------------------------- */

.stButton > button {
    min-height: 49px;
    border-radius: 13px !important;
    border: 1px solid rgba(255,77,87,0.65) !important;

    background:
        linear-gradient(
            135deg,
            #ff4d57,
            #e83e49
        ) !important;

    color: white !important;
    font-weight: 750 !important;

    box-shadow:
        0 8px 24px rgba(255,77,87,0.15);

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);

    box-shadow:
        0 12px 34px rgba(255,77,87,0.24);
}

/* ---------------------------------------------------------
   SECTIONS
--------------------------------------------------------- */

.kicker {
    color: #ff5a63;
    font-size: 0.76rem;
    font-weight: 750;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.section-title {
    color: #f6f6f7;
    font-size: 2rem;
    font-weight: 760;
    letter-spacing: -0.035em;
    margin-bottom: 7px;
}

.section-copy {
    color: #858b95;
    font-size: 1rem;
    line-height: 1.7;
    margin-bottom: 23px;
}

/* ---------------------------------------------------------
   METRIC CARDS
--------------------------------------------------------- */

.result-card {
    position: relative;

    background:
        linear-gradient(
            145deg,
            rgba(20,22,27,0.97),
            rgba(12,14,17,0.98)
        );

    border: 1px solid rgba(255,255,255,0.085);

    border-radius: 20px;

    padding: 22px 22px 20px;

    min-height: 172px;

    box-shadow:
        0 20px 55px rgba(0,0,0,0.22);

    overflow: hidden;
}

.result-card::before {
    content: "";

    position: absolute;

    width: 130px;
    height: 130px;

    right: -70px;
    top: -70px;

    border-radius: 100%;

    background:
        radial-gradient(
            circle,
            rgba(255,77,87,0.18),
            transparent 70%
        );
}

.result-icon {
    width: 35px;
    height: 35px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 11px;

    background: rgba(255,77,87,0.10);

    color: #ff626b;

    font-size: 0.95rem;

    margin-bottom: 22px;
}

.result-label {
    color: #7f858e;

    font-size: 0.72rem;

    font-weight: 750;

    letter-spacing: 0.12em;

    text-transform: uppercase;

    margin-bottom: 7px;
}

.result-value {
    color: #f5f5f6;

    font-size: 1.55rem;

    font-weight: 760;

    letter-spacing: -0.025em;

    line-height: 1.2;

    margin-bottom: 8px;
}

.result-subtext {
    color: #717780;

    font-size: 0.87rem;

    line-height: 1.45;
}

/* ---------------------------------------------------------
   CONFIDENCE BAR
--------------------------------------------------------- */

.confidence-wrap {
    background: #0d0f12;

    border: 1px solid var(--border);

    border-radius: 15px;

    padding: 16px 18px;

    margin-top: 16px;
}

.confidence-head {
    display: flex;

    justify-content: space-between;

    margin-bottom: 11px;

    color: #b4b8bf;

    font-size: 0.88rem;
}

.confidence-track {
    width: 100%;

    height: 7px;

    background: #202329;

    border-radius: 100px;

    overflow: hidden;
}

.confidence-fill {
    height: 100%;

    border-radius: 100px;

    background:
        linear-gradient(
            90deg,
            #ff424d,
            #ff7a81
        );

    box-shadow:
        0 0 18px rgba(255,77,87,0.35);
}

/* ---------------------------------------------------------
   DECISION
--------------------------------------------------------- */

.route-auto {
    background:
        linear-gradient(
            135deg,
            rgba(73,214,156,0.095),
            rgba(13,15,18,0.98) 44%
        );

    border: 1px solid rgba(73,214,156,0.22);

    border-radius: 20px;

    padding: 24px;

    margin-top: 12px;
}

.route-review {
    background:
        linear-gradient(
            135deg,
            rgba(255,191,91,0.10),
            rgba(13,15,18,0.98) 44%
        );

    border: 1px solid rgba(255,191,91,0.25);

    border-radius: 20px;

    padding: 24px;

    margin-top: 12px;
}

.route-badge-auto,
.route-badge-review {
    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 7px 11px;

    border-radius: 999px;

    font-size: 0.78rem;

    font-weight: 700;

    margin-bottom: 13px;
}

.route-badge-auto {
    color: #61e0ad;
    background: rgba(73,214,156,0.10);
}

.route-badge-review {
    color: #ffc86f;
    background: rgba(255,191,91,0.10);
}

.route-title {
    color: #f4f5f6;

    font-size: 1.35rem;

    font-weight: 760;

    letter-spacing: -0.02em;

    margin-bottom: 8px;
}

.route-copy {
    color: #959aa3;

    font-size: 0.96rem;

    line-height: 1.7;
}

/* ---------------------------------------------------------
   ALTERNATIVE CARDS
--------------------------------------------------------- */

.alt-card {
    background:
        linear-gradient(
            145deg,
            #101216,
            #0b0d10
        );

    border: 1px solid rgba(255,255,255,0.075);

    border-radius: 17px;

    padding: 18px;

    min-height: 137px;
}

.alt-position {
    color: #656b74;

    font-size: 0.7rem;

    font-weight: 700;

    letter-spacing: 0.12em;

    text-transform: uppercase;

    margin-bottom: 14px;
}

.alt-intent {
    color: #f1f2f3;

    font-size: 1.12rem;

    font-weight: 700;

    margin-bottom: 7px;
}

.alt-category {
    color: #6f747d;

    font-size: 0.83rem;

    margin-bottom: 14px;
}

.alt-confidence {
    color: #ff636c;

    font-weight: 700;

    font-size: 0.91rem;
}

/* ---------------------------------------------------------
   INFO
--------------------------------------------------------- */

.info-card {
    background:
        linear-gradient(
            145deg,
            rgba(17,19,23,0.96),
            rgba(10,12,14,0.98)
        );

    border: 1px solid rgba(255,255,255,0.075);

    border-radius: 18px;

    padding: 22px;

    height: 100%;
}

.info-label {
    color: #ff5d66;

    font-size: 0.72rem;

    font-weight: 700;

    letter-spacing: 0.12em;

    text-transform: uppercase;

    margin-bottom: 11px;
}

.info-title {
    color: #f2f3f4;

    font-size: 1.15rem;

    font-weight: 700;

    margin-bottom: 10px;
}

.info-copy {
    color: #888e97;

    font-size: 0.93rem;

    line-height: 1.7;
}

/* ---------------------------------------------------------
   NATIVE METRICS
--------------------------------------------------------- */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            #111318,
            #0c0e11
        );

    border: 1px solid rgba(255,255,255,0.075);

    border-radius: 17px;

    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #868b94;
}

[data-testid="stMetricValue"] {
    color: #f5f5f6;
}

/* ---------------------------------------------------------
   DATAFRAME
--------------------------------------------------------- */

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}

/* ---------------------------------------------------------
   EXPANDER
--------------------------------------------------------- */

[data-testid="stExpander"] {
    background: #0d0f12;

    border: 1px solid rgba(255,255,255,0.075);

    border-radius: 15px;
}

/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.decodr-footer {
    text-align: center;

    margin-top: 45px;

    color: #535861;

    font-size: 0.82rem;
}

.footer-dot {
    color: #ff4d57;
    padding: 0 8px;
}

/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 720px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .section-title {
        font-size: 1.65rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0 12px;
        font-size: 0.88rem;
    }

    .result-card {
        min-height: 150px;
    }
}

/* ---------------------------------------------------------
   SCROLLBAR
--------------------------------------------------------- */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #060708;
}

::-webkit-scrollbar-thumb {
    background: #292c31;
    border-radius: 8px;
}

::-webkit-scrollbar-thumb:hover {
    background: #3b3f45;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PRIVATE MODEL
# ============================================================

@st.cache_resource
def load_decodr():

    hf_token = st.secrets["HF_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=hf_token,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        token=hf_token,
    )

    model.eval()

    return tokenizer, model


try:
    tokenizer, model = load_decodr()

except Exception as error:

    st.error(
        "Decodr could not load the private classification model. "
        "Check that the HF_TOKEN secret is configured and has read "
        "access to the Hugging Face repository."
    )

    st.stop()


# ============================================================
# MODEL HELPERS
# ============================================================

def clean_query(text):

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    return " ".join(
        text.strip().split()
    )


def model_label(class_id):

    label = model.config.id2label.get(
        class_id
    )

    if label is None:

        label = model.config.id2label.get(
            str(class_id)
        )

    return str(label)


def readable_intent(intent):

    return intent.replace(
        "_",
        " "
    ).title()


def predict_intent(query):

    cleaned = clean_query(
        query
    )

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    top_probs, top_ids = torch.topk(
        probabilities,
        k=5,
    )

    predictions = []

    for probability, class_id in zip(
        top_probs,
        top_ids
    ):

        intent = model_label(
            class_id.item()
        )

        predictions.append(
            {
                "intent": intent,
                "category": INTENT_TO_CATEGORY.get(
                    intent,
                    "UNKNOWN",
                ),
                "confidence": probability.item(),
            }
        )

    primary = predictions[0]

    return {
        "intent": primary["intent"],
        "category": primary["category"],
        "confidence": primary["confidence"],
        "predictions": predictions,
    }


# ============================================================
# PLOT
# ============================================================

def probability_chart(result):

    top_predictions = result["predictions"]

    labels = [
        readable_intent(
            item["intent"]
        )
        for item in top_predictions
    ]

    probabilities = [
        item["confidence"]
        for item in top_predictions
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=probabilities[::-1],
            y=labels[::-1],
            orientation="h",
            text=[
                f"{value:.2%}"
                for value in probabilities[::-1]
            ],
            textposition="outside",
            cliponaxis=False,
            marker=dict(
                color=[
                    "#343840",
                    "#343840",
                    "#343840",
                    "#4b3438",
                    "#ff4d57",
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Confidence: %{x:.2%}"
                "<extra></extra>"
            ),
        )
    )

    max_probability = max(
        probabilities
    )

    fig.update_layout(
        height=330,
        margin=dict(
            l=10,
            r=70,
            t=20,
            b=15,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#9ca1aa",
            family="Inter, Arial, sans-serif",
        ),
        showlegend=False,
        xaxis=dict(
            range=[
                0,
                min(
                    1.08,
                    max_probability + 0.12,
                )
            ],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.045)",
            tickformat=".0%",
            zeroline=False,
            title=None,
        ),
        yaxis=dict(
            showgrid=False,
            title=None,
        ),
    )

    return fig


# ============================================================
# ANIMATED HERO
# ============================================================

motion_state = "paused" if reduce_motion else "running"

hero_html = f"""
<!DOCTYPE html>

<html>

<head>

<style>

* {{
    box-sizing: border-box;
}}

html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
    overflow: hidden;
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}}

.hero {{
    position: relative;
    width: 100%;
    min-height: 360px;
    overflow: hidden;
    border-radius: 26px;
    background:
        radial-gradient(
            circle at 50% -20%,
            rgba(255,77,87,0.12),
            transparent 42%
        ),
        radial-gradient(
            circle at 85% 50%,
            rgba(98,112,255,0.06),
            transparent 37%
        ),
        linear-gradient(
            145deg,
            #111318,
            #08090b 65%
        );
    border: 1px solid rgba(255,255,255,0.09);
}}

.hero:before {{
    content: "";
    position: absolute;
    inset: 0;

    background:
        linear-gradient(
            120deg,
            transparent 15%,
            rgba(255,255,255,0.018) 50%,
            transparent 75%
        );

    pointer-events: none;
}}

#particles {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
}}

.waves {{
    position: absolute;
    inset: 0;

    width: 100%;
    height: 100%;

    opacity: 0.50;
}}

.wave {{
    fill: none;

    stroke-width: 0.72;

    stroke-linecap: round;

    animation:
        waveMove 10s ease-in-out infinite;

    animation-play-state: {motion_state};
}}

.wave.red {{
    stroke: rgba(255,77,87,0.34);
}}

.wave.white {{
    stroke: rgba(230,232,236,0.15);
}}

.wave.blue {{
    stroke: rgba(111,141,255,0.16);
}}

.wave:nth-child(2) {{
    animation-delay: -1.3s;
}}

.wave:nth-child(3) {{
    animation-delay: -2.5s;
}}

.wave:nth-child(4) {{
    animation-delay: -3.2s;
}}

.wave:nth-child(5) {{
    animation-delay: -4.1s;
}}

.wave:nth-child(6) {{
    animation-delay: -5.0s;
}}

.wave:nth-child(7) {{
    animation-delay: -5.7s;
}}

.wave:nth-child(8) {{
    animation-delay: -6.5s;
}}

.wave:nth-child(9) {{
    animation-delay: -7.3s;
}}

.wave:nth-child(10) {{
    animation-delay: -8.1s;
}}

.wave:nth-child(11) {{
    animation-delay: -8.8s;
}}

.wave:nth-child(12) {{
    animation-delay: -9.3s;
}}

@keyframes waveMove {{

    0%, 100% {{
        transform:
            translateX(-5px)
            translateY(0px);
    }}

    50% {{
        transform:
            translateX(12px)
            translateY(-7px);
    }}
}}

.hero-content {{
    position: relative;

    z-index: 5;

    min-height: 360px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    padding: 45px 30px;

    text-align: center;
}}

.brand-line {{
    display: flex;

    align-items: center;

    justify-content: center;

    gap: 18px;

    margin-bottom: 14px;
}}

.logo {{
    width: 75px;
    height: 75px;

    filter:
        drop-shadow(
            0 12px 24px
            rgba(255,77,87,0.15)
        );
}}

.logo-flow {{
    animation:
        logoPulse
        3.5s
        ease-in-out
        infinite;

    animation-play-state: {motion_state};
}}

@keyframes logoPulse {{

    0%, 100% {{
        opacity: 0.75;
    }}

    50% {{
        opacity: 1;
    }}
}}

.brand-name {{
    color: #f8f8f9;

    font-size: 64px;

    font-weight: 780;

    letter-spacing: -3.5px;

    line-height: 1;
}}

.tagline {{
    color: #f2f2f4;

    font-size: 22px;

    font-weight: 630;

    letter-spacing: -0.5px;

    margin-bottom: 12px;
}}

.description {{
    max-width: 670px;

    color: #8e949d;

    font-size: 15px;

    line-height: 1.65;
}}

.status {{
    margin-top: 22px;

    display: inline-flex;

    align-items: center;

    gap: 8px;

    border:
        1px solid
        rgba(255,255,255,0.08);

    background:
        rgba(255,255,255,0.035);

    color: #797f88;

    border-radius: 100px;

    padding: 8px 13px;

    font-size: 11px;

    letter-spacing: 0.04em;
}}

.dot {{
    width: 6px;
    height: 6px;

    border-radius: 100%;

    background: #ff4d57;

    box-shadow:
        0 0 11px
        rgba(255,77,87,0.8);
}}

@media(max-width: 650px) {{

    .hero {{
        min-height: 330px;
    }}

    .hero-content {{
        min-height: 330px;
        padding: 35px 20px;
    }}

    .brand-line {{
        gap: 11px;
    }}

    .logo {{
        width: 51px;
        height: 51px;
    }}

    .brand-name {{
        font-size: 45px;
        letter-spacing: -2.4px;
    }}

    .tagline {{
        font-size: 17px;
    }}

    .description {{
        font-size: 13px;
    }}
}}

</style>

</head>

<body>

<div class="hero">

<canvas id="particles"></canvas>

<svg
    class="waves"
    viewBox="0 0 1200 360"
    preserveAspectRatio="none"
>

<path
    class="wave red"
    d="M-100 280 C120 110 280 320 460 175 S720 60 900 190 S1120 270 1300 90"
/>

<path
    class="wave white"
    d="M-100 291 C120 121 280 331 460 186 S720 71 900 201 S1120 281 1300 101"
/>

<path
    class="wave red"
    d="M-100 302 C120 132 280 342 460 197 S720 82 900 212 S1120 292 1300 112"
/>

<path
    class="wave blue"
    d="M-100 313 C120 143 280 353 460 208 S720 93 900 223 S1120 303 1300 123"
/>

<path
    class="wave white"
    d="M-100 324 C120 154 280 364 460 219 S720 104 900 234 S1120 314 1300 134"
/>

<path
    class="wave red"
    d="M-100 335 C120 165 280 375 460 230 S720 115 900 245 S1120 325 1300 145"
/>

<path
    class="wave white"
    d="M-100 346 C120 176 280 386 460 241 S720 126 900 256 S1120 336 1300 156"
/>

<path
    class="wave red"
    d="M-100 269 C120 99 280 309 460 164 S720 49 900 179 S1120 259 1300 79"
/>

<path
    class="wave blue"
    d="M-100 258 C120 88 280 298 460 153 S720 38 900 168 S1120 248 1300 68"
/>

<path
    class="wave white"
    d="M-100 247 C120 77 280 287 460 142 S720 27 900 157 S1120 237 1300 57"
/>

<path
    class="wave red"
    d="M-100 236 C120 66 280 276 460 131 S720 16 900 146 S1120 226 1300 46"
/>

<path
    class="wave white"
    d="M-100 225 C120 55 280 265 460 120 S720 5 900 135 S1120 215 1300 35"
/>

</svg>


<div class="hero-content">

<div class="brand-line">

<svg
    class="logo"
    viewBox="0 0 120 120"
    xmlns="http://www.w3.org/2000/svg"
>

<defs>

<linearGradient
    id="dGradient"
    x1="0"
    y1="0"
    x2="1"
    y2="1"
>

<stop
    offset="0%"
    stop-color="#30343b"
/>

<stop
    offset="100%"
    stop-color="#17191e"
/>

</linearGradient>


<linearGradient
    id="flowGradient"
    x1="0"
    x2="1"
>

<stop
    offset="0%"
    stop-color="#ff343f"
/>

<stop
    offset="100%"
    stop-color="#ff737a"
/>

</linearGradient>

</defs>


<path
    d="
    M31 13
    H61
    C93 13
    108 31
    108 60
    C108 89
    93 107
    61 107
    H31
    Z
    "
    fill="url(#dGradient)"
    stroke="rgba(255,255,255,0.08)"
    stroke-width="1"
/>


<path
    d="
    M45 28
    H61
    C79 28
    90 40
    90 60
    C90 80
    79 92
    61 92
    H45
    Z
    "
    fill="#08090b"
/>


<g class="logo-flow">

<circle
    cx="15"
    cy="34"
    r="5"
    fill="#ff4d57"
/>

<circle
    cx="15"
    cy="60"
    r="5"
    fill="#a7b8ff"
/>

<circle
    cx="15"
    cy="86"
    r="5"
    fill="#ff4d57"
/>


<path
    d="
    M20 34
    C43 34
    42 59
    62 60
    "
    fill="none"
    stroke="url(#flowGradient)"
    stroke-width="5"
    stroke-linecap="round"
/>


<path
    d="
    M20 60
    C37 60
    44 60
    62 60
    "
    fill="none"
    stroke="#839cff"
    stroke-width="5"
    stroke-linecap="round"
/>


<path
    d="
    M20 86
    C43 86
    42 61
    62 60
    "
    fill="none"
    stroke="url(#flowGradient)"
    stroke-width="5"
    stroke-linecap="round"
/>


<circle
    cx="65"
    cy="60"
    r="6.5"
    fill="#ff4d57"
/>


<path
    d="
    M71 60
    H94
    "
    stroke="#ff4d57"
    stroke-width="5"
    stroke-linecap="round"
/>


<path
    d="
    M91 52
    L101 60
    L91 68
    "
    fill="none"
    stroke="#ff4d57"
    stroke-width="5"
    stroke-linecap="round"
    stroke-linejoin="round"
/>

</g>

</svg>


<div class="brand-name">
Decodr
</div>

</div>


<div class="tagline">
Decode customer intent. Route smarter.
</div>


<div class="description">
AI-assisted customer-service intent classification
with confidence-based automation and human escalation.
</div>


<div class="status">
<span class="dot"></span>
Fine-tuned RoBERTa · 27 intents · 11 business categories
</div>

</div>

</div>


<script>

const reduceMotion =
    {"true" if reduce_motion else "false"};

const canvas =
    document.getElementById("particles");

const ctx =
    canvas.getContext("2d");

let particles = [];

let width = 0;
let height = 0;

function resize() {{

    const rect =
        canvas.getBoundingClientRect();

    const ratio =
        window.devicePixelRatio || 1;

    width = rect.width;
    height = rect.height;

    canvas.width =
        width * ratio;

    canvas.height =
        height * ratio;

    ctx.setTransform(
        ratio,
        0,
        0,
        ratio,
        0,
        0
    );
}}

function createParticles() {{

    particles = [];

    const count =
        Math.max(
            28,
            Math.min(
                60,
                Math.floor(width / 18)
            )
        );

    for (
        let i = 0;
        i < count;
        i++
    ) {{

        particles.push({{

            x:
                Math.random() * width,

            y:
                Math.random() * height,

            vx:
                (
                    Math.random() - 0.5
                ) * 0.16,

            vy:
                (
                    Math.random() - 0.5
                ) * 0.12,

            size:
                Math.random() * 1.4
                + 0.35,

            alpha:
                Math.random() * 0.38
                + 0.07,

            red:
                Math.random() > 0.80

        }});
    }}
}}

function draw() {{

    ctx.clearRect(
        0,
        0,
        width,
        height
    );

    for (
        const p of particles
    ) {{

        if (!reduceMotion) {{

            p.x += p.vx;
            p.y += p.vy;

            if (p.x < -10)
                p.x = width + 10;

            if (p.x > width + 10)
                p.x = -10;

            if (p.y < -10)
                p.y = height + 10;

            if (p.y > height + 10)
                p.y = -10;
        }}

        ctx.beginPath();

        ctx.arc(
            p.x,
            p.y,
            p.size,
            0,
            Math.PI * 2
        );

        ctx.fillStyle =
            p.red
            ? `rgba(255,77,87,${{p.alpha}})`
            : `rgba(220,224,232,${{p.alpha}})`;

        ctx.fill();
    }}


    for (
        let i = 0;
        i < particles.length;
        i++
    ) {{

        for (
            let j = i + 1;
            j < particles.length;
            j++
        ) {{

            const a =
                particles[i];

            const b =
                particles[j];

            const dx =
                a.x - b.x;

            const dy =
                a.y - b.y;

            const distance =
                Math.sqrt(
                    dx * dx
                    + dy * dy
                );

            if (
                distance < 95
            ) {{

                const opacity =
                    (
                        1 -
                        distance / 95
                    ) * 0.045;

                ctx.beginPath();

                ctx.moveTo(
                    a.x,
                    a.y
                );

                ctx.lineTo(
                    b.x,
                    b.y
                );

                ctx.strokeStyle =
                    `rgba(220,224,232,${{opacity}})`;

                ctx.lineWidth =
                    0.5;

                ctx.stroke();
            }}
        }}
    }}

    requestAnimationFrame(
        draw
    );
}}


resize();
createParticles();
draw();


window.addEventListener(
    "resize",
    () => {{

        resize();
        createParticles();

    }}
);

</script>

</body>

</html>
"""

components.html(
    hero_html,
    height=375,
    scrolling=False,
)


# ============================================================
# MAIN NAVIGATION
# ============================================================

try_tab, insights_tab, about_tab = st.tabs(
    [
        "✦ Try Decodr",
        "Model Insights",
        "About",
    ]
)


# ============================================================
# TAB 1 — CLASSIFIER
# ============================================================

with try_tab:

    st.markdown(
        '<div class="kicker">Customer Query</div>'
        '<div class="section-title">'
        'What does the customer need?'
        '</div>'
        '<div class="section-copy">'
        'Enter a customer-service request in natural language. '
        'Decodr will classify the underlying intent and evaluate '
        'whether the prediction is sufficiently confident for '
        'automatic routing.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.text_area(
        "Customer query",
        key="query_text",
        height=145,
        label_visibility="collapsed",
        placeholder=(
            "Example: I placed an order yesterday "
            "but I need to cancel it."
        ),
    )

    analyze = st.button(
        "Analyze Intent  →",
        type="primary",
        use_container_width=True,
    )

    if analyze:

        if not st.session_state.query_text.strip():

            st.warning(
                "Enter a customer query first."
            )

        else:

            with st.spinner(
                "Decoding customer intent..."
            ):

                st.session_state.prediction_result = predict_intent(
                    st.session_state.query_text
                )

    result = st.session_state.prediction_result

    if result:

        safe_intent = html.escape(
            readable_intent(
                result["intent"]
            )
        )

        safe_category = html.escape(
            result["category"]
        )

        confidence = result["confidence"]

        confidence_percentage = (
            confidence * 100
        )

        st.write("")
        st.write("")

        st.markdown(
            '<div class="kicker">'
            'Classification Result'
            '</div>'
            '<div class="section-title">'
            'Decodr’s interpretation'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(
            3
        )

        with col1:

            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-icon">◆</div>'
                f'<div class="result-label">Predicted Intent</div>'
                f'<div class="result-value">{safe_intent}</div>'
                f'<div class="result-subtext">'
                f'Most likely customer need'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col2:

            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-icon">↳</div>'
                f'<div class="result-label">Business Category</div>'
                f'<div class="result-value">{safe_category}</div>'
                f'<div class="result-subtext">'
                f'Suggested operational workflow'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col3:

            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-icon">%</div>'
                f'<div class="result-label">Model Confidence</div>'
                f'<div class="result-value">{confidence:.2%}</div>'
                f'<div class="result-subtext">'
                f'Operating threshold: {AUTO_THRESHOLD:.2%}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="confidence-wrap">'
            f'<div class="confidence-head">'
            f'<span>Prediction confidence</span>'
            f'<span>{confidence:.2%}</span>'
            f'</div>'
            f'<div class="confidence-track">'
            f'<div class="confidence-fill" '
            f'style="width:{confidence_percentage:.2f}%">'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.write("")

        st.markdown(
            '<div class="kicker">'
            'Routing Decision'
            '</div>',
            unsafe_allow_html=True,
        )

        if confidence >= AUTO_THRESHOLD:

            st.markdown(
                f'<div class="route-auto">'
                f'<div class="route-badge-auto">'
                f'● AUTOMATION CANDIDATE'
                f'</div>'
                f'<div class="route-title">'
                f'Automatic routing recommended'
                f'</div>'
                f'<div class="route-copy">'
                f'The predicted intent is '
                f'<strong>{safe_intent}</strong> '
                f'with a confidence of '
                f'<strong>{confidence:.2%}</strong>. '
                f'This exceeds the validation-selected '
                f'<strong>{AUTO_THRESHOLD:.2%}</strong> operating '
                f'threshold, so the query is eligible to be routed '
                f'to the <strong>{safe_category}</strong> workflow.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f'<div class="route-review">'
                f'<div class="route-badge-review">'
                f'● REVIEW REQUIRED'
                f'</div>'
                f'<div class="route-title">'
                f'Human review recommended'
                f'</div>'
                f'<div class="route-copy">'
                f'The predicted intent is '
                f'<strong>{safe_intent}</strong>, '
                f'but the confidence of '
                f'<strong>{confidence:.2%}</strong> '
                f'is below the validation-selected '
                f'<strong>{AUTO_THRESHOLD:.2%}</strong> operating '
                f'threshold. The query should therefore be reviewed '
                f'before an operational routing decision is made.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if show_alternatives:

            st.write("")
            st.write("")

            st.markdown(
                '<div class="kicker">'
                'Alternative Interpretations'
                '</div>'
                '<div class="section-title" '
                'style="font-size:1.55rem;">'
                'Other possibilities considered'
                '</div>'
                '<div class="section-copy">'
                'The next highest-probability intent predictions '
                'provide context around model uncertainty.'
                '</div>',
                unsafe_allow_html=True,
            )

            alt1 = result["predictions"][1]
            alt2 = result["predictions"][2]

            alternative_col1, alternative_col2 = st.columns(
                2
            )

            with alternative_col1:

                st.markdown(
                    f'<div class="alt-card">'
                    f'<div class="alt-position">'
                    f'Second most likely'
                    f'</div>'
                    f'<div class="alt-intent">'
                    f'{html.escape(readable_intent(alt1["intent"]))}'
                    f'</div>'
                    f'<div class="alt-category">'
                    f'{html.escape(alt1["category"])}'
                    f'</div>'
                    f'<div class="alt-confidence">'
                    f'{alt1["confidence"]:.2%} confidence'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with alternative_col2:

                st.markdown(
                    f'<div class="alt-card">'
                    f'<div class="alt-position">'
                    f'Third most likely'
                    f'</div>'
                    f'<div class="alt-intent">'
                    f'{html.escape(readable_intent(alt2["intent"]))}'
                    f'</div>'
                    f'<div class="alt-category">'
                    f'{html.escape(alt2["category"])}'
                    f'</div>'
                    f'<div class="alt-confidence">'
                    f'{alt2["confidence"]:.2%} confidence'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if show_probability_chart:

            st.write("")

            st.plotly_chart(
                probability_chart(
                    result
                ),
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
            )

        if show_explanation:

            st.write("")

            with st.expander(
                "How does Decodr make this decision?"
            ):

                st.markdown(
                    f"""
Decodr uses the final fine-tuned **RoBERTa sequence-classification model**
evaluated in the dissertation.

The submitted customer query is tokenised and classified into one of
**27 customer-service intents**. The class receiving the highest predicted
probability becomes the primary intent prediction.

The confidence of that prediction is then compared with the
validation-selected operating threshold of **{AUTO_THRESHOLD:.2%}**.

**Confidence ≥ {AUTO_THRESHOLD:.2%}**  
The prediction is treated as a candidate for automatic routing.

**Confidence < {AUTO_THRESHOLD:.2%}**  
Human review is recommended before routing.

The broader business category is mapped **after** intent classification.
It is not supplied to RoBERTa as an input.
"""
                )


# ============================================================
# TAB 2 — MODEL INSIGHTS
# ============================================================

with insights_tab:

    st.markdown(
        '<div class="kicker">'
        'Held-Out Evaluation'
        '</div>'
        '<div class="section-title">'
        'Model performance'
        '</div>'
        '<div class="section-copy">'
        'Comparative results from the 3,645-query held-out test set. '
        'The final test partition was not used during model selection.'
        '</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3, metric4 = st.columns(
        4
    )

    with metric1:
        st.metric(
            "RoBERTa Macro-F1",
            "0.99898",
        )

    with metric2:
        st.metric(
            "Correct predictions",
            "3,642 / 3,645",
        )

    with metric3:
        st.metric(
            "Total errors",
            "3",
        )

    with metric4:
        st.metric(
            "High-risk errors",
            "1",
        )

    st.write("")
    st.write("")

    model_results = pd.DataFrame(
        {
            "Model": [
                "RoBERTa",
                "DistilBERT",
                "Linear SVM",
                "Logistic Regression",
                "Multinomial Naive Bayes",
            ],
            "Macro-F1": [
                0.998982,
                0.997925,
                0.994143,
                0.992812,
                0.992262,
            ],
            "Test Errors": [
                3,
                7,
                20,
                25,
                26,
            ],
            "High-Risk Errors": [
                1,
                4,
                14,
                17,
                17,
            ],
        }
    )

    st.dataframe(
        model_results,
        hide_index=True,
        use_container_width=True,
    )

    st.write("")
    st.write("")

    st.markdown(
        '<div class="kicker">'
        'Confidence-Aware Automation'
        '</div>'
        '<div class="section-title" '
        'style="font-size:1.55rem;">'
        'Validation-selected operating point'
        '</div>',
        unsafe_allow_html=True,
    )

    threshold1, threshold2, threshold3, threshold4 = st.columns(
        4
    )

    with threshold1:
        st.metric(
            "Threshold",
            "99.10%",
        )

    with threshold2:
        st.metric(
            "Automation coverage",
            "99.70%",
        )

    with threshold3:
        st.metric(
            "Selective accuracy",
            "99.92%",
        )

    with threshold4:
        st.metric(
            "Human review",
            "11 / 3,612",
        )

    st.caption(
        "The confidence operating point was selected using validation-set "
        "predictions. It achieved 99.6955% coverage and 99.9167% selective "
        "accuracy, with 11 of 3,612 validation queries referred for review."
    )

    st.write("")
    st.write("")

    insight_col1, insight_col2, insight_col3 = st.columns(
        3
    )

    with insight_col1:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">Predictive performance</div>'
            '<div class="info-title">Small headline gain</div>'
            '<div class="info-copy">'
            'RoBERTa improved Macro-F1 over Linear SVM by '
            'approximately 0.484 percentage points. The aggregate '
            'difference is therefore modest.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with insight_col2:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">Operational risk</div>'
            '<div class="info-title">Fewer consequential errors</div>'
            '<div class="info-copy">'
            'The distinction becomes clearer when residual errors '
            'are examined. RoBERTa produced 3 test errors, including '
            '1 high-risk error, compared with 20 total and 14 '
            'high-risk errors for Linear SVM.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with insight_col3:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">Deployment trade-off</div>'
            '<div class="info-title">Accuracy is not the only criterion</div>'
            '<div class="info-copy">'
            'RoBERTa delivered the strongest predictive result but '
            'required substantially greater training time and model '
            'storage than the traditional baselines.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    efficiency = pd.DataFrame(
        {
            "Model": [
                "RoBERTa",
                "DistilBERT",
                "Linear SVM",
                "Logistic Regression",
                "Multinomial Naive Bayes",
            ],
            "Training Time (s)": [
                322.721,
                117.780,
                1.131,
                7.950,
                0.158,
            ],
            "Model Size (MB)": [
                478.99,
                256.18,
                1.06,
                1.06,
                2.02,
            ],
        }
    )

    st.markdown(
        '<div class="kicker">'
        'Computational Efficiency'
        '</div>'
        '<div class="section-title" '
        'style="font-size:1.55rem;">'
        'Performance comes with a cost'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        efficiency,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with about_tab:

    st.markdown(
        '<div class="kicker">'
        'Research Prototype'
        '</div>'
        '<div class="section-title">'
        'About Decodr'
        '</div>'
        '<div class="section-copy">'
        'A proof-of-concept interface translating the dissertation '
        'model evaluation into an interpretable customer-service '
        'routing workflow.'
        '</div>',
        unsafe_allow_html=True,
    )

    about1, about2 = st.columns(
        2
    )

    with about1:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">Purpose</div>'
            '<div class="info-title">'
            'From intent prediction to operational decision'
            '</div>'
            '<div class="info-copy">'
            'Decodr demonstrates how a customer query can be '
            'classified into a fine-grained intent and subsequently '
            'mapped to a broader business workflow. Confidence adds '
            'a decision layer that determines whether the prediction '
            'is suitable for automatic handling or should be referred '
            'for human review.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with about2:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">Model</div>'
            '<div class="info-title">'
            'Fine-tuned RoBERTa'
            '</div>'
            '<div class="info-copy">'
            'The deployed classifier is the final RoBERTa model '
            'evaluated in the dissertation. It predicts among '
            '27 customer-service intents. These are subsequently '
            'mapped to 11 broader operational categories.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    about3, about4 = st.columns(
        2
    )

    with about3:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">Human-in-the-loop</div>'
            '<div class="info-title">'
            'Selective automation'
            '</div>'
            '<div class="info-copy">'
            'The prototype does not assume every model prediction '
            'should trigger an automated action. Lower-confidence '
            'cases are deliberately referred for review, illustrating '
            'a hybrid automation approach rather than unconditional '
            'end-to-end automation.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with about4:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">Scope</div>'
            '<div class="info-title">'
            'Academic demonstration'
            '</div>'
            '<div class="info-copy">'
            'Decodr is an MSc Business Analytics research prototype, '
            'not a production customer-service system. Performance '
            'reported here reflects the dissertation dataset and '
            'should not be interpreted as guaranteed performance on '
            'live operational traffic.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    st.warning(
        "The underlying dataset is synthetic, English-language and "
        "closed-set. Real-world deployment would require evaluation "
        "on operational customer queries, monitoring for distribution "
        "shift and mechanisms for queries that fall outside the known "
        "intent taxonomy."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="decodr-footer">'
    'Decodr'
    '<span class="footer-dot">◆</span>'
    'MSc Business Analytics research prototype'
    '</div>',
    unsafe_allow_html=True,
)
