import html as html_lib
import unicodedata

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Decodr — Customer Intent Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "MehakR/decodr-roberta-intent-classifier"

AUTO_THRESHOLD = 0.991
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
    "Cancel an order":
        "I placed an order yesterday but I need to cancel it.",

    "Payment problem":
        "My payment keeps failing when I try to complete the order.",

    "Track a refund":
        "Where is my refund? I still haven't received it.",

    "Change shipping address":
        "Can I change the delivery address for my order?",

    "Recover password":
        "I forgot my password and cannot access my account.",

    "Contact a human":
        "I need to speak to a human agent.",

    "Ambiguous account request":
        "I need to change something on my account but I am not sure what.",
}


# ============================================================
# SESSION STATE
# ============================================================

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ◈ Decodr")
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

            st.session_state.query_text = (
                EXAMPLE_QUERIES[selected_example]
            )

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
        "Validation-selected threshold used for "
        "confidence-based routing."
    )


# ============================================================
# GLOBAL DESIGN
# ============================================================

st.markdown(
    """
<style>

:root {
    --purple: #6D4AFF;
    --purple-dark: #5135D5;
    --purple-mid: #8D74FF;
    --lavender: #EEE9FF;
    --lavender-soft: #F7F5FF;

    --bg: #F8F7FC;
    --card: #FFFFFF;

    --border: #E9E5F2;

    --text: #1E1B2B;
    --text-secondary: #615B6D;
    --muted: #8D8797;

    --green: #17966B;
    --green-bg: #EAFAF4;

    --amber: #B7791F;
    --amber-bg: #FFF8E8;
}


/* =========================================================
   APP
========================================================= */

html,
body,
[class*="css"] {
    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


[data-testid="stAppViewContainer"] {

    background:

        radial-gradient(
            circle at 7% 0%,
            rgba(109,74,255,0.10),
            transparent 25%
        ),

        radial-gradient(
            circle at 95% 10%,
            rgba(175,157,255,0.12),
            transparent 27%
        ),

        linear-gradient(
            180deg,
            #FCFBFF 0%,
            #F8F7FC 100%
        );

    color: var(--text);
}


[data-testid="stHeader"] {
    background: transparent;
}


.block-container {
    max-width: 1160px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}


/* =========================================================
   SIDEBAR
========================================================= */

[data-testid="stSidebar"] {

    background:

        linear-gradient(
            180deg,
            #FCFBFF,
            #F4F1FC
        );

    border-right:
        1px solid #E8E3F1;
}


[data-testid="stSidebar"] * {
    color: #2E2940;
}


/* =========================================================
   TABS — CLEAN MINIMAL STYLE
========================================================= */

.stTabs [data-baseweb="tab-list"] {

    gap: 28px;

    background: transparent;

    border: none;

    border-bottom:
        1px solid #E7E3EF;

    padding: 0;

    margin-top: 10px;
    margin-bottom: 30px;

    box-shadow: none;
}


.stTabs [data-baseweb="tab"] {

    height: 52px;

    padding: 0 2px;

    background:
        transparent !important;

    border:
        none !important;

    border-radius:
        0 !important;

    color:
        #4F4A59;

    font-weight:
        600;

    box-shadow:
        none !important;
}


/* ACTIVE TAB */

.stTabs button[role="tab"][aria-selected="true"] {

    color:
        #6D4AFF !important;

    background:
        transparent !important;

    border:
        none !important;

    border-bottom:
        3px solid #6D4AFF !important;

    box-shadow:
        none !important;
}


/* REMOVE DEFAULT STREAMLIT HIGHLIGHT */

.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}


/* REMOVE CLICK / FOCUS BOX */

.stTabs button[role="tab"]:focus,
.stTabs button[role="tab"]:focus-visible {

    outline:
        none !important;

    box-shadow:
        none !important;
}


/* =========================================================
   SECTION HEADINGS
========================================================= */

.kicker {

    color:
        #6D4AFF;

    font-size:
        0.74rem;

    font-weight:
        760;

    letter-spacing:
        0.16em;

    text-transform:
        uppercase;

    margin-bottom:
        7px;
}


.section-title {

    color:
        #1E1B2B;

    font-size:
        2rem;

    font-weight:
        780;

    letter-spacing:
        -0.035em;

    margin-bottom:
        7px;
}


.section-copy {

    color:
        #777181;

    font-size:
        1rem;

    line-height:
        1.7;

    margin-bottom:
        23px;
}


/* =========================================================
   INPUT
========================================================= */

.stTextArea textarea {

    background:
        rgba(255,255,255,0.96) !important;

    color:
        #24202F !important;

    border:
        1px solid #E3DFEB !important;

    border-radius:
        17px !important;

    font-size:
        1.04rem !important;

    line-height:
        1.65 !important;

    padding:
        18px !important;

    box-shadow:
        0 8px 28px
        rgba(50,35,100,0.035);
}


.stTextArea textarea:focus {

    border-color:
        #8D74FF !important;

    box-shadow:

        0 0 0 3px
        rgba(109,74,255,0.08),

        0 12px 35px
        rgba(67,46,150,0.07) !important;
}


/* =========================================================
   BUTTON
========================================================= */

.stButton > button {

    min-height:
        50px;

    border-radius:
        13px !important;

    border:
        1px solid #6547E6 !important;

    background:

        linear-gradient(
            135deg,
            #7658FF,
            #5B3CDD
        ) !important;

    color:
        white !important;

    font-weight:
        750 !important;

    box-shadow:
        0 8px 25px
        rgba(109,74,255,0.18);

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease;
}


.stButton > button:hover {

    transform:
        translateY(-1px);

    box-shadow:
        0 13px 34px
        rgba(109,74,255,0.27);
}


/* =========================================================
   RESULT CARDS
========================================================= */

.result-card {

    position:
        relative;

    min-height:
        174px;

    padding:
        22px;

    overflow:
        hidden;

    background:

        linear-gradient(
            145deg,
            #FFFFFF,
            #FBFAFF
        );

    border:
        1px solid #E8E4F0;

    border-radius:
        21px;

    box-shadow:
        0 14px 40px
        rgba(49,33,103,0.055);
}


.result-card::after {

    content: "";

    position:
        absolute;

    width:
        140px;

    height:
        140px;

    right:
        -75px;

    top:
        -75px;

    border-radius:
        100%;

    background:

        radial-gradient(
            circle,
            rgba(109,74,255,0.11),
            transparent 70%
        );
}


.result-icon {

    width:
        38px;

    height:
        38px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        12px;

    background:
        #EFEAFF;

    color:
        #6243E2;

    font-size:
        1rem;

    font-weight:
        750;

    margin-bottom:
        21px;
}


.result-label {

    color:
        #918B9A;

    font-size:
        0.70rem;

    font-weight:
        750;

    letter-spacing:
        0.12em;

    text-transform:
        uppercase;

    margin-bottom:
        7px;
}


.result-value {

    color:
        #221E30;

    font-size:
        1.52rem;

    font-weight:
        780;

    letter-spacing:
        -0.025em;

    line-height:
        1.2;

    margin-bottom:
        8px;
}


.result-subtext {

    color:
        #928C9C;

    font-size:
        0.86rem;

    line-height:
        1.45;
}


/* =========================================================
   CONFIDENCE BAR
========================================================= */

.confidence-wrap {

    margin-top:
        16px;

    padding:
        17px 18px;

    background:
        #FFFFFF;

    border:
        1px solid #E8E4F0;

    border-radius:
        16px;

    box-shadow:
        0 8px 25px
        rgba(49,33,103,0.035);
}


.confidence-head {

    display:
        flex;

    justify-content:
        space-between;

    color:
        #716B7C;

    font-size:
        0.88rem;

    margin-bottom:
        11px;
}


.confidence-track {

    width:
        100%;

    height:
        8px;

    overflow:
        hidden;

    background:
        #EEEAF5;

    border-radius:
        999px;
}


.confidence-fill {

    height:
        100%;

    border-radius:
        999px;

    background:

        linear-gradient(
            90deg,
            #5E3FE1,
            #A28DFF
        );

    box-shadow:
        0 0 16px
        rgba(109,74,255,0.24);
}


/* =========================================================
   ROUTING
========================================================= */

.route-auto {

    margin-top:
        12px;

    padding:
        24px;

    background:

        linear-gradient(
            135deg,
            #EAFBF5,
            #FFFFFF 58%
        );

    border:
        1px solid #CDEFE2;

    border-radius:
        20px;

    box-shadow:
        0 10px 30px
        rgba(30,110,82,0.04);
}


.route-review {

    margin-top:
        12px;

    padding:
        24px;

    background:

        linear-gradient(
            135deg,
            #FFF8E8,
            #FFFFFF 58%
        );

    border:
        1px solid #F2DFC1;

    border-radius:
        20px;
}


.route-badge-auto,
.route-badge-review {

    display:
        inline-flex;

    align-items:
        center;

    padding:
        7px 11px;

    border-radius:
        999px;

    font-size:
        0.74rem;

    font-weight:
        740;

    letter-spacing:
        0.04em;

    margin-bottom:
        13px;
}


.route-badge-auto {

    color:
        #158A63;

    background:
        #DFF7EE;
}


.route-badge-review {

    color:
        #A97018;

    background:
        #FFF0D2;
}


.route-title {

    color:
        #221E30;

    font-size:
        1.35rem;

    font-weight:
        770;

    letter-spacing:
        -0.02em;

    margin-bottom:
        8px;
}


.route-copy {

    color:
        #65606D;

    font-size:
        0.96rem;

    line-height:
        1.72;
}


/* =========================================================
   ALTERNATIVE CARDS
========================================================= */

.alt-card {

    min-height:
        140px;

    padding:
        19px;

    background:

        linear-gradient(
            145deg,
            #FFFFFF,
            #FBFAFF
        );

    border:
        1px solid #E8E4F0;

    border-radius:
        18px;

    box-shadow:
        0 10px 28px
        rgba(49,33,103,0.04);
}


.alt-position {

    color:
        #A09AA8;

    font-size:
        0.69rem;

    font-weight:
        730;

    letter-spacing:
        0.12em;

    text-transform:
        uppercase;

    margin-bottom:
        14px;
}


.alt-intent {

    color:
        #24202F;

    font-size:
        1.14rem;

    font-weight:
        740;

    margin-bottom:
        6px;
}


.alt-category {

    color:
        #96909F;

    font-size:
        0.82rem;

    margin-bottom:
        14px;
}


.alt-confidence {

    color:
        #6344DE;

    font-size:
        0.91rem;

    font-weight:
        730;
}


/* =========================================================
   INFORMATION CARDS
========================================================= */

.info-card {

    height:
        100%;

    padding:
        22px;

    background:

        linear-gradient(
            145deg,
            #FFFFFF,
            #FBFAFF
        );

    border:
        1px solid #E8E4F0;

    border-radius:
        19px;

    box-shadow:
        0 10px 30px
        rgba(49,33,103,0.04);
}


.info-label {

    color:
        #6D4AFF;

    font-size:
        0.70rem;

    font-weight:
        740;

    letter-spacing:
        0.12em;

    text-transform:
        uppercase;

    margin-bottom:
        11px;
}


.info-title {

    color:
        #252130;

    font-size:
        1.14rem;

    font-weight:
        740;

    margin-bottom:
        10px;
}


.info-copy {

    color:
        #777181;

    font-size:
        0.93rem;

    line-height:
        1.72;
}


/* =========================================================
   STREAMLIT METRICS — UPDATED TO PREVENT TRUNCATION
========================================================= */

[data-testid="stMetric"] {

    padding:
        18px;

    min-height:
        145px;

    background:

        linear-gradient(
            145deg,
            #FFFFFF,
            #FBFAFF
        );

    border:
        1px solid #E8E4F0;

    border-radius:
        17px;

    box-shadow:
        0 9px 26px
        rgba(49,33,103,0.04);

    overflow:
        visible !important;
}


[data-testid="stMetricLabel"] {

    color:
        #7D7787;

    white-space:
        normal !important;
}


[data-testid="stMetricValue"] {

    color:
        #241F31;

    overflow:
        visible !important;
}


[data-testid="stMetricValue"] > div {

    font-size:
        1.85rem !important;

    line-height:
        1.2 !important;

    white-space:
        nowrap !important;

    overflow:
        visible !important;

    text-overflow:
        clip !important;
}


/* =========================================================
   DATAFRAME
========================================================= */

[data-testid="stDataFrame"] {

    overflow:
        hidden;

    border:
        1px solid #E8E4F0;

    border-radius:
        16px;
}


/* =========================================================
   EXPANDER
========================================================= */

[data-testid="stExpander"] {

    background:
        #FFFFFF;

    border:
        1px solid #E7E3EF;

    border-radius:
        16px;
}


/* =========================================================
   ALERTS
========================================================= */

[data-testid="stAlert"] {
    border-radius: 16px;
}


/* =========================================================
   FOOTER
========================================================= */

.decodr-footer {

    margin-top:
        45px;

    text-align:
        center;

    color:
        #9992A1;

    font-size:
        0.82rem;
}


.footer-dot {

    color:
        #6D4AFF;

    padding:
        0 8px;
}


/* =========================================================
   MOBILE
========================================================= */

@media(max-width:720px) {

    .block-container {

        padding-left:
            1rem;

        padding-right:
            1rem;
    }


    .section-title {
        font-size: 1.65rem;
    }


    .stTabs [data-baseweb="tab-list"] {
        gap: 17px;
    }


    .stTabs [data-baseweb="tab"] {

        padding:
            0 1px;

        font-size:
            0.86rem;
    }


    .result-card {
        min-height: 150px;
    }


    [data-testid="stMetricValue"] > div {
        font-size: 1.55rem !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD PRIVATE MODEL
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

except Exception:

    st.error(
        "Decodr could not load the private classification model. "
        "Please check the Hugging Face access configuration."
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


def get_model_label(class_id):

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

    cleaned_query = clean_query(
        query
    )

    inputs = tokenizer(
        cleaned_query,
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

        intent = get_model_label(
            class_id.item()
        )

        predictions.append(
            {
                "intent":
                    intent,

                "category":
                    INTENT_TO_CATEGORY.get(
                        intent,
                        "UNKNOWN",
                    ),

                "confidence":
                    float(
                        probability.item()
                    ),
            }
        )

    primary = predictions[0]

    return {
        "intent":
            primary["intent"],

        "category":
            primary["category"],

        "confidence":
            primary["confidence"],

        "predictions":
            predictions,
    }


# ============================================================
# PROBABILITY CHART
# ============================================================

def probability_chart(result):

    predictions = result["predictions"]

    labels = [
        readable_intent(
            prediction["intent"]
        )
        for prediction in predictions
    ]

    values = [
        prediction["confidence"]
        for prediction in predictions
    ]

    colors = [
        "#DCD6F1",
        "#CEC5F0",
        "#B7A8EF",
        "#9983F3",
        "#6D4AFF",
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(

            x=
                values[::-1],

            y=
                labels[::-1],

            orientation=
                "h",

            text=[
                f"{value:.2%}"
                for value in values[::-1]
            ],

            textposition=
                "outside",

            marker=dict(
                color=colors
            ),

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Confidence: %{x:.2%}"
                "<extra></extra>"
            ),
        )
    )

    max_value = max(
        values
    )

    fig.update_layout(

        height=
            325,

        margin=dict(
            l=10,
            r=75,
            t=15,
            b=10,
        ),

        paper_bgcolor=
            "rgba(0,0,0,0)",

        plot_bgcolor=
            "rgba(0,0,0,0)",

        font=dict(
            color="#665F72",
            family="Inter, Arial, sans-serif",
        ),

        showlegend=
            False,

        xaxis=dict(

            range=[
                0,
                min(
                    1.08,
                    max_value + 0.13,
                )
            ],

            showgrid=
                True,

            gridcolor=
                "rgba(72,61,100,0.07)",

            tickformat=
                ".0%",

            zeroline=
                False,
        ),

        yaxis=dict(
            showgrid=False
        ),
    )

    return fig


# ============================================================
# PREMIUM ANIMATED HERO
# ============================================================

motion_state = (
    "paused"
    if reduce_motion
    else "running"
)

reduce_motion_js = (
    "true"
    if reduce_motion
    else "false"
)


hero_html = """
<!DOCTYPE html>

<html>

<head>

<style>

* {
    box-sizing: border-box;
}


html,
body {

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
}


.hero {

    position: relative;

    width: 100%;

    min-height: 365px;

    overflow: hidden;

    border-radius: 28px;

    background:

        radial-gradient(
            circle at 50% -25%,
            rgba(109,74,255,0.20),
            transparent 45%
        ),

        radial-gradient(
            circle at 92% 75%,
            rgba(195,182,255,0.25),
            transparent 38%
        ),

        radial-gradient(
            circle at 5% 80%,
            rgba(229,223,255,0.50),
            transparent 34%
        ),

        linear-gradient(
            145deg,
            #FFFFFF,
            #F7F4FF
        );

    border:
        1px solid
        rgba(109,74,255,0.13);

    box-shadow:

        0 20px 55px
        rgba(65,44,130,0.07),

        inset 0 1px 0
        rgba(255,255,255,0.95);
}


.hero::after {

    content: "";

    position: absolute;

    inset: 0;

    pointer-events: none;

    opacity: 0.40;

    background:

        linear-gradient(
            110deg,
            transparent 15%,
            rgba(255,255,255,0.72) 48%,
            transparent 72%
        );
}


#particles {

    position: absolute;

    inset: 0;

    width: 100%;

    height: 100%;
}


.wave-container {

    position: absolute;

    inset: 0;

    width: 100%;

    height: 100%;

    opacity: 0.64;
}


.wave {

    fill: none;

    stroke-width: 0.85;

    stroke-linecap: round;

    animation:
        waveMove
        9s
        ease-in-out
        infinite;

    animation-play-state:
        __MOTION_STATE__;
}


.wave.primary {
    stroke: rgba(109,74,255,0.34);
}


.wave.secondary {
    stroke: rgba(156,133,255,0.20);
}


.wave.grey {
    stroke: rgba(80,72,98,0.11);
}


.wave:nth-child(2) {
    animation-delay: -0.8s;
}

.wave:nth-child(3) {
    animation-delay: -1.6s;
}

.wave:nth-child(4) {
    animation-delay: -2.4s;
}

.wave:nth-child(5) {
    animation-delay: -3.2s;
}

.wave:nth-child(6) {
    animation-delay: -4s;
}

.wave:nth-child(7) {
    animation-delay: -4.8s;
}

.wave:nth-child(8) {
    animation-delay: -5.6s;
}

.wave:nth-child(9) {
    animation-delay: -6.4s;
}

.wave:nth-child(10) {
    animation-delay: -7.2s;
}

.wave:nth-child(11) {
    animation-delay: -8s;
}


@keyframes waveMove {

    0%,
    100% {

        transform:
            translateX(-7px)
            translateY(0px);
    }


    50% {

        transform:
            translateX(14px)
            translateY(-8px);
    }
}


.hero-content {

    position: relative;

    z-index: 5;

    min-height: 365px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    padding:
        45px 30px;

    text-align: center;
}


.brand-row {

    display: flex;

    align-items: center;

    justify-content: center;

    gap: 17px;

    margin-bottom: 13px;
}


.logo {

    width: 72px;

    height: 72px;

    filter:

        drop-shadow(
            0 12px 24px
            rgba(109,74,255,0.20)
        );
}


.logo-flow {

    animation:
        logoPulse
        3.3s
        ease-in-out
        infinite;

    animation-play-state:
        __MOTION_STATE__;
}


@keyframes logoPulse {

    0%,
    100% {
        opacity: 0.72;
    }


    50% {
        opacity: 1;
    }
}


.brand-name {

    color:
        #251B47;

    font-size:
        64px;

    font-weight:
        800;

    letter-spacing:
        -3.8px;

    line-height:
        1;
}


.tagline {

    color:
        #352A56;

    font-size:
        22px;

    font-weight:
        650;

    letter-spacing:
        -0.45px;

    margin-bottom:
        12px;
}


.description {

    max-width:
        690px;

    color:
        #777083;

    font-size:
        15px;

    line-height:
        1.65;
}


.status {

    margin-top:
        22px;

    display:
        inline-flex;

    align-items:
        center;

    gap:
        8px;

    padding:
        8px 13px;

    color:
        #756D82;

    font-size:
        11px;

    letter-spacing:
        0.04em;

    background:
        rgba(255,255,255,0.64);

    border:
        1px solid
        rgba(109,74,255,0.12);

    border-radius:
        999px;

    backdrop-filter:
        blur(10px);
}


.status-dot {

    width:
        6px;

    height:
        6px;

    border-radius:
        100%;

    background:
        #6D4AFF;

    box-shadow:
        0 0 12px
        rgba(109,74,255,0.55);
}


@media(max-width:650px) {

    .hero {

        min-height:
            335px;
    }


    .hero-content {

        min-height:
            335px;

        padding:
            35px 20px;
    }


    .brand-row {

        gap:
            10px;
    }


    .logo {

        width:
            51px;

        height:
            51px;
    }


    .brand-name {

        font-size:
            44px;

        letter-spacing:
            -2.5px;
    }


    .tagline {

        font-size:
            17px;
    }


    .description {

        font-size:
            13px;
    }
}

</style>

</head>


<body>


<div class="hero">


<canvas id="particles"></canvas>


<svg
    class="wave-container"
    viewBox="0 0 1200 365"
    preserveAspectRatio="none"
>


<path
    class="wave primary"
    d="M-100 290 C100 100 280 325 455 170 S720 50 900 190 S1120 285 1300 80"
/>


<path
    class="wave secondary"
    d="M-100 302 C100 112 280 337 455 182 S720 62 900 202 S1120 297 1300 92"
/>


<path
    class="wave grey"
    d="M-100 314 C100 124 280 349 455 194 S720 74 900 214 S1120 309 1300 104"
/>


<path
    class="wave primary"
    d="M-100 326 C100 136 280 361 455 206 S720 86 900 226 S1120 321 1300 116"
/>


<path
    class="wave secondary"
    d="M-100 338 C100 148 280 373 455 218 S720 98 900 238 S1120 333 1300 128"
/>


<path
    class="wave grey"
    d="M-100 278 C100 88 280 313 455 158 S720 38 900 178 S1120 273 1300 68"
/>


<path
    class="wave primary"
    d="M-100 266 C100 76 280 301 455 146 S720 26 900 166 S1120 261 1300 56"
/>


<path
    class="wave secondary"
    d="M-100 254 C100 64 280 289 455 134 S720 14 900 154 S1120 249 1300 44"
/>


<path
    class="wave grey"
    d="M-100 242 C100 52 280 277 455 122 S720 2 900 142 S1120 237 1300 32"
/>


<path
    class="wave primary"
    d="M-100 230 C100 40 280 265 455 110 S720 -10 900 130 S1120 225 1300 20"
/>


<path
    class="wave secondary"
    d="M-100 218 C100 28 280 253 455 98 S720 -22 900 118 S1120 213 1300 8"
/>


</svg>


<div class="hero-content">


<div class="brand-row">


<svg
    class="logo"
    viewBox="0 0 120 120"
    xmlns="http://www.w3.org/2000/svg"
>


<defs>


<linearGradient
    id="logoBody"
    x1="0"
    y1="0"
    x2="1"
    y2="1"
>

<stop
    offset="0%"
    stop-color="#8064FF"
/>

<stop
    offset="100%"
    stop-color="#4E2FCB"
/>

</linearGradient>


<linearGradient
    id="signalGradient"
    x1="0"
    x2="1"
>

<stop
    offset="0%"
    stop-color="#C9BEFF"
/>

<stop
    offset="100%"
    stop-color="#6D4AFF"
/>

</linearGradient>


</defs>


<path
    d="M31 13 H61 C93 13 108 31 108 60 C108 89 93 107 61 107 H31 Z"
    fill="url(#logoBody)"
/>


<path
    d="M46 29 H61 C79 29 91 40 91 60 C91 80 79 91 61 91 H46 Z"
    fill="#F9F7FF"
/>


<g class="logo-flow">


<circle
    cx="15"
    cy="34"
    r="5"
    fill="#7658FF"
/>


<circle
    cx="15"
    cy="60"
    r="5"
    fill="#B4A4FF"
/>


<circle
    cx="15"
    cy="86"
    r="5"
    fill="#7658FF"
/>


<path
    d="M20 34 C43 34 43 57 62 60"
    fill="none"
    stroke="url(#signalGradient)"
    stroke-width="5"
    stroke-linecap="round"
/>


<path
    d="M20 60 H62"
    fill="none"
    stroke="#9C86FF"
    stroke-width="5"
    stroke-linecap="round"
/>


<path
    d="M20 86 C43 86 43 63 62 60"
    fill="none"
    stroke="url(#signalGradient)"
    stroke-width="5"
    stroke-linecap="round"
/>


<circle
    cx="65"
    cy="60"
    r="7"
    fill="#6D4AFF"
/>


<path
    d="M71 60 H95"
    fill="none"
    stroke="#6D4AFF"
    stroke-width="5"
    stroke-linecap="round"
/>


<path
    d="M91 52 L101 60 L91 68"
    fill="none"
    stroke="#6D4AFF"
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

<span class="status-dot"></span>

Fine-tuned RoBERTa · 27 intents · 11 business categories

</div>


</div>


</div>


<script>


const reduceMotion =
    __REDUCE_MOTION__;


const canvas =
    document.getElementById("particles");


const ctx =
    canvas.getContext("2d");


let particles = [];

let width = 0;

let height = 0;


function resizeCanvas() {

    const rect =
        canvas.getBoundingClientRect();

    const ratio =
        window.devicePixelRatio || 1;

    width =
        rect.width;

    height =
        rect.height;

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
}


function createParticles() {

    particles = [];

    const count =
        Math.max(
            30,
            Math.min(
                62,
                Math.floor(width / 18)
            )
        );


    for (
        let i = 0;
        i < count;
        i++
    ) {

        particles.push({

            x:
                Math.random()
                * width,

            y:
                Math.random()
                * height,

            vx:
                (Math.random() - 0.5)
                * 0.15,

            vy:
                (Math.random() - 0.5)
                * 0.11,

            size:
                Math.random()
                * 1.45
                + 0.35,

            alpha:
                Math.random()
                * 0.24
                + 0.06,

            purple:
                Math.random()
                > 0.50
        });
    }
}


function drawParticles() {

    ctx.clearRect(
        0,
        0,
        width,
        height
    );


    for (
        const particle
        of particles
    ) {

        if (!reduceMotion) {

            particle.x +=
                particle.vx;

            particle.y +=
                particle.vy;


            if (
                particle.x < -10
            ) {
                particle.x =
                    width + 10;
            }


            if (
                particle.x > width + 10
            ) {
                particle.x =
                    -10;
            }


            if (
                particle.y < -10
            ) {
                particle.y =
                    height + 10;
            }


            if (
                particle.y > height + 10
            ) {
                particle.y =
                    -10;
            }
        }


        ctx.beginPath();


        ctx.arc(
            particle.x,
            particle.y,
            particle.size,
            0,
            Math.PI * 2
        );


        if (
            particle.purple
        ) {

            ctx.fillStyle =
                "rgba(109,74,255,"
                + particle.alpha
                + ")";

        } else {

            ctx.fillStyle =
                "rgba(126,118,145,"
                + particle.alpha
                + ")";
        }


        ctx.fill();
    }


    for (
        let i = 0;
        i < particles.length;
        i++
    ) {

        for (
            let j = i + 1;
            j < particles.length;
            j++
        ) {

            const first =
                particles[i];

            const second =
                particles[j];


            const dx =
                first.x
                - second.x;

            const dy =
                first.y
                - second.y;


            const distance =
                Math.sqrt(
                    dx * dx
                    + dy * dy
                );


            if (
                distance < 95
            ) {

                const opacity =
                    (
                        1
                        - distance / 95
                    )
                    * 0.035;


                ctx.beginPath();


                ctx.moveTo(
                    first.x,
                    first.y
                );


                ctx.lineTo(
                    second.x,
                    second.y
                );


                ctx.strokeStyle =
                    "rgba(109,74,255,"
                    + opacity
                    + ")";


                ctx.lineWidth =
                    0.5;


                ctx.stroke();
            }
        }
    }


    requestAnimationFrame(
        drawParticles
    );
}


resizeCanvas();

createParticles();

drawParticles();


window.addEventListener(
    "resize",
    function () {

        resizeCanvas();

        createParticles();
    }
);


</script>


</body>

</html>
"""


hero_html = hero_html.replace(
    "__MOTION_STATE__",
    motion_state,
)


hero_html = hero_html.replace(
    "__REDUCE_MOTION__",
    reduce_motion_js,
)


components.html(
    hero_html,
    height=385,
    scrolling=False,
)


# ============================================================
# NAVIGATION
# ============================================================

try_tab, insights_tab, about_tab = st.tabs(
    [
        "✦ Try Decodr",
        "Model Insights",
        "About",
    ]
)


# ============================================================
# TRY DECODR
# ============================================================

with try_tab:


    st.markdown(
        '<div class="kicker">'
        'Customer Query'
        '</div>'

        '<div class="section-title">'
        'What does the customer need?'
        '</div>'

        '<div class="section-copy">'
        'Enter a customer-service request in natural language. '
        'Decodr identifies the underlying intent and evaluates '
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


    analyze_button = st.button(
        "Analyze Intent  →",
        type="primary",
        use_container_width=True,
    )


    if analyze_button:


        if not st.session_state.query_text.strip():

            st.warning(
                "Enter a customer query first."
            )


        else:


            with st.spinner(
                "Decoding customer intent..."
            ):


                st.session_state.prediction_result = (
                    predict_intent(
                        st.session_state.query_text
                    )
                )


    result = (
        st.session_state.prediction_result
    )


    if result:


        safe_intent = html_lib.escape(
            readable_intent(
                result["intent"]
            )
        )


        safe_category = html_lib.escape(
            result["category"]
        )


        confidence = (
            result["confidence"]
        )


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


        column1, column2, column3 = (
            st.columns(3)
        )


        with column1:


            st.markdown(
                '<div class="result-card">'

                '<div class="result-icon">'
                '◆'
                '</div>'

                '<div class="result-label">'
                'Predicted Intent'
                '</div>'

                '<div class="result-value">'
                + safe_intent +
                '</div>'

                '<div class="result-subtext">'
                'Most likely customer need'
                '</div>'

                '</div>',
                unsafe_allow_html=True,
            )


        with column2:


            st.markdown(
                '<div class="result-card">'

                '<div class="result-icon">'
                '↳'
                '</div>'

                '<div class="result-label">'
                'Business Category'
                '</div>'

                '<div class="result-value">'
                + safe_category +
                '</div>'

                '<div class="result-subtext">'
                'Suggested operational workflow'
                '</div>'

                '</div>',
                unsafe_allow_html=True,
            )


        with column3:


            st.markdown(
                '<div class="result-card">'

                '<div class="result-icon">'
                '%'
                '</div>'

                '<div class="result-label">'
                'Model Confidence'
                '</div>'

                '<div class="result-value">'
                + f"{confidence:.2%}" +
                '</div>'

                '<div class="result-subtext">'
                'Operating threshold: '
                + f"{AUTO_THRESHOLD:.2%}" +
                '</div>'

                '</div>',
                unsafe_allow_html=True,
            )


        st.markdown(
            '<div class="confidence-wrap">'

            '<div class="confidence-head">'

            '<span>'
            'Prediction confidence'
            '</span>'

            '<span>'
            + f"{confidence:.2%}" +
            '</span>'

            '</div>'

            '<div class="confidence-track">'

            '<div class="confidence-fill" '
            'style="width:'
            + f"{confidence_percentage:.2f}%"
            + '">'
            '</div>'

            '</div>'

            '</div>',
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
                '<div class="route-auto">'

                '<div class="route-badge-auto">'
                '● AUTOMATION CANDIDATE'
                '</div>'

                '<div class="route-title">'
                'Automatic routing recommended'
                '</div>'

                '<div class="route-copy">'

                'The predicted intent is '

                '<strong>'
                + safe_intent +
                '</strong> '

                'with a confidence of '

                '<strong>'
                + f"{confidence:.2%}" +
                '</strong>. '

                'This exceeds the validation-selected '

                '<strong>'
                + f"{AUTO_THRESHOLD:.2%}" +
                '</strong> '

                'operating threshold, so the query is '
                'eligible to be routed to the '

                '<strong>'
                + safe_category +
                '</strong> '

                'workflow.'

                '</div>'

                '</div>',
                unsafe_allow_html=True,
            )


        else:


            st.markdown(
                '<div class="route-review">'

                '<div class="route-badge-review">'
                '● REVIEW REQUIRED'
                '</div>'

                '<div class="route-title">'
                'Human review recommended'
                '</div>'

                '<div class="route-copy">'

                'The predicted intent is '

                '<strong>'
                + safe_intent +
                '</strong>, '

                'but the confidence of '

                '<strong>'
                + f"{confidence:.2%}" +
                '</strong> '

                'is below the validation-selected '

                '<strong>'
                + f"{AUTO_THRESHOLD:.2%}" +
                '</strong> '

                'operating threshold. '
                'The query should therefore be '
                'reviewed before an operational '
                'routing decision is made.'

                '</div>'

                '</div>',
                unsafe_allow_html=True,
            )


        # ====================================================
        # ALTERNATIVES
        # ====================================================

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
                'The next highest-probability predictions provide '
                'additional context around model uncertainty.'
                '</div>',
                unsafe_allow_html=True,
            )


            second_prediction = (
                result["predictions"][1]
            )


            third_prediction = (
                result["predictions"][2]
            )


            alt_column1, alt_column2 = (
                st.columns(2)
            )


            with alt_column1:


                second_intent = (
                    html_lib.escape(
                        readable_intent(
                            second_prediction["intent"]
                        )
                    )
                )


                second_category = (
                    html_lib.escape(
                        second_prediction["category"]
                    )
                )


                st.markdown(
                    '<div class="alt-card">'

                    '<div class="alt-position">'
                    'Second most likely'
                    '</div>'

                    '<div class="alt-intent">'
                    + second_intent +
                    '</div>'

                    '<div class="alt-category">'
                    + second_category +
                    '</div>'

                    '<div class="alt-confidence">'
                    + f'{second_prediction["confidence"]:.2%}'
                    + ' confidence'
                    '</div>'

                    '</div>',
                    unsafe_allow_html=True,
                )


            with alt_column2:


                third_intent = (
                    html_lib.escape(
                        readable_intent(
                            third_prediction["intent"]
                        )
                    )
                )


                third_category = (
                    html_lib.escape(
                        third_prediction["category"]
                    )
                )


                st.markdown(
                    '<div class="alt-card">'

                    '<div class="alt-position">'
                    'Third most likely'
                    '</div>'

                    '<div class="alt-intent">'
                    + third_intent +
                    '</div>'

                    '<div class="alt-category">'
                    + third_category +
                    '</div>'

                    '<div class="alt-confidence">'
                    + f'{third_prediction["confidence"]:.2%}'
                    + ' confidence'
                    '</div>'

                    '</div>',
                    unsafe_allow_html=True,
                )


        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        if show_probability_chart:


            st.write("")


            st.plotly_chart(
                probability_chart(
                    result
                ),
                use_container_width=True,
                config={
                    "displayModeBar":
                        False,

                    "responsive":
                        True,
                },
            )


        # ====================================================
        # EXPLANATION
        # ====================================================

        if show_explanation:


            st.write("")


            with st.expander(
                "How does Decodr make this decision?"
            ):


                st.markdown(
                    f"""
Decodr uses the final fine-tuned **RoBERTa sequence-classification model**
evaluated in the dissertation.

The customer query is classified into one of **27 customer-service intents**.
The intent receiving the highest predicted probability becomes the primary
prediction.

The confidence of that prediction is then compared with the
validation-selected operating threshold of **{AUTO_THRESHOLD:.2%}**.

**Confidence ≥ {AUTO_THRESHOLD:.2%}**  
→ Candidate for automatic routing.

**Confidence < {AUTO_THRESHOLD:.2%}**  
→ Human review recommended.

The broader business category is assigned **after** intent prediction.
It is not supplied to RoBERTa as an input.
"""
                )


# ============================================================
# MODEL INSIGHTS
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


    metric1, metric2, metric3, metric4 = (
        st.columns(4)
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
            "Total test errors",
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


    threshold1, threshold2, threshold3, threshold4 = (
        st.columns(4)
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
        "At the selected validation threshold, coverage was "
        "99.6955% and selective accuracy was 99.9167%. "
        "Eleven of 3,612 validation queries were referred for review."
    )


    st.write("")
    st.write("")


    insight1, insight2, insight3 = (
        st.columns(3)
    )


    with insight1:


        st.markdown(
            '<div class="info-card">'

            '<div class="info-label">'
            'Predictive Performance'
            '</div>'

            '<div class="info-title">'
            'Small headline gain'
            '</div>'

            '<div class="info-copy">'
            'RoBERTa improved Macro-F1 over Linear SVM by '
            'approximately 0.484 percentage points. '
            'The aggregate performance difference is therefore modest.'
            '</div>'

            '</div>',
            unsafe_allow_html=True,
        )


    with insight2:


        st.markdown(
            '<div class="info-card">'

            '<div class="info-label">'
            'Operational Risk'
            '</div>'

            '<div class="info-title">'
            'Fewer consequential errors'
            '</div>'

            '<div class="info-copy">'
            'RoBERTa produced 3 test errors, including '
            '1 high-risk error, compared with 20 total and '
            '14 high-risk errors for Linear SVM.'
            '</div>'

            '</div>',
            unsafe_allow_html=True,
        )


    with insight3:


        st.markdown(
            '<div class="info-card">'

            '<div class="info-label">'
            'Deployment Trade-Off'
            '</div>'

            '<div class="info-title">'
            'Performance has a cost'
            '</div>'

            '<div class="info-copy">'
            'RoBERTa achieved the strongest predictive result '
            'but required substantially greater training time '
            'and model storage than the traditional baselines.'
            '</div>'

            '</div>',
            unsafe_allow_html=True,
        )


    st.write("")
    st.write("")


    st.markdown(
        '<div class="kicker">'
        'Computational Efficiency'
        '</div>'

        '<div class="section-title" '
        'style="font-size:1.55rem;">'
        'Performance versus resource requirements'
        '</div>',
        unsafe_allow_html=True,
    )


    efficiency_results = pd.DataFrame(
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


    st.dataframe(
        efficiency_results,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# ABOUT
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


    about1, about2 = (
        st.columns(2)
    )


    with about1:


        st.markdown(
            '<div class="info-card">'

            '<div class="info-label">'
            'Purpose'
            '</div>'

            '<div class="info-title">'
            'From prediction to routing'
            '</div>'

            '<div class="info-copy">'
            'Decodr demonstrates how a customer query can be '
            'classified into a fine-grained intent and subsequently '
            'mapped to a broader business workflow. Confidence adds '
            'a decision layer determining whether the prediction is '
            'suitable for automatic handling or should first be '
            'reviewed by a human.'
            '</div>'

            '</div>',
            unsafe_allow_html=True,
        )


    with about2:


        st.markdown(
            '<div class="info-card">'

            '<div class="info-label">'
            'Model'
            '</div>'

            '<div class="info-title">'
            'Fine-tuned RoBERTa'
            '</div>'

            '<div class="info-copy">'
            'The deployed classifier is the final RoBERTa model '
            'evaluated in the dissertation. It predicts among '
            '27 customer-service intents, which are subsequently '
            'mapped to 11 broader operational categories.'
            '</div>'

            '</div>',
            unsafe_allow_html=True,
        )


    st.write("")


    about3, about4 = (
        st.columns(2)
    )


    with about3:


        st.markdown(
            '<div class="info-card">'

            '<div class="info-label">'
            'Human-in-the-Loop'
            '</div>'

            '<div class="info-title">'
            'Selective automation'
            '</div>'

            '<div class="info-copy">'
            'The prototype does not assume every prediction should '
            'trigger an automated action. Lower-confidence cases are '
            'deliberately referred for human review, illustrating a '
            'hybrid automation approach.'
            '</div>'

            '</div>',
            unsafe_allow_html=True,
        )


    with about4:


        st.markdown(
            '<div class="info-card">'

            '<div class="info-label">'
            'Scope'
            '</div>'

            '<div class="info-title">'
            'Academic demonstration'
            '</div>'

            '<div class="info-copy">'
            'Decodr is an MSc Business Analytics research prototype, '
            'not a production customer-service deployment. Performance '
            'reflects the dissertation dataset and should not be '
            'interpreted as guaranteed live-system performance.'
            '</div>'

            '</div>',
            unsafe_allow_html=True,
        )


    st.write("")


    st.warning(
        "The underlying dataset is synthetic, English-language and "
        "closed-set. Real-world deployment would require evaluation "
        "on operational customer queries, monitoring for distribution "
        "shift and mechanisms for queries outside the known intent taxonomy."
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
