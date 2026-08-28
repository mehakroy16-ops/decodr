import html
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
# RESEARCH CONFIG
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

    "Ambiguous request":
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

    st.markdown("#### Operating point")

    st.metric(
        "Routing threshold",
        "99.10%",
    )

    st.caption(
        "Validation-selected threshold used for "
        "confidence-based routing."
    )


# ============================================================
# GLOBAL UI
# ============================================================

st.markdown(
    """
<style>

:root {

    --purple: #6D4AFF;
    --purple-dark: #4F35D2;
    --purple-soft: #EEE9FF;
    --purple-soft-2: #F6F3FF;

    --bg: #F8F7FC;

    --surface: #FFFFFF;
    --surface-soft: #FBFAFF;

    --border: #E9E5F2;

    --text: #1D1B2A;
    --text-2: #494557;
    --muted: #858090;

    --green: #18A875;
    --green-soft: #EAFBF5;

    --amber: #D88B16;
    --amber-soft: #FFF8E9;
}


/* ---------------------------------------------------------
   APP BACKGROUND
--------------------------------------------------------- */

[data-testid="stAppViewContainer"] {

    background:

        radial-gradient(
            circle at 5% 0%,
            rgba(109, 74, 255, 0.09),
            transparent 26%
        ),

        radial-gradient(
            circle at 95% 12%,
            rgba(181, 160, 255, 0.11),
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

    max-width: 1180px;

    padding-top: 1.3rem;
    padding-bottom: 4rem;
}


/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #FBFAFF,
            #F5F2FC
        );

    border-right: 1px solid #E9E5F2;
}


[data-testid="stSidebar"] * {
    color: #2C2840;
}


/* ---------------------------------------------------------
   HEADINGS
--------------------------------------------------------- */

h1,
h2,
h3 {

    color: #1D1B2A !important;
}


/* ---------------------------------------------------------
   TABS
--------------------------------------------------------- */

.stTabs [data-baseweb="tab-list"] {

    gap: 8px;

    background:
        rgba(
            255,
            255,
            255,
            0.88
        );

    border:
        1px solid
        #E8E4F1;

    padding: 7px;

    border-radius: 17px;

    margin-top: 10px;
    margin-bottom: 30px;

    box-shadow:
        0 8px 30px
        rgba(58, 39, 120, 0.05);
}


.stTabs [data-baseweb="tab"] {

    height: 47px;

    border-radius: 11px;

    padding:
        0 23px;

    color: #777181;

    font-weight: 650;
}


.stTabs [aria-selected="true"] {

    color: #5A3FE0 !important;

    background:
        linear-gradient(
            135deg,
            #F0EBFF,
            #F8F6FF
        ) !important;

    box-shadow:
        inset 0 0 0 1px
        rgba(109,74,255,0.13);
}


.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}


/* ---------------------------------------------------------
   INPUT
--------------------------------------------------------- */

.stTextArea textarea {

    background:
        rgba(
            255,
            255,
            255,
            0.95
        ) !important;

    color:
        #242131 !important;

    border:
        1px solid
        #E3DFEB !important;

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
        rgba(52,35,105,0.035);
}


.stTextArea textarea:focus {

    border-color:
        #8A70FF !important;

    box-shadow:

        0 0 0 3px
        rgba(109,74,255,0.08),

        0 12px 35px
        rgba(67,46,150,0.07) !important;
}


/* ---------------------------------------------------------
   BUTTON
--------------------------------------------------------- */

.stButton > button {

    min-height: 49px;

    border-radius:
        13px !important;

    border:
        1px solid
        #6042E8 !important;

    background:

        linear-gradient(
            135deg,
            #7455FF,
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
        0 12px 34px
        rgba(109,74,255,0.28);
}


/* ---------------------------------------------------------
   SECTION TITLES
--------------------------------------------------------- */

.kicker {

    color:
        #7051EF;

    font-size:
        0.76rem;

    font-weight:
        750;

    letter-spacing:
        0.16em;

    text-transform:
        uppercase;

    margin-bottom:
        7px;
}


.section-title {

    color:
        #1C1928;

    font-size:
        2rem;

    font-weight:
        770;

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


/* ---------------------------------------------------------
   RESULT CARDS
--------------------------------------------------------- */

.result-card {

    position:
        relative;

    background:

        linear-gradient(
            145deg,
            #FFFFFF,
            #FBFAFF
        );

    border:
        1px solid
        #E9E5F2;

    border-radius:
        21px;

    padding:
        22px;

    min-height:
        172px;

    box-shadow:
        0 14px 40px
        rgba(49,33,103,0.055);

    overflow:
        hidden;
}


.result-card::before {

    content: "";

    position:
        absolute;

    width:
        130px;

    height:
        130px;

    right:
        -65px;

    top:
        -65px;

    border-radius:
        100%;

    background:

        radial-gradient(
            circle,
            rgba(109,74,255,0.10),
            transparent 70%
        );
}


.result-icon {

    width:
        36px;

    height:
        36px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        11px;

    background:
        #F0EBFF;

    color:
        #6645EA;

    font-size:
        0.95rem;

    margin-bottom:
        21px;
}


.result-label {

    color:
        #8A8493;

    font-size:
        0.71rem;

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
        #211E2C;

    font-size:
        1.52rem;

    font-weight:
        770;

    letter-spacing:
        -0.025em;

    line-height:
        1.2;

    margin-bottom:
        8px;
}


.result-subtext {

    color:
        #8B8595;

    font-size:
        0.86rem;

    line-height:
        1.45;
}


/* ---------------------------------------------------------
   CONFIDENCE BAR
--------------------------------------------------------- */

.confidence-wrap {

    background:
        #FFFFFF;

    border:
        1px solid
        #E8E4F0;

    border-radius:
        16px;

    padding:
        17px 18px;

    margin-top:
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

    margin-bottom:
        11px;

    color:
        #716B7C;

    font-size:
        0.88rem;
}


.confidence-track {

    width:
        100%;

    height:
        8px;

    background:
        #EEEAF5;

    border-radius:
        100px;

    overflow:
        hidden;
}


.confidence-fill {

    height:
        100%;

    border-radius:
        100px;

    background:

        linear-gradient(
            90deg,
            #5E3FE1,
            #9F89FF
        );

    box-shadow:
        0 0 15px
        rgba(109,74,255,0.20);
}


/* ---------------------------------------------------------
   ROUTING DECISION
--------------------------------------------------------- */

.route-auto {

    background:

        linear-gradient(
            135deg,
            #EAFBF5,
            #FFFFFF 55%
        );

    border:
        1px solid
        #CBEFE1;

    border-radius:
        20px;

    padding:
        24px;

    margin-top:
        12px;

    box-shadow:
        0 12px 30px
        rgba(29,120,89,0.04);
}


.route-review {

    background:

        linear-gradient(
            135deg,
            #FFF8E9,
            #FFFFFF 55%
        );

    border:
        1px solid
        #F3DFC0;

    border-radius:
        20px;

    padding:
        24px;

    margin-top:
        12px;
}


.route-badge-auto,
.route-badge-review {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        8px;

    padding:
        7px 11px;

    border-radius:
        999px;

    font-size:
        0.76rem;

    font-weight:
        720;

    margin-bottom:
        13px;
}


.route-badge-auto {

    color:
        #138C62;

    background:
        #E3F8F0;
}


.route-badge-review {

    color:
        #B67819;

    background:
        #FFF1D6;
}


.route-title {

    color:
        #211E2C;

    font-size:
        1.35rem;

    font-weight:
        760;

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
        1.7;
}


/* ---------------------------------------------------------
   ALTERNATIVE CARDS
--------------------------------------------------------- */

.alt-card {

    background:

        linear-gradient(
            145deg,
            #FFFFFF,
            #FBFAFF
        );

    border:
        1px solid
        #E8E4F0;

    border-radius:
        18px;

    padding:
        19px;

    min-height:
        140px;

    box-shadow:
        0 10px 28px
        rgba(49,33,103,0.04);
}


.alt-position {

    color:
        #A09AA8;

    font-size:
        0.7rem;

    font-weight:
        720;

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
        1.12rem;

    font-weight:
        720;

    margin-bottom:
        7px;
}


.alt-category {

    color:
        #918B9A;

    font-size:
        0.83rem;

    margin-bottom:
        14px;
}


.alt-confidence {

    color:
        #6445DF;

    font-weight:
        720;

    font-size:
        0.91rem;
}


/* ---------------------------------------------------------
   INFO CARDS
--------------------------------------------------------- */

.info-card {

    background:

        linear-gradient(
            145deg,
            #FFFFFF,
            #FBFAFF
        );

    border:
        1px solid
        #E8E4F0;

    border-radius:
        19px;

    padding:
        22px;

    height:
        100%;

    box-shadow:
        0 10px 30px
        rgba(49,33,103,0.04);
}


.info-label {

    color:
        #7051EF;

    font-size:
        0.71rem;

    font-weight:
        730;

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
        720;

    margin-bottom:
        10px;
}


.info-copy {

    color:
        #777181;

    font-size:
        0.93rem;

    line-height:
        1.7;
}


/* ---------------------------------------------------------
   STREAMLIT METRICS
--------------------------------------------------------- */

[data-testid="stMetric"] {

    background:
        linear-gradient(
            145deg,
            #FFFFFF,
            #FBFAFF
        );

    border:
        1px solid
        #E8E4F0;

    border-radius:
        17px;

    padding:
        18px;

    box-shadow:
        0 9px 26px
        rgba(49,33,103,0.04);
}


[data-testid="stMetricLabel"] {
    color: #7D7787;
}


[data-testid="stMetricValue"] {
    color: #241F31;
}


/* ---------------------------------------------------------
   DATAFRAMES
--------------------------------------------------------- */

[data-testid="stDataFrame"] {

    border-radius:
        16px;

    overflow:
        hidden;

    border:
        1px solid
        #E8E4F0;
}


/* ---------------------------------------------------------
   EXPANDER
--------------------------------------------------------- */

[data-testid="stExpander"] {

    background:
        #FFFFFF;

    border:
        1px solid
        #E7E3EF;

    border-radius:
        16px;
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.decodr-footer {

    text-align:
        center;

    margin-top:
        45px;

    color:
        #9892A0;

    font-size:
        0.82rem;
}


.footer-dot {

    color:
        #704FF0;

    padding:
        0 8px;
}


/* ---------------------------------------------------------
   RESPONSIVE
--------------------------------------------------------- */

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

    .stTabs [data-baseweb="tab"] {

        padding:
            0 11px;

        font-size:
            0.87rem;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
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
        "Check the Hugging Face access configuration."
    )

    st.stop()


# ============================================================
# MODEL FUNCTIONS
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

                "category":
                    INTENT_TO_CATEGORY.get(
                        intent,
                        "UNKNOWN",
                    ),

                "confidence":
                    probability.item(),
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
# CHART
# ============================================================

def probability_chart(result):

    predictions = result["predictions"]

    labels = [
        readable_intent(
            x["intent"]
        )
        for x in predictions
    ]

    values = [
        x["confidence"]
        for x in predictions
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=values[::-1],
            y=labels[::-1],

            orientation="h",

            text=[
                f"{v:.2%}"
                for v in values[::-1]
            ],

            textposition="outside",

            marker=dict(
                color=[
                    "#DAD4ED",
                    "#CCC3EB",
                    "#B9AAEF",
                    "#9983F3",
                    "#6D4AFF",
                ]
            ),

            hovertemplate=(
                "<b>%{y}</b><br>"
                "Confidence: %{x:.2%}"
                "<extra></extra>"
            ),
        )
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
            color="#645E70",
            family="Inter, Arial, sans-serif",
        ),

        showlegend=False,

        xaxis=dict(

            range=[
                0,
                min(
                    1.08,
                    max(values) + 0.12,
                )
            ],

            showgrid=True,

            gridcolor=
                "rgba(74,61,105,0.07)",

            tickformat=".0%",

            zeroline=False,
        ),

        yaxis=dict(
            showgrid=False
        ),
    )

    return fig


# ============================================================
# PREMIUM PURPLE HERO
# ============================================================

motion_state = (
    "paused"
    if reduce_motion
    else "running"
)


hero_html = f"""
<!DOCTYPE html>

<html>

<head>

<style>

* {{
    box-sizing: border-box;
}}

html,
body {{

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

    min-height: 365px;

    overflow: hidden;

    border-radius: 27px;

    background:

        radial-gradient(
            circle at 50% -30%,
            rgba(115,77,255,0.18),
            transparent 48%
        ),

        radial-gradient(
            circle at 90% 70%,
            rgba(190,176,255,0.24),
            transparent 38%
        ),

        radial-gradient(
            circle at 5% 80%,
            rgba(224,217,255,0.42),
            transparent 35%
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
        inset 0 1px 0
        rgba(255,255,255,0.9);
}}


.hero::after {{

    content: "";

    position: absolute;

    inset: 0;

    background:

        linear-gradient(
            110deg,
            transparent 15%,
            rgba(255,255,255,0.58) 48%,
            transparent 70%
        );

    opacity: 0.35;

    pointer-events: none;
}}


/* PARTICLES */

#particles {{

    position: absolute;

    inset: 0;

    width: 100%;

    height: 100%;
}}


/* WAVES */

.waves {{

    position: absolute;

    inset: 0;

    width: 100%;

    height: 100%;

    opacity: 0.57;
}}


.wave {{

    fill: none;

    stroke-width: 0.85;

    stroke-linecap: round;

    animation:

        waveMove
        10s
        ease-in-out
        infinite;

    animation-play-state:
        {motion_state};
}}


.wave.purple {{

    stroke:
        rgba(109,74,255,0.32);
}}


.wave.light {{

    stroke:
        rgba(159,137,255,0.20);
}}


.wave.grey {{

    stroke:
        rgba(90,83,106,0.12);
}}


.wave:nth-child(2) {{
    animation-delay: -1s;
}}

.wave:nth-child(3) {{
    animation-delay: -2s;
}}

.wave:nth-child(4) {{
    animation-delay: -3s;
}}

.wave:nth-child(5) {{
    animation-delay: -4s;
}}

.wave:nth-child(6) {{
    animation-delay: -5s;
}}

.wave:nth-child(7) {{
    animation-delay: -6s;
}}

.wave:nth-child(8) {{
    animation-delay: -7s;
}}

.wave:nth-child(9) {{
    animation-delay: -8s;
}}

.wave:nth-child(10) {{
    animation-delay: -9s;
}}


@keyframes waveMove {{

    0%,100% {{

        transform:
            translateX(-6px)
            translateY(0px);
    }}

    50% {{

        transform:
            translateX(13px)
            translateY(-8px);
    }}
}}


/* HERO CONTENT */

.hero-content {{

    position: relative;

    z-index: 5;

    min-height: 365px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    padding:
        48px 30px;

    text-align: center;
}}


.brand-line {{

    display: flex;

    align-items: center;

    justify-content: center;

    gap: 17px;

    margin-bottom:
        13px;
}}


/* LOGO */

.logo {{

    width: 73px;

    height: 73px;

    filter:

        drop-shadow(
            0 12px 24px
            rgba(109,74,255,0.18)
        );
}}


.logo-flow {{

    animation:
        logoPulse
        3.3s
        ease-in-out
        infinite;

    animation-play-state:
        {motion_state};
}}


@keyframes logoPulse {{

    0%,100% {{
        opacity: 0.72;
    }}

    50% {{
        opacity: 1;
    }}
}}


.brand-name {{

    color:
        #251B47;

    font-size:
        63px;

    font-weight:
        790;

    letter-spacing:
        -3.6px;

    line-height:
        1;
}}


.tagline {{

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
}}


.description {{

    max-width:
        690px;

    color:
        #777083;

    font-size:
        15px;

    line-height:
        1.65;
}}


.status {{

    margin-top:
        22px;

    display:
        inline-flex;

    align-items:
        center;

    gap:
        8px;

    border:
        1px solid
        rgba(109,74,255,0.12);

    background:
        rgba(255,255,255,0.65);

    color:
        #756D82;

    border-radius:
        100px;

    padding:
        8px 13px;

    font-size:
        11px;

    letter-spacing:
        0.04em;

    backdrop-filter:
        blur(10px);
}}


.dot {{

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
}}


/* MOBILE */

@media(max-width:650px) {{

    .hero {{
        min-height: 335px;
    }}

    .hero-content {{

        min-height: 335px;

        padding:
            36px 20px;
    }}

    .brand-line {{
        gap: 10px;
    }}

    .logo {{

        width: 51px;
        height: 51px;
    }}

    .brand-name {{

        font-size: 44px;

        letter-spacing:
            -2.5px;
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
    viewBox="0 0 1200 365"
    preserveAspectRatio="none"
>


<path
    class="wave purple"
    d="
    M-100 290
    C110 110 270 320 450 170
    S720 55 900 190
    S1120 285 1300 80
    "
/>


<path
    class="wave light"
    d="
    M-100 302
    C110 122 270 332 450 182
    S720 67 900 202
    S1120 297 1300 92
    "
/>


<path
    class="wave grey"
    d="
    M-100 314
    C110 134 270 344 450 194
    S720 79 900 214
    S1120 309 1300 104
    "
/>


<path
    class="wave purple"
    d="
    M-100 326
    C110 146 270 356 450 206
    S720 91 900 226
    S1120 321 1300 116
    "
/>


<path
    class="wave light"
    d="
    M-100 338
    C110 158 270 368 450 218
    S720 103 900 238
    S1120 333 1300 128
    "
/>


<path
    class="wave grey"
    d="
    M-100 278
    C110 98 270 308 450 158
    S720 43 900 178
    S1120 273 1300 68
    "
/>


<path
    class="wave purple"
    d="
    M-100 266
    C110 86 270 296 450 146
    S720 31 900 166
    S1120 261 1300 56
    "
/>


<path
    class="wave light"
    d="
    M-100 254
    C110 74 270 284 450 134
    S720 19 900 154
    S1120 249 1300 44
    "
/>


<path
    class="wave grey"
    d="
    M-100 242
    C110 62 270 272 450 122
    S720 7 900 142
    S1120 237 1300 32
    "
/>


<path
    class="wave purple"
    d="
    M-100 230
    C110 50 270 260 450 110
    S720 -5 900 130
    S1120 225 1300 20
    "
/>


</svg>


<div class="hero-content">


<div class="brand-line">


<!-- CUSTOM DECODR LOGO -->

<svg
    class="logo"
    viewBox="0 0 120 120"
    xmlns="http://www.w3.org/2000/svg"
>


<defs>


<linearGradient
    id="dBody"
    x1="0"
    y1="0"
    x2="1"
    y2="1"
>

<stop
    offset="0%"
    stop-color="#7658FF"
/>

<stop
    offset="100%"
    stop-color="#4D2ECF"
/>

</linearGradient>


<linearGradient
    id="flow"
    x1="0"
    x2="1"
>

<stop
    offset="0%"
    stop-color="#C6B9FF"
/>

<stop
    offset="100%"
    stop-color="#6D4AFF"
/>

</linearGradient>


</defs>


<!-- D -->


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
    fill="url(#dBody)"
/>


<path
    d="
    M46 29
    H61
    C79 29
    91 40
    91 60
    C91 80
    79 91
    61 91
    H46
    Z
    "
    fill="#F9F7FF"
/>


<!-- ROUTING STREAMS -->

<g class="logo-flow">


<circle
    cx="15"
    cy="34"
    r="5"
    fill="#7B5CFF"
/>


<circle
    cx="15"
    cy="60"
    r="5"
    fill="#B8A9FF"
/>


<circle
    cx="15"
    cy="86"
    r="5"
    fill="#7B5CFF"
/>


<path
    d="
    M20 34
    C42 34
    43 57
    62 60
    "
    fill="none"
    stroke="url(#flow)"
    stroke-width="5"
    stroke-linecap="round"
/>


<path
    d="
    M20 60
    H62
    "
    fill="none"
    stroke="#9C86FF"
    stroke-width="5"
    stroke-linecap="round"
/>


<path
    d="
    M20 86
    C42 86
    43 63
    62 60
    "
    fill="none"
    stroke="url(#flow)"
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
    d="
    M71 60
    H95
    "
    stroke="#6D4AFF"
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
}}


function createParticles() {{

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
    ) {{

        particles.push({{

            x:
                Math.random() * width,

            y:
                Math.random() * height,

            vx:
                (
                    Math.random() - 0.5
                ) * 0.15,

            vy:
                (
                    Math.random() - 0.5
                ) * 0.11,

            size:
                Math.random()
                * 1.45
                + 0.35,

            alpha:
                Math.random()
                * 0.25
                + 0.06,

            purple:
                Math.random()
                > 0.55

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


            if (
                p.x < -10
            )
                p.x =
                    width + 10;


            if (
                p.x > width + 10
            )
                p.x =
                    -10;


            if (
                p.y < -10
            )
                p.y =
                    height + 10;


            if (
                p.y > height + 10
            )
                p.y =
                    -10;
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

            p.purple

            ? `rgba(109,74,255,${{p.alpha}})`

            : `rgba(120,112,140,${{p.alpha}})`;


        ctx.fill();

    }


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
                    ) * 0.035;


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

                    `rgba(109,74,255,${{opacity}})`;


                ctx.lineWidth =
                    0.5;


                ctx.stroke();

            }}

        }}

    }


    requestAnimationFrame(
        draw
    );

}


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
    height=380,
    scrolling=False,
)


# ============================================================
# TABS
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
        '<div class="kicker">Customer Query</div>'
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

                st.session_state.prediction_result = (
                    predict_intent(
                        st.session_state.query_text
                    )
                )


    result = (
        st.session_state.prediction_result
    )


    if result:


        safe_intent = html.escape(
            readable_intent(
                result["intent"]
            )
        )


        safe_category = html.escape(
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


        col1, col2, col3 = st.columns(
            3
        )


        with col1:

            st.markdown(
                f'<div class="result-card">'
                f'<div class="result-icon">◆</div>'
                f'<div class="result-label">'
                f'Predicted Intent'
                f'</div>'
                f'<div class="result-value">'
                f'{safe_intent}'
                f'</div>'
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
                f'<div class="result-label">'
                f'Business Category'
                f'</div>'
                f'<div class="result-value">'
                f'{safe_category}'
                f'</div>'
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
                f'<div class="result-label">'
                f'Model Confidence'
                f'</div>'
                f'<div class="result-value">'
                f'{confidence:.2%}'
                f'</div>'
                f'<div class="result-subtext">'
                f'Operating threshold: '
                f'{AUTO_THRESHOLD:.2%}'
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
                f'with confidence of '
                f'<strong>{confidence:.2%}</strong>. '
                f'This exceeds the validation-selected '
                f'<strong>{AUTO_THRESHOLD:.2%}</strong> '
                f'operating threshold, making the query '
                f'eligible for routing to the '
                f'<strong>{safe_category}</strong> workflow.'
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
                f'but confidence of '
                f'<strong>{confidence:.2%}</strong> '
                f'is below the validation-selected '
                f'<strong>{AUTO_THRESHOLD:.2%}</strong> '
                f'operating threshold. The query should '
                f'therefore be reviewed before routing.'
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
                'The next highest-probability predictions '
                'provide additional context around model uncertainty.'
                '</div>',
                unsafe_allow_html=True,
            )


            alt1 = (
                result["predictions"][1]
            )


            alt2 = (
                result["predictions"][2]
            )


            alternative_col1, alternative_col2 = (
                st.columns(2)
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
                    "displayModeBar":
                        False,

                    "responsive":
                        True,
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

The customer query is classified into one of **27 customer-service intents**.
The intent receiving the highest predicted probability becomes the primary
prediction.

Decodr then compares the prediction confidence with the validation-selected
operating threshold of **{AUTO_THRESHOLD:.2%}**.

**Confidence ≥ {AUTO_THRESHOLD:.2%}**  
→ Candidate for automatic routing.

**Confidence < {AUTO_THRESHOLD:.2%}**  
→ Human review recommended.

The broader business category is assigned **after** intent prediction and is
not provided to RoBERTa as an input.
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


    t1, t2, t3, t4 = st.columns(
        4
    )


    with t1:

        st.metric(
            "Threshold",
            "99.10%",
        )


    with t2:

        st.metric(
            "Automation coverage",
            "99.70%",
        )


    with t3:

        st.metric(
            "Selective accuracy",
            "99.92%",
        )


    with t4:

        st.metric(
            "Human review",
            "11 / 3,612",
        )


    st.caption(
        "The confidence operating point was selected using validation-set "
        "predictions. At the selected threshold, coverage was 99.6955% "
        "and selective accuracy was 99.9167%."
    )


    st.write("")
    st.write("")


    insight1, insight2, insight3 = st.columns(
        3
    )


    with insight1:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">'
            'Predictive performance'
            '</div>'
            '<div class="info-title">'
            'Small headline gain'
            '</div>'
            '<div class="info-copy">'
            'RoBERTa improved Macro-F1 over Linear SVM by '
            'approximately 0.484 percentage points. '
            'The aggregate difference is therefore modest.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )


    with insight2:

        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">'
            'Operational risk'
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
            'Deployment trade-off'
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
        'Performance versus resource requirements'
        '</div>',
        unsafe_allow_html=True,
    )


    st.dataframe(
        efficiency,
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


    about1, about2 = st.columns(
        2
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
            'a decision layer that determines whether a prediction '
            'is suitable for automatic handling or should first be '
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


    about3, about4 = st.columns(
        2
    )


    with about3:


        st.markdown(
            '<div class="info-card">'
            '<div class="info-label">'
            'Human-in-the-loop'
            '</div>'
            '<div class="info-title">'
            'Selective automation'
            '</div>'
            '<div class="info-copy">'
            'The prototype does not assume every prediction should '
            'trigger an automated action. Lower-confidence cases are '
            'deliberately referred for human review.'
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
        "on operational queries, monitoring for distribution shift "
        "and mechanisms for queries outside the known intent taxonomy."
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
