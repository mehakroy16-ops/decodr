import unicodedata

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Decodr",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_ID = "MehakR/decodr-roberta-intent-classifier"

AUTO_THRESHOLD = 0.991
MAX_LENGTH = 32


# ============================================================
# INTENT → BUSINESS CATEGORY
# ============================================================

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
    "track_refund": "REFUND"
}


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    /* Hide Streamlit decoration */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 2.2rem 1rem;
    }

    .brand {
        font-size: 4.6rem;
        font-weight: 800;
        letter-spacing: -4px;
        line-height: 1;
        margin-bottom: 0.7rem;
    }

    .tagline {
        font-size: 1.45rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .hero-description {
        font-size: 1rem;
        opacity: 0.65;
        max-width: 650px;
        margin: auto;
    }

    /* Small section label */
    .eyebrow {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        opacity: 0.55;
        margin-bottom: 0.35rem;
    }

    /* Result card */
    .result-card {
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        min-height: 135px;
    }

    .result-label {
        font-size: 0.78rem;
        opacity: 0.55;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.8px;
        margin-bottom: 0.55rem;
    }

    .result-value {
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .result-subtext {
        font-size: 0.86rem;
        opacity: 0.58;
        margin-top: 0.45rem;
    }

    /* Routing cards */
    .route-auto {
        border-radius: 18px;
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(31, 157, 85, 0.35);
        background: rgba(31, 157, 85, 0.08);
        margin-bottom: 1rem;
    }

    .route-review {
        border-radius: 18px;
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(230, 165, 35, 0.40);
        background: rgba(230, 165, 35, 0.09);
        margin-bottom: 1rem;
    }

    .route-title {
        font-size: 1.25rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .route-copy {
        font-size: 0.95rem;
        opacity: 0.78;
    }

    /* Alternative prediction card */
    .alternative-card {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 16px;
        padding: 1.15rem 1.3rem;
    }

    .alternative-rank {
        font-size: 0.74rem;
        opacity: 0.52;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.7px;
    }

    .alternative-name {
        font-size: 1.1rem;
        font-weight: 650;
        margin-top: 0.4rem;
    }

    .alternative-confidence {
        font-size: 0.9rem;
        opacity: 0.60;
        margin-top: 0.2rem;
    }

    /* Button */
    .stButton > button {
        border-radius: 12px;
        height: 3.1rem;
        font-weight: 650;
        font-size: 1rem;
    }

    /* Text area */
    .stTextArea textarea {
        border-radius: 14px;
        font-size: 1.05rem;
        padding: 1rem;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }

    /* Footer */
    .decodr-footer {
        text-align: center;
        opacity: 0.45;
        font-size: 0.8rem;
        padding-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD PRIVATE MODEL
# ============================================================

@st.cache_resource
def load_decodr():

    hf_token = st.secrets["HF_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=hf_token
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        token=hf_token
    )

    model.eval()

    return tokenizer, model


try:
    tokenizer, model = load_decodr()

except Exception:

    st.error(
        "Decodr could not load its classification model. "
        "Please check the private model access configuration."
    )

    st.stop()


# ============================================================
# FUNCTIONS
# ============================================================

def clean_query(text):

    text = unicodedata.normalize("NFKC", text)

    return " ".join(
        text.strip().split()
    )


def get_label(class_id):

    label = model.config.id2label.get(class_id)

    if label is None:
        label = model.config.id2label.get(str(class_id))

    return str(label)


def readable_intent(intent):

    return intent.replace("_", " ").title()


def predict_intent(query):

    query = clean_query(query)

    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    top_probs, top_ids = torch.topk(
        probabilities,
        k=3
    )

    predictions = []

    for probability, class_id in zip(
        top_probs,
        top_ids
    ):

        class_id = class_id.item()

        intent = get_label(class_id)

        predictions.append(
            {
                "intent": intent,
                "confidence": probability.item()
            }
        )

    primary = predictions[0]

    category = INTENT_TO_CATEGORY.get(
        primary["intent"],
        "UNKNOWN"
    )

    return {
        "intent": primary["intent"],
        "category": category,
        "confidence": primary["confidence"],
        "predictions": predictions
    }


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="brand">
            Decodr
        </div>

        <div class="tagline">
            Decode customer intent. Route smarter.
        </div>

        <div class="hero-description">
            AI-assisted customer-service intent classification
            with confidence-based automation and human escalation.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

classifier_tab, insights_tab, about_tab = st.tabs(
    [
        "◆ Try Decodr",
        "Model Insights",
        "About"
    ]
)


# ============================================================
# TAB 1 — CLASSIFIER
# ============================================================

with classifier_tab:

    st.write("")

    st.markdown(
        '<div class="eyebrow">Customer Query</div>',
        unsafe_allow_html=True
    )

    st.header(
        "What does the customer need?"
    )

    st.caption(
        "Enter a customer-service request in natural language."
    )

    query = st.text_area(
        "Customer query",
        placeholder=(
            "e.g. I placed an order yesterday "
            "but I need to cancel it."
        ),
        height=150,
        label_visibility="collapsed"
    )

    decode_button = st.button(
        "Decode Intent →",
        type="primary",
        use_container_width=True
    )

    if decode_button:

        if not query.strip():

            st.warning(
                "Enter a customer query first."
            )

        else:

            with st.spinner(
                "Decoding customer intent..."
            ):

                result = predict_intent(query)

            st.write("")
            st.divider()
            st.write("")

            # ------------------------------------------------
            # CLASSIFICATION RESULT
            # ------------------------------------------------

            st.markdown(
                '<div class="eyebrow">Classification Result</div>',
                unsafe_allow_html=True
            )

            st.header(
                "Decodr's interpretation"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-label">
                            Predicted Intent
                        </div>

                        <div class="result-value">
                            {readable_intent(result["intent"])}
                        </div>

                        <div class="result-subtext">
                            Most likely customer need
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-label">
                            Business Category
                        </div>

                        <div class="result-value">
                            {result["category"]}
                        </div>

                        <div class="result-subtext">
                            Suggested workflow
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c3:

                st.markdown(
                    f"""
                    <div class="result-card">

                        <div class="result-label">
                            Model Confidence
                        </div>

                        <div class="result-value">
                            {result["confidence"]:.2%}
                        </div>

                        <div class="result-subtext">
                            Routing threshold: {AUTO_THRESHOLD:.2%}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # Confidence progress bar
            st.progress(
                result["confidence"],
                text=(
                    f'Prediction confidence: '
                    f'{result["confidence"]:.2%}'
                )
            )

            st.write("")
            st.write("")

            # ------------------------------------------------
            # ROUTING DECISION
            # ------------------------------------------------

            st.markdown(
                '<div class="eyebrow">Routing Decision</div>',
                unsafe_allow_html=True
            )

            if result["confidence"] >= AUTO_THRESHOLD:

                st.markdown(
                    f"""
                    <div class="route-auto">

                        <div class="route-title">
                            ✓ Automatic routing recommended
                        </div>

                        <div class="route-copy">
                            Confidence of
                            <strong>{result["confidence"]:.2%}</strong>
                            exceeds the
                            <strong>{AUTO_THRESHOLD:.2%}</strong>
                            operating threshold.

                            This query can be routed to the
                            <strong>{result["category"]}</strong>
                            workflow.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="route-review">

                        <div class="route-title">
                            ⚠ Human review recommended
                        </div>

                        <div class="route-copy">
                            Confidence of
                            <strong>{result["confidence"]:.2%}</strong>
                            is below the
                            <strong>{AUTO_THRESHOLD:.2%}</strong>
                            operating threshold.

                            The prediction should therefore be
                            reviewed before the query is routed.
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # ------------------------------------------------
            # ALTERNATIVE PREDICTIONS
            # ------------------------------------------------

            st.markdown(
                '<div class="eyebrow">Alternative Interpretations</div>',
                unsafe_allow_html=True
            )

            st.subheader(
                "Other possibilities considered"
            )

            alt1 = result["predictions"][1]
            alt2 = result["predictions"][2]

            a1, a2 = st.columns(2)

            with a1:

                st.markdown(
                    f"""
                    <div class="alternative-card">

                        <div class="alternative-rank">
                            Second most likely
                        </div>

                        <div class="alternative-name">
                            {readable_intent(alt1["intent"])}
                        </div>

                        <div class="alternative-confidence">
                            {alt1["confidence"]:.2%} confidence
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with a2:

                st.markdown(
                    f"""
                    <div class="alternative-card">

                        <div class="alternative-rank">
                            Third most likely
                        </div>

                        <div class="alternative-name">
                            {readable_intent(alt2["intent"])}
                        </div>

                        <div class="alternative-confidence">
                            {alt2["confidence"]:.2%} confidence
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # ------------------------------------------------
            # EXPLANATION
            # ------------------------------------------------

            with st.expander(
                "How does Decodr make this decision?"
            ):

                st.markdown(
                    f"""
                    Decodr uses the fine-tuned **RoBERTa**
                    sequence-classification model evaluated in
                    the dissertation.

                    The customer's text is classified into one of
                    **27 customer-service intents**.

                    The intent with the highest model probability
                    becomes the primary prediction.

                    Decodr then compares that confidence with the
                    validation-selected threshold of
                    **{AUTO_THRESHOLD:.2%}**.

                    **≥ {AUTO_THRESHOLD:.2%}**
                    → eligible for automatic routing

                    **< {AUTO_THRESHOLD:.2%}**
                    → human review recommended
                    """
                )


# ============================================================
# TAB 2 — MODEL INSIGHTS
# ============================================================

with insights_tab:

    st.write("")

    st.markdown(
        '<div class="eyebrow">Research Results</div>',
        unsafe_allow_html=True
    )

    st.header(
        "Model performance"
    )

    st.caption(
        "Held-out evaluation across five classification models."
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "RoBERTa Macro-F1",
            "0.99898"
        )

    with m2:
        st.metric(
            "Test Errors",
            "3 / 3,645"
        )

    with m3:
        st.metric(
            "High-Risk Errors",
            "1"
        )

    with m4:
        st.metric(
            "Best Model",
            "RoBERTa"
        )

    st.write("")
    st.divider()
    st.write("")

    results = pd.DataFrame(
        {
            "Model": [
                "RoBERTa",
                "DistilBERT",
                "Linear SVM",
                "Logistic Regression",
                "Multinomial Naive Bayes"
            ],
            "Macro-F1": [
                0.998982,
                0.997925,
                0.994143,
                0.992812,
                0.992262
            ],
            "Test Errors": [
                3,
                7,
                20,
                25,
                26
            ],
            "High-Risk Errors": [
                1,
                4,
                14,
                17,
                17
            ]
        }
    )

    st.subheader(
        "Comparative performance"
    )

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True
    )

    st.write("")

    st.subheader(
        "Held-out test errors"
    )

    chart_data = (
        results[
            ["Model", "Test Errors"]
        ]
        .set_index("Model")
    )

    st.bar_chart(
        chart_data
    )

    st.write("")
    st.divider()
    st.write("")

    st.markdown(
        '<div class="eyebrow">Confidence-Based Routing</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Validation-selected operating point"
    )

    x1, x2, x3, x4 = st.columns(4)

    with x1:
        st.metric(
            "Threshold",
            "99.10%"
        )

    with x2:
        st.metric(
            "Automation Coverage",
            "99.70%"
        )

    with x3:
        st.metric(
            "Selective Accuracy",
            "99.92%"
        )

    with x4:
        st.metric(
            "Human Review",
            "11 / 3,612"
        )

    st.caption(
        "Confidence operating-point statistics are based "
        "on validation-set analysis rather than the final "
        "held-out test set."
    )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with about_tab:

    st.write("")

    st.markdown(
        '<div class="eyebrow">Research Prototype</div>',
        unsafe_allow_html=True
    )

    st.header(
        "About Decodr"
    )

    st.markdown(
        """
        **Decodr** is an AI-assisted customer-service intent
        classification prototype developed as part of an
        MSc Business Analytics dissertation.

        It demonstrates how Natural Language Processing can
        support the first stage of customer-service automation:
        understanding what a customer wants and deciding whether
        the query can be routed automatically or should be
        reviewed by a human.
        """
    )

    st.write("")

    st.subheader(
        "What Decodr demonstrates"
    )

    st.markdown(
        """
        **27 customer-service intents**  
        Fine-grained prediction of the customer's request.

        **11 business categories**  
        Broader operational grouping for routing.

        **Prediction confidence**  
        An indication of model certainty.

        **Alternative interpretations**  
        The next two most likely intents considered by the model.

        **Confidence-based escalation**  
        Lower-confidence queries are sent for human review
        rather than automatically routed.
        """
    )

    st.write("")

    st.subheader(
        "Research model"
    )

    st.markdown(
        """
        The application uses the final fine-tuned **RoBERTa**
        classifier evaluated in the dissertation.

        Only the customer's query is provided to the
        classification model. Business category and routing
        information are added after the intent prediction.
        """
    )

    st.write("")

    st.subheader(
        "Scope"
    )

    st.warning(
        "Decodr is an academic proof of concept, not a live "
        "customer-service deployment. Its results demonstrate "
        "technical feasibility on the dissertation dataset and "
        "should not be interpreted as production performance "
        "without evaluation on real operational customer queries."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="decodr-footer">
        Decodr · MSc Business Analytics research prototype
    </div>
    """,
    unsafe_allow_html=True
)



       

    
