import unicodedata

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# PAGE CONFIGURATION
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

# Validation-selected operating threshold
AUTO_THRESHOLD = 0.991

# Maximum sequence length used by final RoBERTa model
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
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
<style>
.block-container {
    max-width: 1100px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

h1 {
    font-size: 4.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -3px !important;
    text-align: center;
    margin-bottom: 0.25rem !important;
}

.hero-tagline {
    text-align: center;
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.hero-description {
    text-align: center;
    font-size: 1rem;
    opacity: 0.65;
    max-width: 720px;
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 2rem;
}

.section-label {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    opacity: 0.55;
    margin-bottom: -0.4rem;
}

div[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.18);
    border-radius: 16px;
    padding: 18px 20px;
}

div[data-testid="stMetricLabel"] {
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    font-size: 1.55rem;
}

.stButton > button {
    height: 3.2rem;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 650;
}

.stTextArea textarea {
    border-radius: 14px;
    font-size: 1.05rem;
    padding: 1rem;
}

button[data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 600;
}

.footer-text {
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
# LOAD PRIVATE HUGGING FACE MODEL
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
        "Please check the private Hugging Face access configuration."
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_query(text):

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = " ".join(
        text.strip().split()
    )

    return text


def get_label(class_id):

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

    query = clean_query(query)

    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
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
        k=3
    )

    predictions = []

    for probability, class_id in zip(
        top_probs,
        top_ids
    ):

        class_id = class_id.item()

        intent = get_label(
            class_id
        )

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

st.title("Decodr")

st.markdown(
    '<div class="hero-tagline">'
    'Decode customer intent. Route smarter.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-description">'
    'AI-assisted customer-service intent classification '
    'with confidence-based automation and human escalation.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN TABS
# ============================================================

classifier_tab, insights_tab, about_tab = st.tabs(
    [
        "◆ Try Decodr",
        "Model Insights",
        "About"
    ]
)


# ============================================================
# TAB 1 — TRY DECODR
# ============================================================

with classifier_tab:

    st.write("")

    st.markdown(
        '<p class="section-label">'
        'Customer Query'
        '</p>',
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
                "Please enter a customer query first."
            )

        else:

            with st.spinner(
                "Decoding customer intent..."
            ):

                result = predict_intent(
                    query
                )

            st.write("")
            st.divider()
            st.write("")

            # =================================================
            # CLASSIFICATION RESULT
            # =================================================

            st.markdown(
                '<p class="section-label">'
                'Classification Result'
                '</p>',
                unsafe_allow_html=True
            )

            st.header(
                "Decodr's interpretation"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Intent",
                    readable_intent(
                        result["intent"]
                    )
                )

                st.caption(
                    "Most likely customer need"
                )

            with col2:

                st.metric(
                    "Business Category",
                    result["category"]
                )

                st.caption(
                    "Suggested operational workflow"
                )

            with col3:

                st.metric(
                    "Model Confidence",
                    f'{result["confidence"]:.2%}'
                )

                st.caption(
                    f'Routing threshold: '
                    f'{AUTO_THRESHOLD:.2%}'
                )

            st.write("")

            # Confidence bar

            st.progress(
                float(result["confidence"]),
                text=(
                    f'Prediction confidence: '
                    f'{result["confidence"]:.2%}'
                )
            )

            st.write("")
            st.write("")

            # =================================================
            # ROUTING DECISION
            # =================================================

            st.markdown(
                '<p class="section-label">'
                'Routing Decision'
                '</p>',
                unsafe_allow_html=True
            )

            st.subheader(
                "Recommended next action"
            )

            if result["confidence"] >= AUTO_THRESHOLD:

                st.success(
                    "✓ Automatic routing recommended"
                )

                st.markdown(
                    f"""
**Confidence:** {result["confidence"]:.2%}  
**Required threshold:** {AUTO_THRESHOLD:.2%}

The prediction exceeds the selected operating threshold.

This query can therefore be routed to the **{result["category"]}** workflow.
"""
                )

            else:

                st.warning(
                    "⚠ Human review recommended"
                )

                st.markdown(
                    f"""
**Confidence:** {result["confidence"]:.2%}  
**Required threshold:** {AUTO_THRESHOLD:.2%}

The prediction does not meet the selected operating threshold.

The query should therefore be **reviewed by a human before routing**.
"""
                )

            st.write("")
            st.write("")

            # =================================================
            # ALTERNATIVE INTERPRETATIONS
            # =================================================

            st.markdown(
                '<p class="section-label">'
                'Alternative Interpretations'
                '</p>',
                unsafe_allow_html=True
            )

            st.subheader(
                "Other possibilities considered"
            )

            st.caption(
                "The next two highest-probability intents produced by the model."
            )

            alt1 = result["predictions"][1]
            alt2 = result["predictions"][2]

            alt_col1, alt_col2 = st.columns(2)

            with alt_col1:

                with st.container(
                    border=True
                ):

                    st.caption(
                        "SECOND MOST LIKELY"
                    )

                    st.subheader(
                        readable_intent(
                            alt1["intent"]
                        )
                    )

                    st.metric(
                        "Confidence",
                        f'{alt1["confidence"]:.2%}'
                    )

            with alt_col2:

                with st.container(
                    border=True
                ):

                    st.caption(
                        "THIRD MOST LIKELY"
                    )

                    st.subheader(
                        readable_intent(
                            alt2["intent"]
                        )
                    )

                    st.metric(
                        "Confidence",
                        f'{alt2["confidence"]:.2%}'
                    )

            st.write("")

            # =================================================
            # EXPLANATION
            # =================================================

            with st.expander(
                "How does Decodr make this decision?"
            ):

                st.markdown(
                    f"""
Decodr uses the fine-tuned **RoBERTa sequence-classification model**
evaluated in the dissertation.

A customer query is classified into one of **27 customer-service intents**.

The intent receiving the highest model probability becomes the primary
prediction.

Decodr then compares the confidence of that prediction with the
validation-selected operating threshold of **{AUTO_THRESHOLD:.2%}**.

**Confidence ≥ {AUTO_THRESHOLD:.2%}**  
→ Eligible for automatic routing.

**Confidence < {AUTO_THRESHOLD:.2%}**  
→ Human review recommended.

The broader business category is assigned only after the intent prediction
and is not supplied to RoBERTa as an input.
"""
                )


# ============================================================
# TAB 2 — MODEL INSIGHTS
# ============================================================

with insights_tab:

    st.write("")

    st.markdown(
        '<p class="section-label">'
        'Dissertation Results'
        '</p>',
        unsafe_allow_html=True
    )

    st.header(
        "Model performance"
    )

    st.caption(
        "Performance on the 3,645-query held-out test set."
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        st.metric(
            "RoBERTa Macro-F1",
            "0.99898"
        )

    with metric2:

        st.metric(
            "Test Errors",
            "3 / 3,645"
        )

    with metric3:

        st.metric(
            "High-Risk Errors",
            "1"
        )

    with metric4:

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
        "Held-out test misclassifications"
    )

    error_data = (
        results[
            ["Model", "Test Errors"]
        ]
        .set_index("Model")
    )

    st.bar_chart(
        error_data
    )

    st.write("")
    st.divider()
    st.write("")

    # ========================================================
    # CONFIDENCE ROUTING
    # ========================================================

    st.markdown(
        '<p class="section-label">'
        'Confidence-Based Routing'
        '</p>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Validation-selected operating point"
    )

    route1, route2, route3, route4 = st.columns(4)

    with route1:

        st.metric(
            "Threshold",
            "99.10%"
        )

    with route2:

        st.metric(
            "Automation Coverage",
            "99.70%"
        )

    with route3:

        st.metric(
            "Selective Accuracy",
            "99.92%"
        )

    with route4:

        st.metric(
            "Human Review",
            "11 / 3,612"
        )

    st.caption(
        "Confidence-based operating-point statistics are based "
        "on the validation set rather than the final held-out test set."
    )

    st.info(
        "The highest-performing model is not evaluated solely "
        "through headline predictive performance. The dissertation "
        "also examines error severity, computational requirements "
        "and confidence-based escalation."
    )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with about_tab:

    st.write("")

    st.markdown(
        '<p class="section-label">'
        'Research Prototype'
        '</p>',
        unsafe_allow_html=True
    )

    st.header(
        "About Decodr"
    )

    st.markdown(
        """
**Decodr** is an AI-assisted customer-service intent classification
prototype developed as part of an MSc Business Analytics dissertation.

The application demonstrates how Natural Language Processing can support
the first stage of customer-service automation: understanding what a
customer wants and determining whether the request can be routed
automatically or should first be reviewed by a human.
"""
    )

    st.write("")
    st.divider()
    st.write("")

    st.subheader(
        "What Decodr demonstrates"
    )

    feature1, feature2 = st.columns(2)

    with feature1:

        with st.container(
            border=True
        ):

            st.subheader(
                "27 intents"
            )

            st.write(
                "Fine-grained classification of the customer's request."
            )

        with st.container(
            border=True
        ):

            st.subheader(
                "Prediction confidence"
            )

            st.write(
                "Model certainty is displayed for each classification."
            )

        with st.container(
            border=True
        ):

            st.subheader(
                "Alternative interpretations"
            )

            st.write(
                "The next two highest-probability intents are displayed."
            )

    with feature2:

        with st.container(
            border=True
        ):

            st.subheader(
                "11 categories"
            )

            st.write(
                "Intent predictions are mapped to broader business workflows."
            )

        with st.container(
            border=True
        ):

            st.subheader(
                "Automatic routing"
            )

            st.write(
                "High-confidence predictions can be treated as routing candidates."
            )

        with st.container(
            border=True
        ):

            st.subheader(
                "Human escalation"
            )

            st.write(
                "Lower-confidence predictions are referred for review."
            )

    st.write("")
    st.divider()
    st.write("")

    st.subheader(
        "Research model"
    )

    st.markdown(
        """
Decodr uses the final fine-tuned **RoBERTa** classifier evaluated in
the dissertation.

Only the customer's query is provided to the model as an input.

The predicted business category and routing recommendation are generated
after the intent classification has been produced.
"""
    )

    st.write("")

    st.subheader(
        "Scope and limitations"
    )

    st.warning(
        "Decodr is an academic proof of concept rather than a live "
        "customer-service deployment. Its results demonstrate technical "
        "feasibility on the dissertation dataset and should not be "
        "interpreted as production performance without evaluation on "
        "real operational customer queries."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer-text">'
    'Decodr · MSc Business Analytics research prototype'
    '</div>',
    unsafe_allow_html=True
)
