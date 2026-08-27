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
    layout="wide"
)


# ============================================================
# DECODR SETTINGS
# ============================================================

MODEL_ID = "MehakR/decodr-roberta-intent-classifier"

# Validation-selected confidence threshold from dissertation
AUTO_THRESHOLD = 0.991

# Final RoBERTa sequence length
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
# SMALL AMOUNT OF CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1150px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    .decodr-title {
        font-size: 4rem;
        font-weight: 750;
        letter-spacing: -2px;
        margin-bottom: 0;
    }

    .decodr-subtitle {
        font-size: 1.25rem;
        opacity: 0.75;
        margin-top: -8px;
        margin-bottom: 25px;
    }

    .section-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.60;
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

except Exception as e:

    st.error(
        "Decodr could not load its classification model. "
        "Please check the Hugging Face model access configuration."
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_query(text):

    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.strip().split())

    return text


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

    for prob, class_id in zip(top_probs, top_ids):

        class_id = class_id.item()

        intent = get_label(class_id)

        predictions.append(
            {
                "intent": intent,
                "confidence": prob.item()
            }
        )

    main_prediction = predictions[0]

    category = INTENT_TO_CATEGORY.get(
        main_prediction["intent"],
        "UNKNOWN"
    )

    return {
        "intent": main_prediction["intent"],
        "category": category,
        "confidence": main_prediction["confidence"],
        "predictions": predictions
    }


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="decodr-title">Decodr</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="decodr-subtitle">'
    'Decode customer intent. Route smarter.'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "AI-assisted customer-service intent classification "
    "and confidence-based routing."
)

st.divider()


# ============================================================
# TABS
# ============================================================

classifier_tab, insights_tab, about_tab = st.tabs(
    [
        "Try Decodr",
        "Model Insights",
        "About"
    ]
)


# ============================================================
# TAB 1 — LIVE CLASSIFIER
# ============================================================

with classifier_tab:

    st.markdown(
        '<p class="section-label">Customer Query</p>',
        unsafe_allow_html=True
    )

    st.subheader(
        "What does the customer need?"
    )

    query = st.text_area(
        "Enter a customer-service query",
        placeholder=(
            "Example: I placed an order yesterday "
            "but I need to cancel it."
        ),
        height=150,
        label_visibility="collapsed"
    )

    decode_button = st.button(
        "Decode Intent",
        type="primary",
        use_container_width=True
    )

    if decode_button:

        if not query.strip():

            st.warning(
                "Enter a customer query before running Decodr."
            )

        else:

            result = predict_intent(query)

            st.divider()

            st.markdown(
                '<p class="section-label">Classification Result</p>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Intent",
                    readable_intent(result["intent"])
                )

            with col2:

                st.metric(
                    "Business Category",
                    result["category"]
                )

            with col3:

                st.metric(
                    "Model Confidence",
                    f'{result["confidence"]:.2%}'
                )

            st.write("")

            # --------------------------------------------
            # ROUTING DECISION
            # --------------------------------------------

            st.markdown(
                "### Routing Decision"
            )

            if result["confidence"] >= AUTO_THRESHOLD:

                st.success(
                    "✓ Automatic routing recommended"
                )

                st.write(
                    f"This query can be routed to the "
                    f"**{result['category']}** workflow."
                )

                st.caption(
                    f"Confidence exceeds the "
                    f"{AUTO_THRESHOLD:.3f} validation-selected "
                    f"operating threshold."
                )

            else:

                st.warning(
                    "⚠ Human review recommended"
                )

                st.write(
                    "The model's confidence is below the "
                    "selected operating threshold, so the query "
                    "should be reviewed before routing."
                )

                st.caption(
                    f"Current confidence: "
                    f"{result['confidence']:.2%} | "
                    f"Threshold: {AUTO_THRESHOLD:.3f}"
                )

            # --------------------------------------------
            # ALTERNATIVE PREDICTIONS
            # --------------------------------------------

            st.markdown(
                "### Alternative Interpretations"
            )

            alternatives = pd.DataFrame(
                [
                    {
                        "Intent": readable_intent(pred["intent"]),
                        "Confidence": f'{pred["confidence"]:.2%}'
                    }
                    for pred in result["predictions"][1:]
                ]
            )

            st.dataframe(
                alternatives,
                hide_index=True,
                use_container_width=True
            )

            with st.expander(
                "How does Decodr make this decision?"
            ):

                st.write(
                    """
                    Decodr uses a fine-tuned RoBERTa
                    sequence-classification model to assign the
                    customer query to one of 27 customer-service
                    intents.

                    The model's highest probability determines
                    the predicted intent. The prediction is then
                    compared with the confidence threshold
                    selected during validation.

                    Predictions meeting the threshold are treated
                    as candidates for automatic routing. Lower
                    confidence predictions are referred for human
                    review.
                    """
                )


# ============================================================
# TAB 2 — MODEL INSIGHTS
# ============================================================

with insights_tab:

    st.markdown(
        '<p class="section-label">Dissertation Results</p>',
        unsafe_allow_html=True
    )

    st.subheader(
        "How did the models perform?"
    )

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

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "### Held-Out Test Errors"
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

    st.markdown(
        "### Confidence-Based Routing"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Threshold",
            "0.991"
        )

    with c2:

        st.metric(
            "Automation Coverage",
            "99.70%"
        )

    with c3:

        st.metric(
            "Selective Accuracy",
            "99.92%"
        )

    with c4:

        st.metric(
            "Human Review",
            "11 / 3,612"
        )

    st.caption(
        "Confidence-based operating-point results were "
        "obtained from the validation set."
    )

    st.info(
        "RoBERTa achieved the highest predictive performance, "
        "but the dissertation also evaluates computational "
        "cost, model size and business-risk errors when "
        "considering practical deployment."
    )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with about_tab:

    st.markdown(
        '<p class="section-label">Research Prototype</p>',
        unsafe_allow_html=True
    )

    st.subheader(
        "About Decodr"
    )

    st.write(
        """
        **Decodr** is an academic proof-of-concept developed
        as part of an MSc Business Analytics dissertation.

        The application demonstrates how Natural Language
        Processing can be used to classify customer-service
        queries and support automated routing decisions.
        """
    )

    st.markdown(
        "### What Decodr demonstrates"
    )

    st.write(
        """
        - Classification across **27 customer-service intents**
        - Mapping to **11 broader business categories**
        - Prediction confidence
        - Alternative likely interpretations
        - Confidence-based automatic routing
        - Human escalation for lower-confidence queries
        """
    )

    st.markdown(
        "### Model"
    )

    st.write(
        """
        The demonstration uses the fine-tuned **RoBERTa**
        model evaluated in the dissertation.

        The model receives only the customer query as its
        classification input. Category labels and routing
        recommendations are generated after prediction.
        """
    )

    st.markdown(
        "### Important limitation"
    )

    st.warning(
        "Decodr is a research demonstration rather than a "
        "production customer-service system. The model was "
        "evaluated using the dissertation dataset and would "
        "require further testing on real operational queries "
        "before deployment."
    )

    st.caption(
        "Queries entered into this prototype are used for "
        "inference and are not intentionally stored by the "
        "application."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Decodr • MSc Business Analytics research prototype"
)
