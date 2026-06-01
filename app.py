import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Sentiment Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

/* ======================================================
GLOBAL
====================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background:
        linear-gradient(
            135deg,
            #0b1120,
            #111827,
            #172554
        );

    color: white;
}


/* ======================================================
SIDEBAR
====================================================== */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0f172a,
            #111827
        );

    border-right:
        1px solid rgba(255,255,255,0.06);
}


/* ======================================================
HERO SECTION
====================================================== */

.hero-container {
    position: relative;
    overflow: hidden;
    padding: 3rem;
    border-radius: 30px;
    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.22),
            rgba(124,58,237,0.20),
            rgba(236,72,153,0.18)
        );

    border:
        1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(16px);
    text-align: center;
    margin-bottom: 2rem;
    box-shadow:
        0px 10px 45px rgba(0,0,0,0.45);
}


/* floating glow */

.hero-container::before {
    content: "";
    position: absolute;
    width: 350px;
    height: 350px;
    background:
        radial-gradient(
            circle,
            rgba(255,255,255,0.18),
            transparent 70%
        );

    top: -120px;
    right: -120px;
    filter: blur(50px);
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    color: white;
    text-shadow:
        0px 0px 12px rgba(96,165,250,0.55);
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 1.1rem;
    margin-top: 8px;
}


/* ======================================================
TEXT AREA
====================================================== */

.stTextArea textarea {
    background:
        rgba(255,255,255,0.06);
    color: white;
    border-radius: 18px;
    border:
        1px solid rgba(255,255,255,0.08);
    font-size: 17px;
    padding: 18px;
}


/* ======================================================
BUTTON
====================================================== */

.stButton button {
    width: 100%;
    height: 62px;
    border-radius: 18px;
    border: none;
    font-size: 19px;
    font-weight: 800;
    color: white;
    background:
        linear-gradient(
            90deg,
            #2563eb,
            #7c3aed,
            #ec4899
        );

    background-size: 200% auto;
    transition:
        0.4s ease;
    box-shadow:
        0px 6px 24px rgba(124,58,237,0.35);
}


/* animated hover */

.stButton button:hover {
    background-position: right center;
    transform:
        translateY(-3px) scale(1.01);
    box-shadow:
        0px 10px 30px rgba(236,72,153,0.35);
}

/* ======================================================
PREMIUM METRIC CARDS
====================================================== */

.metric-card {
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.10),
            rgba(255,255,255,0.04)
        );
    border-radius: 28px;
    padding: 32px;
    text-align: center;
    border:
        1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(14px);
    box-shadow:
        0px 8px 35px rgba(0,0,0,0.35);
    transition:
        all 0.35s ease;
}


/* animated gradient glow */
.metric-card::before {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    background:
        radial-gradient(
            circle,
            rgba(59,130,246,0.35),
            transparent 70%
        );
    top: -120px;
    right: -120px;
    opacity: 0;
    transition: opacity 0.4s ease;
}


/* hover animation */
.metric-card:hover {
    transform:
        translateY(-10px) scale(1.03);
    border:
        1px solid rgba(96,165,250,0.35);
    box-shadow:
        0px 12px 45px rgba(59,130,246,0.25),
        0px 0px 25px rgba(168,85,247,0.18);
}


/* activate glow */
.metric-card:hover::before {
    opacity: 1;
}


/* title */
.metric-title {
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 14px;
    letter-spacing: 0.5px;
}


/* value */

.metric-value {
    font-size: 3rem;
    font-weight: 900;
    color: white;
    transition:
        transform 0.3s ease,
        text-shadow 0.3s ease;
}


/* value hover */

.metric-card:hover .metric-value {
    transform: scale(1.08);
    text-shadow:
        0px 0px 20px rgba(255,255,255,0.3),
        0px 0px 40px rgba(96,165,250,0.25);
}


/* ======================================================
DATAFRAME
====================================================== */

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}


/* ======================================================
SCROLLBAR
====================================================== */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background:
        linear-gradient(
            #3b82f6,
            #8b5cf6
        );
    border-radius: 10px;
}

</style>

""", unsafe_allow_html=True)


# ======================================================
# LOAD MODELS
# ======================================================

simple_rnn_model = load_model(
    "simple_rnn_model.h5"
)

lstm_model = load_model(
    "lstm_model.h5"
)

gru_model = load_model(
    "gru_model.h5"
)


# ======================================================
# TOKENIZER
# ======================================================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


max_length = 300


# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("⚡ AI Control Center")

selected_model = st.sidebar.selectbox(

    "Choose Model",

    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success(
    "Deep Learning Powered Sentiment Analysis"
)

st.sidebar.markdown(
    """
    ### Models Included
    
    ✅ SimpleRNN  
    ✅ LSTM  
    ✅ GRU  
    
    ### Features
    
    ✔ Real-time prediction  
    ✔ Confidence analysis  
    ✔ Interactive charts  
    ✔ Multi-model comparison  
    """
)


# ======================================================
# HERO SECTION
# ======================================================

hero_section = """
<div class="hero-container">
<h1 class="hero-title">🎬 AI Sentiment Intelligence</h1>
<p class="hero-subtitle">
Deep Learning Powered Movie Review Classification System
</p>
</div>
"""

st.markdown(
    hero_section,
    unsafe_allow_html=True
)


# ======================================================
# INPUT
# ======================================================

review = st.text_area(

    "✍ Enter your movie review",

    placeholder="Type your review here..."
)


# ======================================================
# PREDICTION FUNCTION
# ======================================================

def predict_sentiment(model, text):

    sequence = tokenizer.texts_to_sequences([text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0][0]

    positive_probability = float(prediction)

    negative_probability = float(1 - prediction)

    if prediction >= 0.5:

        sentiment = "Positive"

    else:

        sentiment = "Negative"

    confidence = max(
        positive_probability,
        negative_probability
    ) * 100

    return (
        sentiment,
        confidence,
        positive_probability,
        negative_probability
    )


# ======================================================
# BUTTON
# ======================================================

if st.button("🚀 Analyze Sentiment"):

    if review.strip() == "":

        st.warning(
            "Please enter a movie review."
        )

    else:

        if selected_model == "SimpleRNN":

            model = simple_rnn_model

        elif selected_model == "LSTM":

            model = lstm_model

        else:

            model = gru_model


        (
            sentiment,
            confidence,
            positive_probability,
            negative_probability
        ) = predict_sentiment(
            model,
            review
        )


        # ==================================================
        # METRICS
        # ==================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-title">Prediction</p>
                    <h2 class="metric-value">{sentiment}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:
                
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-title">Confidence</p>
                    <h2 class="metric-value">{confidence:.2f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-title">Selected Model</p>
                    <h2 class="metric-value">{selected_model}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ==================================================
        # CHART
        # ==================================================

        st.subheader("📊 Sentiment Analytics")

        fig = go.Figure()

        fig.add_trace(

            go.Bar(

                x=["Positive", "Negative"],

                y=[
                    positive_probability,
                    negative_probability
                ],

                text=[
                    f"{positive_probability:.2%}",
                    f"{negative_probability:.2%}"
                ],

                textposition='auto'
            )
        )

        fig.update_layout(

            template="plotly_dark",

            height=500,

            paper_bgcolor="rgba(0,0,0,0)",

            plot_bgcolor="rgba(0,0,0,0)",

            title="Prediction Confidence"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ==================================================
        # MODEL COMPARISON
        # ==================================================

        st.subheader("⚡ Multi-Model Comparison")

        models = {

            "SimpleRNN": simple_rnn_model,

            "LSTM": lstm_model,

            "GRU": gru_model
        }

        results = []

        for model_name, model_object in models.items():

            (
                result,
                conf,
                pos,
                neg
            ) = predict_sentiment(
                model_object,
                review
            )

            results.append({

                "Model": model_name,

                "Prediction": result,

                "Confidence": f"{conf:.2f}%"
            })

        comparison_df = pd.DataFrame(results)

        st.dataframe(
            comparison_df,
            use_container_width=True
        )