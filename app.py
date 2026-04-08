import streamlit as st
import pandas as pd
from transformers.pipelines import pipeline

st.title("Определение категории/тега научной статьи по названию и аннотации")

CATEGORIES = [
    "Computer Science",
    "Algorithms and data structures",
    "Mathematics",
    "Artificial Intelligence",
    "Machine Learning",
    "Data Structures and Algorithms",
    "Computational Complexity",
    "Discrete Mathematics",
    "Commutative Algebra",
    "Algebraic Geometry",
    "Number Theory",
    "Mathematical Physics",
    "Probability Theory",
    "Optimization and Control",
    "Quantum Physics",
    "Condensed Matter Physics",
    "Astrophysics",
    "Biology",
    "Genomics",
    "Neuroscience",
    "Economics",
    "Statistics",
    "Cryptography",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Other",
]

@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_text(text, categories):
    classifier = load_model()
    return classifier(text, categories)

title = st.text_input("Название статьи", placeholder="Например: Equal-Subset-Sum Faster Than the Meet-in-the-Middle")
use_abstract = st.checkbox("Использовать аннотацию (abstract) для предсказания", value=False)
abstract = st.text_area("", placeholder="""In the Equal-Subset-Sum problem, we are given a set S of N integers and the problem is to decide if there exist two disjoint nonempty subsets of S, whose elements sum up to the same value. The problem is NP-complete. 
""", height=200, disabled=not use_abstract, label_visibility="collapsed")

st.divider()
st.caption("Модель: facebook/bart-large-mnli (zero-shot классификация)")

if st.button("Классифицировать", type="primary"):
    if not title:
        st.error("\"Задачу с условием может решить любой дурак. А ты попробуй реши задачу без условия!\" (Заполните хотя бы название статьи)")
    else:
        with st.spinner("Выпекаем токены для ответа"):
            if use_abstract and abstract:
                text_to_classify = f"{title}. {abstract}"
            else:
                text_to_classify = title

            result = classify_text(text_to_classify, CATEGORIES)

            data = {
                "label": result["labels"],
                "score": result["scores"]
            }
            df = pd.DataFrame(data)
            df = df.sort_values("score", ascending=False)

            confidence = 0
            df95 = pd.DataFrame()
            for _, row in df.iterrows():
                confidence += row['score']
                df95 = pd.concat([df95, row.to_frame().T])
                if confidence >= 0.95:
                    break

            score_threshold = 0.01
            st.subheader("Уверенность модели (топ-95%)")
            for _, row in df95.iterrows():
                score = float(row['score'])
                if score < score_threshold:
                    break
                display_text = f"{row['label']} ({score:.1%})"
                st.progress(score, text=display_text)