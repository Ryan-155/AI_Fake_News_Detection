import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



# ==============================
# 🔥 LOAD ALL MODELS (CACHED)
# ==============================
@st.cache_resource
def load_models():

    # ---------- CORE FILES ----------
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("corpus.pkl", "rb") as f:
        corpus = pickle.load(f)

    corpus_embeddings = np.load("embeddings.npy")

    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    # ---------- EMBEDDING MODEL ----------
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # ---------- NLI MODEL ----------
    from transformers import pipeline
    nli_model = pipeline(
        "text-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )

    # ---------- LLM (LIGHTWEIGHT FOR STREAMLIT) ----------
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    llm_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)

    return (
        vectorizer,
        corpus,
        corpus_embeddings,
        config,
        embed_model,
        nli_model,
        tokenizer,
        llm
    )


vectorizer, corpus, corpus_embeddings, config, embed_model, nli_model, tokenizer, llm = load_models()

# ==============================
# 🔍 RETRIEVAL MODULE
# ==============================
def retrieve(claim, k=5):

    claim_vec = vectorizer.transform([claim])
    tfidf_scores = cosine_similarity(claim_vec, vectorizer.transform(corpus))[0]

    claim_emb = embed_model.encode([claim])
    semantic_scores = cosine_similarity(claim_emb, corpus_embeddings)[0]

    scores = tfidf_scores + semantic_scores
    top_k_idx = scores.argsort()[-k:][::-1]

    return [corpus[i] for i in top_k_idx]


# ==============================
# 🧠 CREDIBILITY SCORE
# ==============================
def credibility_score(evidence):
    score = 0.6
    text = evidence.lower()

    if "wikipedia" in text:
        score += 0.3
    if "study" in text or "research" in text:
        score += 0.1
    if len(text) > 200:
        score += 0.05

    return min(score, 1.0)


# ==============================
# 🧪 PREDICTION ENGINE
# ==============================
def predict(claim):

    evidences = retrieve(claim)

    final_score = 0
    details = []

    for ev in evidences:

        result = nli_model(f"{claim} </s></s> {ev}")[0]

        label = result["label"]
        confidence = result["score"]

        if label == "ENTAILMENT":
            n = 1
        elif label == "CONTRADICTION":
            n = -1
        else:
            n = 0

        cred = credibility_score(ev)

        weighted_score = n * confidence * cred
        final_score += weighted_score

        details.append(ev)

    if final_score > 0.5:
        verdict = "REAL"
    elif final_score < -0.5:
        verdict = "FAKE"
    else:
        verdict = "UNCERTAIN"

    return verdict, final_score, details


# ==============================
# 🧾 LLM EXPLANATION MODULE
# ==============================
def generate_explanation(claim, verdict, evidences):

    context = "\n".join(evidences[:3])

    prompt = f"""
You are a fact-checking assistant.

Claim: {claim}
Verdict: {verdict}

Evidence:
{context}

Explain in simple terms why this claim is classified this way.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    output = llm.generate(**inputs, max_new_tokens=120)

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ==============================
# 🌐 STREAMLIT UI
# ==============================
st.title("🧠 AI Fake News Detection System")

claim = st.text_input("Enter a news claim:")

if st.button("Analyze"):

    verdict, score, evidences = predict(claim)

    explanation = generate_explanation(claim, verdict, evidences)

    st.subheader("Prediction")
    st.write(verdict)

    st.subheader("Confidence Score")
    st.write(score)

    st.subheader("Evidence Used")
    for e in evidences:
        st.write("-", e)

    st.subheader("AI Explanation")
    st.write(explanation)
