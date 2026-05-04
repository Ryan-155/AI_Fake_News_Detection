import gradio as gr
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD MODELS
# =========================
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
corpus = pickle.load(open("corpus.pkl", "rb"))
corpus_embeddings = np.load("embeddings.npy")

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

nli_model = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

llm_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)


# =========================
# RETRIEVAL
# =========================
def retrieve(claim, k=5):
    claim_vec = vectorizer.transform([claim])
    tfidf_scores = cosine_similarity(claim_vec, vectorizer.transform(corpus))[0]

    claim_emb = embed_model.encode([claim])
    sem_scores = cosine_similarity(claim_emb, corpus_embeddings)[0]

    scores = tfidf_scores + sem_scores
    top_k = scores.argsort()[-k:][::-1]

    return [corpus[i] for i in top_k]


# =========================
# CREDIBILITY
# =========================
def credibility(text):
    score = 0.6
    t = text.lower()

    if "wikipedia" in t:
        score += 0.3
    if "study" in t or "research" in t:
        score += 0.1

    return min(score, 1.0)


# =========================
# PREDICTION
# =========================
def predict(claim):

    evidences = retrieve(claim)

    final_score = 0
    for ev in evidences:

        result = nli_model(f"{claim} </s></s> {ev}")[0]

        label = result["label"]
        conf = result["score"]

        if label == "ENTAILMENT":
            s = 1
        elif label == "CONTRADICTION":
            s = -1
        else:
            s = 0

        final_score += s * conf * credibility(ev)

    if final_score > 0.5:
        verdict = "REAL"
    elif final_score < -0.5:
        verdict = "FAKE"
    else:
        verdict = "UNCERTAIN"

    return verdict, final_score, "\n\n".join(evidences)


# =========================
# LLM EXPLANATION
# =========================
def explain(claim):

    verdict, score, evidences = predict(claim)

    prompt = f"""
Claim: {claim}
Verdict: {verdict}

Evidence:
{evidences}

Explain why this is classified as {verdict}.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = llm.generate(**inputs, max_new_tokens=120)

    explanation = tokenizer.decode(output[0], skip_special_tokens=True)

    return verdict, score, evidences, explanation


# =========================
# GRADIO UI
# =========================
demo = gr.Interface(
    fn=explain,
    inputs=gr.Textbox(label="Enter News Claim"),
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Number(label="Score"),
        gr.Textbox(label="Evidence"),
        gr.Textbox(label="AI Explanation")
    ],
    title="🧠 AI Fake News Detection System",
    description="Retrieval + NLI + Credibility + LLM Explanation"
)

demo.launch()
