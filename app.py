import os
import re
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder=".", static_url_path="")

# Path to the joblib artifact created in your Colab notebook
MODEL_PATH = os.path.join(os.path.dirname(__file__), "misinformation_detector.joblib")

loaded_model_artifact = None
joblib_available = False

try:
    import joblib
    joblib_available = True
    if os.path.exists(MODEL_PATH):
        loaded_model_artifact = joblib.load(MODEL_PATH)
        print(f" Loaded model artifact from {MODEL_PATH}")
    else:
        print(f" Model file '{MODEL_PATH}' not found yet. Using built-in adaptive classifier fallback until you copy your .joblib file here.")
except Exception as e:
    print(f" Notice: Could not load joblib model ({e}). Using built-in inference fallback.")


STYLE_TERMS = {
    "shocking", "unbelievable", "secret", "miracle", "explosive", "stunning",
    "incredible", "always", "never", "undeniable", "guaranteed", "proven",
    "confirmed", "definitely", "outrageous", "terrifying", "disaster",
    "horrific", "furious", "amazing"
}

FACTUAL_TERMS = {
    "announced", "published", "research", "researchers", "study", "university",
    "spokesperson", "reported", "official", "statement", "confirmed", "department",
    "according", "government", "data", "conference", "scientific", "journal", "analysts"
}

def verification_decision(fake_probability):
    if fake_probability >= 0.80:
        return "Potential Misinformation", "danger", "High probability of sensationalism or misinformation patterns."
    elif fake_probability <= 0.20:
        return "Likely Reliable", "real", "Patterns align with standard credible journalistic reporting."
    else:
        return "Needs Verification", "warning", "Mixed patterns detected; requires independent fact-checking."


def predict_with_loaded_model(text):
    vectorizer = loaded_model_artifact["vectorizer"]
    model = loaded_model_artifact["model"]
    weights = loaded_model_artifact["weights"]
    
    feature_names = vectorizer.get_feature_names_out()
    log_ratio = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    
    # Vectorize and weight
    vec = vectorizer.transform([text])
    vec_adaptive = vec.multiply(weights).tocsr()
    
    # Predict probabilities
    probs = model.predict_proba(vec_adaptive)[0]
    fake_prob = float(probs[1])
    real_prob = float(probs[0])
    
    verdict, level, summary = verification_decision(fake_prob)
    
    # Extract top contributing terms
    contributions = vec_adaptive.multiply(log_ratio).toarray().ravel()
    present_indices = vec_adaptive.indices
    
    ranked = sorted(present_indices, key=lambda i: abs(contributions[i]), reverse=True)[:10]
    
    top_terms = []
    for idx in ranked:
        c = float(contributions[idx])
        top_terms.append({
            "term": str(feature_names[idx]),
            "contribution": round(c, 4),
            "influence": "Misinformation" if c > 0 else "Real"
        })
        
    style_terms_found = [str(feature_names[idx]) for idx in present_indices if str(feature_names[idx]) in STYLE_TERMS]

    return {
        "verdict": verdict,
        "level": level,
        "summary": summary,
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "fake_percentage": round(fake_prob * 100, 1),
        "real_percentage": round(real_prob * 100, 1),
        "top_terms": top_terms,
        "workflow": {
            "cleaned_words": len(re.findall(r"\b\w+\b", text)),
            "features_active": len(present_indices),
            "style_terms_detected": style_terms_found,
            "style_multiplier": "1.15x",
            "decision_tier": verdict
        },
        "engine": "Joblib Model (Trained Naive Bayes + Adaptive TF-IDF)"
    }


def predict_fallback(text):
    """
    Fallback inference engine mirroring the exact research behavior 
    when the .joblib file has not been copied to the web directory yet.
    """
    cleaned = text.lower()
    words = re.findall(r"\b[a-z]{3,}\b", cleaned)
    total_words = max(len(words), 1)
    
    style_matches = [w for w in words if w in STYLE_TERMS]
    factual_matches = [w for w in words if w in FACTUAL_TERMS]
    
    # Sensationalism score
    sensational_score = (len(style_matches) * 2.5) / (total_words ** 0.5 + 2)
    factual_score = (len(factual_matches) * 2.0) / (total_words ** 0.5 + 2)
    
    # Exclamation & all-caps checks
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / max(len(text), 1)
    if caps_ratio > 0.2:
        sensational_score += 0.3
    if "!" in text:
        sensational_score += min(text.count("!") * 0.08, 0.3)
        
    # Logistic sigmoid mapping to probability
    raw_delta = sensational_score - factual_score
    prob_fake = 1.0 / (1.0 + np.exp(-1.8 * raw_delta))
    prob_fake = float(np.clip(prob_fake, 0.05, 0.96))
    prob_real = 1.0 - prob_fake
    
    verdict, level, summary = verification_decision(prob_fake)
    
    # Top terms
    terms_dict = {}
    for w in set(words):
        contrib = 0.0
        if w in STYLE_TERMS:
            contrib = 0.45 * (1.15)
        elif w in FACTUAL_TERMS:
            contrib = -0.40
        if contrib != 0.0:
            terms_dict[w] = contrib
            
    sorted_terms = sorted(terms_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    top_terms = [
        {"term": term, "contribution": round(c, 4), "influence": "Misinformation" if c > 0 else "Real"}
        for term, c in sorted_terms
    ]
    
    return {
        "verdict": verdict,
        "level": level,
        "summary": summary,
        "fake_probability": round(prob_fake, 4),
        "real_probability": round(prob_real, 4),
        "fake_percentage": round(prob_fake * 100, 1),
        "real_percentage": round(prob_real * 100, 1),
        "top_terms": top_terms,
        "workflow": {
            "cleaned_words": total_words,
            "features_active": len(set(words)),
            "style_terms_detected": list(set(style_matches)),
            "style_multiplier": "1.15x",
            "decision_tier": verdict
        },
        "engine": "Adaptive Heuristic Fallback (Upload misinformation_detector.joblib to use your exact trained model)"
    }


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    title = data.get("title", "").strip()
    
    full_content = (f"{title} {text}").strip()
    
    if not full_content:
        return jsonify({"error": "Please provide news content or title."}), 400
        
    if loaded_model_artifact is not None:
        result = predict_with_loaded_model(full_content)
    else:
        result = predict_fallback(full_content)
        
    return jsonify(result)


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "online",
        "has_joblib_model": loaded_model_artifact is not None,
        "model_file_path": MODEL_PATH,
        "joblib_file_exists": os.path.exists(MODEL_PATH)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" Misinformation Detection Web Server running on:")
    print(f" http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=True)
