"""Public deployment of the EXSU 500 TF-IDF logistic regression classifier.

Run: python app.py --port 8000 (for local testing)
No dataset is loaded, and input text is not intentionally logged or saved.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "tfidf_lr_42.joblib"
CLASSES = [
    "Cardiovascular / Pulmonary", "ENT / Ophthalmology", "Gastroenterology",
    "General Medicine", "Hematology - Oncology", "Neurology",
    "Obstetrics / Gynecology / Pediatrics", "Orthopedic", "Psychiatry / Psychology",
    "Radiology", "Urology / Nephrology",
]

bundle = joblib.load(MODEL)  # only load your group's trusted model file
vectorizer = bundle["vectorizer"]
classifier = bundle["model"]
if list(classifier.classes_) != list(range(len(CLASSES))):
    raise RuntimeError("Unexpected class ordering in model")


def predict(text):
    if not isinstance(text, str) or len(text.strip()) < 20:
        return {"error": "Please enter at least 20 characters of fictional or de-identified text."}
    if len(text) > 12000:
        return {"error": "Input is too long (maximum 12,000 characters)."}
    features = vectorizer.transform([text])
    probabilities = classifier.predict_proba(features)[0]
    best = int(np.argmax(probabilities))
    top = np.argsort(-probabilities)[:3]
    row = features.tocoo()
    coef = classifier.coef_[best]
    vocabulary = vectorizer.get_feature_names_out()
    contributors = sorted(((float(row.data[i] * coef[row.col[i]]), int(row.col[i]))
                           for i in range(len(row.data))), reverse=True)
    words = [{"term": str(vocabulary[i]), "note": None}
             for value, i in contributors if value > 0][:8]
    return {
        "specialty": CLASSES[best],
        "confidence": float(probabilities[best]),
        "probs": [{"class": CLASSES[i], "prob": float(probabilities[i])} for i in top],
        "why": words,
        "note": "Model-estimated probabilities; not clinical certainty. Positive word contributions shown."
    }


class Handler(BaseHTTPRequestHandler):
    def _respond(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._respond(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._respond(200, b"ok", "text/plain; charset=utf-8")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > 50000:
                self._respond(413, b'{"error":"Request too large or empty."}', "application/json")
                return
            request = json.loads(self.rfile.read(n))
            if not isinstance(request, dict) or request.get("model", "lr") != "lr":
                result = {"error": "Only TF-IDF + logistic regression is available."}
            else:
                result = predict(request.get("text", ""))
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            result = {"error": "Invalid input."}
        except Exception:
            result = {"error": "Prediction failed."}
        self._respond(200, json.dumps(result).encode("utf-8"), "application/json")

    def log_message(self, format, *args):
        pass  # Do not log submitted text or access details


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"Ready on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
