---
title: Specialty Router
emoji: 🩺
sdk: docker
app_port: 7860
pinned: false
---

# Specialty Router — EXSU 500

Course prototype for research and educational purposes. Not a medical device. Not for clinical use.

## Deployment

1. Put the trusted model artifact `tfidf_lr_42.joblib` in `models/` beside these files.
2. For local testing, activate the existing Python environment, install `requirements.txt`, and run `python app.py`.
3. This is a Docker Space configuration. Creation of compute-backed Spaces may require a paid HF account.
4. Do not upload any MTSamples notes, PHI, credentials, or other training artifacts.
5. Review model file size and source before uploading; never load untrusted `.joblib` files.
6. If hosted on an alternative service, use the included `Dockerfile` and expose port 7860.

The public app contains only the TF-IDF + Logistic Regression model and preserves the original local interface styling.
