# -*- coding: utf-8 -*-
"""
FACIAL INHERITANCE PREDICTOR
=============================
Upload photos of two parents + optional family members.
Blends facial embeddings and generates a predicted child face.
"""

import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from deepface import DeepFace


def get_face_embedding(image_path, model_name="Facenet512"):
    """Extract 512-number facial fingerprint from a photo."""
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            enforce_detection=False
        )
        return np.array(result[0]["embedding"])
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def analyze_face(image_path):
    """Analyze age, gender, race, emotion from a photo."""
    try:
        result = DeepFace.analyze(
            img_path=image_path,
            actions=["age", "gender", "race", "emotion"],
            enforce_detection=False
        )
        return result[0]
    except Exception as e:
        print(f"Analysis error: {e}")
        return None


def blend_embeddings(embeddings_with_weights):
    """
    Core function: blend face embeddings by weight.
    embeddings_with_weights = [(embedding, weight), ...]
    Weights are auto-normalized to sum to 1.0
    """
    total = sum(w for _, w in embeddings_with_weights)
    blended = np.zeros_like(embeddings_with_weights[0][0], dtype=np.float64)
    for emb, w in embeddings_with_weights:
        blended += emb * (w / total)
    return blended


def cosine_similarity(e1, e2):
    """How similar are two face embeddings? 1.0 = identical."""
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
