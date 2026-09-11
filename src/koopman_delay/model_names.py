"""Public model names and the immutable archive-key mapping.

Only these archive keys retain the earlier compact names. Frozen NPZ payloads
are deliberately not rewritten when directories and presentation labels change.
"""

# Markov RBF is absent from the controlled protocol and has no archive key.
MODEL_NAMES = {
    "mlp_koopman": {"legacy_model_id": "M1", "prediction_key": "prediction_mlp"},
    "lstm_koopman": {"legacy_model_id": "M2", "prediction_key": "prediction_lstm"},
    "hakan_koopman": {"legacy_model_id": "M3", "prediction_key": "prediction_hakan"},
    "dlm_koopman": {
        "legacy_model_id": "M5",
        "prediction_key": "prediction_m5",
    },
    "rbf_markov": {"legacy_model_id": "M6", "prediction_key": None},
    "rbf_physical_delay": {"legacy_model_id": "M8", "prediction_key": "prediction_m8"},
}

SLUG_BY_PREDICTION_KEY = {
    row["prediction_key"]: name for name, row in MODEL_NAMES.items() if row["prediction_key"] is not None
}
