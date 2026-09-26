"""Replace the body of estimate_duration with your existing predictor."""


def estimate_duration(text: str, exaggeration: float, cfg_weight: float) -> float:
    # EXAMPLE ONLY. Replace this with your trained/fit duration estimator.
    words = max(1, len(text.split()))
    base_wpm = 152.0
    # Approximate relationship only; your estimator should replace this.
    rate = base_wpm * (1.0 + 0.18 * (exaggeration - 0.5) + 0.12 * (cfg_weight - 0.5))
    punctuation_seconds = 0.08 * sum(text.count(c) for c in ",;:")
    punctuation_seconds += 0.16 * sum(text.count(c) for c in ".!?")
    return 60.0 * words / rate + punctuation_seconds
