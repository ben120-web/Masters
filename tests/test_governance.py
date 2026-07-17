from ecg_denoising.config import PromotionConfig
from ecg_denoising.governance import assess_promotion


def test_promotion_rejects_candidate_with_harmful_worst_case() -> None:
    decision = assess_promotion(
        {"snr_improvement_db": 1.2},
        [
            {"input_snr_db": 0.0, "snr_improvement_db": 2.5},
            {"input_snr_db": 24.0, "snr_improvement_db": -0.4},
        ],
        PromotionConfig(0.5, 0.0),
    )

    assert decision["decision"] == "rejected"
    assert decision["clinical_release_authorized"] is False
    assert decision["checks"][0]["passed"] is True
    assert decision["checks"][1]["passed"] is False
