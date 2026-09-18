"""Feedback labels shared by the Reacher comparison cases."""


def feedback_type_for_method(method: str) -> str:
    if method == "Tube-MPPI":
        return "latent_tube_mppi"
    if method == "Robust-MPPI":
        return "latent_robust_mppi"
    return "none"
