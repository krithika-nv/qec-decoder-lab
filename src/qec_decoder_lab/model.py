"""Explicit, reproducible definitions of surface-code memory experiments."""
from dataclasses import asdict, dataclass

NOISE_PROFILES = ("circuit_level", "data_only", "measurement_only", "custom")


@dataclass(frozen=True)
class MemoryConfig:
    distance: int = 3
    rounds: int = 3
    basis: str = "x"
    p: float = 0.005
    noise_profile: str = "circuit_level"
    after_clifford_depolarization: float | None = None
    before_round_data_depolarization: float | None = None
    before_measure_flip_probability: float | None = None
    after_reset_flip_probability: float | None = None

    def __post_init__(self) -> None:
        if self.distance < 3 or self.distance % 2 == 0:
            raise ValueError("distance must be an odd integer >= 3")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.basis not in ("x", "z"):
            raise ValueError("basis must be x or z")
        if self.noise_profile not in NOISE_PROFILES:
            raise ValueError(f"noise_profile must be one of {NOISE_PROFILES}")
        for name, value in asdict(self).items():
            if name == "p" or name.endswith(("depolarization", "probability")):
                if value is not None and not 0 <= value <= 1:
                    raise ValueError(f"{name} must be between 0 and 1")
        if self.noise_profile == "custom":
            names = (
                "after_clifford_depolarization", "before_round_data_depolarization",
                "before_measure_flip_probability", "after_reset_flip_probability",
            )
            if any(getattr(self, name) is None for name in names):
                raise ValueError("custom profile requires all four noise probabilities")

    def noise_parameters(self) -> dict[str, float]:
        p = self.p
        if self.noise_profile == "circuit_level":
            return {
                "after_clifford_depolarization": p,
                "before_round_data_depolarization": p,
                "before_measure_flip_probability": p,
                "after_reset_flip_probability": p,
            }
        if self.noise_profile == "data_only":
            return {
                "after_clifford_depolarization": 0.0,
                "before_round_data_depolarization": p,
                "before_measure_flip_probability": 0.0,
                "after_reset_flip_probability": 0.0,
            }
        if self.noise_profile == "measurement_only":
            return {
                "after_clifford_depolarization": 0.0,
                "before_round_data_depolarization": 0.0,
                "before_measure_flip_probability": p,
                "after_reset_flip_probability": 0.0,
            }
        return {
            "after_clifford_depolarization": float(self.after_clifford_depolarization),
            "before_round_data_depolarization": float(self.before_round_data_depolarization),
            "before_measure_flip_probability": float(self.before_measure_flip_probability),
            "after_reset_flip_probability": float(self.after_reset_flip_probability),
        }

    def circuit(self):
        import stim

        return stim.Circuit.generated(
            f"surface_code:rotated_memory_{self.basis}",
            distance=self.distance,
            rounds=self.rounds,
            **self.noise_parameters(),
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "effective_noise_probabilities": self.noise_parameters()}
