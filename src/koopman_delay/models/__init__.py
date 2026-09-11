"""Unified identification model implementations."""
from .rbf_edmdc import RBFEDMDc, RBFEDMDcConfig

__all__ = [
    "RBFEDMDc",
    "RBFEDMDcConfig",
]

try:
    from .delayed_lorenz_mlp import MLPKoopman
    from .dlm import DLMKoopman, LSTMHaKANKoopman, LSTMOnlyKoopman, StateLSTMHaKANKoopman
    from .strict_family_baselines import StrictLSTMKoopman, StrictMLPKoopman, StrictHaKANKoopman
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
else:
    __all__ += [
        "LSTMHaKANKoopman",
        "DLMKoopman",
        "LSTMOnlyKoopman",
        "StateLSTMHaKANKoopman",
        "MLPKoopman",
        "StrictMLPKoopman",
        "StrictLSTMKoopman",
        "StrictHaKANKoopman",
    ]
