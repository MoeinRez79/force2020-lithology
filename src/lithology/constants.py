"""Fixed vocabulary for the FORCE 2020 lithology dataset.

The competition encodes lithology as sparse integer codes. We map them to a
dense 0..11 index space so they can be used as classification targets.
The ordering below is used everywhere (penalty matrix rows/cols, model logits).
"""
from __future__ import annotations

# Official FORCE 2020 lithology code -> human-readable name.
# (These integer keys are the values found in FORCE_2020_LITHOFACIES_LITHOLOGY.)
LITHOLOGY_CODES: dict[int, str] = {
    30000: "Sandstone",
    65030: "Sandstone/Shale",
    65000: "Shale",
    80000: "Marl",
    74000: "Dolomite",
    70000: "Limestone",
    70032: "Chalk",
    88000: "Halite",
    86000: "Anhydrite",
    99000: "Tuff",
    90000: "Coal",
    93000: "Basement",
}

# Dense index <-> code <-> name. Order here defines the class index.
CODE_TO_IDX: dict[int, int] = {code: i for i, code in enumerate(LITHOLOGY_CODES)}
IDX_TO_CODE: dict[int, int] = {i: code for code, i in CODE_TO_IDX.items()}
IDX_TO_NAME: dict[int, str] = {i: LITHOLOGY_CODES[c] for i, c in IDX_TO_CODE.items()}
N_CLASSES: int = len(LITHOLOGY_CODES)
IGNORE_INDEX: int = -100  # padded / unlabelled depth samples

# Candidate wireline + drilling logs. Only DEPTH_MD and GR are guaranteed to
# exist; everything else is missing for large parts of the dataset, which is
# the central modelling challenge.
LOG_COLUMNS: list[str] = [
    "CALI", "RDEP", "RHOB", "DRHO", "SGR", "GR", "RMED", "RMIC",
    "NPHI", "PEF", "RSHA", "DTC", "SP", "BS", "ROP", "DTS", "DCAL", "MUDWEIGHT",
]
DEPTH_COL = "DEPTH_MD"
WELL_COL = "WELL"
X_COL, Y_COL = "X_LOC", "Y_LOC"
LABEL_COL = "FORCE_2020_LITHOFACIES_LITHOLOGY"
CONF_COL = "FORCE_2020_LITHOFACIES_CONFIDENCE"

# The confidence column (1 high / 2 medium / 3 low) is almost universally
# ignored by public solutions. We turn it into a per-sample loss weight so the
# model trusts clean labels more than noisy ones.
CONFIDENCE_WEIGHTS: dict[int, float] = {1: 1.0, 2: 0.7, 3: 0.4}
DEFAULT_CONF_WEIGHT: float = 0.7
