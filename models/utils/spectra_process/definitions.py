import numpy as np
import pandas as pd


# ------ reference MassFormer ------ #
def none_or_nan(thing):
    if thing is None:
        return True
    elif isinstance(thing, float) and np.isnan(thing):
        return True
    elif pd.isnull(thing):
        return True
    else:
        return False

CHARGE_FACTOR_MAP = {
    1: 1.0,
    2: 0.9,
    3: 0.85,
    4: 0.8,
    5: 0.75,
    'large': 0.75
}
