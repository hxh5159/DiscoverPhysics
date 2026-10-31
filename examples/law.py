"""Interface demonstration only; this is not a solution to a contest world."""
import numpy as np

LAW_API_VERSION = "1.0.0"
PARAMS = {"x_scale": 0.2, "y_scale": 0.3}
BODY_PROPS = {}
HIDDEN = []


def acceleration(t, bodies, params):
    # A toy anisotropic linear field, solely to demonstrate the interface.
    result = np.zeros((len(bodies), 2), dtype=np.float64)
    for i, body in enumerate(bodies):
        if body["kind"] == "probe":
            result[i, 0] = -params["x_scale"] * body["pos"][0]
            result[i, 1] = -params["y_scale"] * body["pos"][1]
    return result
