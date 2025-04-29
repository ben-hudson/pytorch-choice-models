import numpy as np
import pytest

from route_choice.utils import get_turn_angle


# clockwise is positive rotation
@pytest.mark.parametrize(
    "in_bearing,out_bearing,turn_angle",
    [
        (360, 90, 90),
        (0, 90, 90),
        (90, 180, 90),
        (180, 270, 90),
        (270, 360, 90),
        (270, 0, 90),
        (90, 360, -90),
        (90, 0, -90),
        (360, 270, -90),
        (0, 270, -90),
        (270, 180, -90),
        (180, 90, -90),
    ],
)
def test_turn_angle(in_bearing, out_bearing, turn_angle):
    ret = get_turn_angle(in_bearing, out_bearing)
    assert np.isclose(ret, turn_angle, atol=1e-6).all()
