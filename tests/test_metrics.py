import numpy as np
import pytest

from brainchimp.metrics import dice, hd95, volume_error


def test_dice_empty_mask():
    empty = np.zeros((3, 3, 3), dtype=bool)

    assert dice(empty, empty) == 1


def test_dice_known_overlap():
    truth = np.zeros((3, 3, 3), dtype=bool)
    pred = np.zeros_like(truth)

    truth[0, 0, 0] = True
    truth[0, 0, 1] = True
    truth[0, 0, 2] = True

    pred[0, 0, 0] = True
    pred[0, 0, 1] = True

    assert dice(truth, pred) == pytest.approx(4 / 5)


def test_hd95_concentric_spheres():
    bound = 16
    coords = np.ogrid[-bound:bound, -bound:bound, -bound:bound]
    squared_dist = sum(coord**2 for coord in coords)

    sphere_10 = squared_dist < 10**2
    sphere_12 = squared_dist < 12**2

    res = hd95(sphere_10, sphere_12, voxel_size=(1, 1, 1))

    assert res == pytest.approx(2.0, abs=0.5)


def test_hd95_anisotropic_voxels():
    truth = np.zeros((3, 3, 3), dtype=bool)
    pred = np.zeros_like(truth)

    truth[1, 1, 0] = True
    pred[1, 1, 1] = True

    assert hd95(truth, pred, voxel_size=(0.5, 0.5, 1.5)) == pytest.approx(1.5)


def test_volume_error_empty():
    empty = np.zeros((3, 3, 3), dtype=bool)

    assert volume_error(empty, empty, (1.0, 1.0, 1.0)) == (0.0, 0.0)


def test_volume_error_0div():
    truth = np.zeros((3, 3, 3), dtype=bool)
    pred = np.ones_like(truth)

    with pytest.raises(ZeroDivisionError):
        volume_error(truth, pred, (1.0, 1.0, 1.0))


def test_volume_error_known():
    truth = np.zeros((3, 3, 3), dtype=bool)
    pred = np.zeros_like(truth)

    truth[0, 0, 0] = True
    truth[0, 0, 1] = True
    truth[0, 0, 2] = True

    pred[0, 0, 0] = True
    pred[0, 0, 1] = True

    error_percent, error_cm3 = volume_error(truth, pred, voxel_size=(1, 1, 1))

    assert error_percent == pytest.approx(100 / 3)
    assert error_cm3 == pytest.approx(0.001)
