import nibabel as nib
import nibabel.processing
import numpy as np
import pytest

from brainchimp.conform import conform, unconform
from brainchimp.metrics import dice


@pytest.fixture(
    params=[
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 1.0),
    ],
    ids=["0.5mm", "1mm", "0.5x0.5x0.1mm"],
)
def test_mask(nmt_mask, request):
    source = nib.load(nmt_mask)
    return nib.processing.resample_to_output(source, voxel_sizes=request.param, order=0)


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_border(test_mask, scale):
    res = conform(test_mask, scale, order=0)
    data = res.get_fdata() > 0

    assert not data[0, :, :].any()
    assert not data[-1, :, :].any()
    assert not data[:, 0, :].any()
    assert not data[:, -1, :].any()
    assert not data[:, :, 0].any()
    assert not data[:, :, -1].any()


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_volume(test_mask, scale):
    original_data = test_mask.get_fdata() > 0
    original_vol = original_data.sum() * np.prod(test_mask.header.get_zooms()[:3])

    res = conform(test_mask, scale, order=0)

    res_data = res.get_fdata() > 0
    res_vol = res_data.sum() * np.prod(res.header.get_zooms()[:3])

    assert np.isclose(res_vol / scale**3, original_vol, rtol=0.02)


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_dice(test_mask, scale):
    original = test_mask.get_fdata() > 0
    conformed = conform(test_mask, scale, order=0)
    restored = unconform(conformed, test_mask, scale).get_fdata() > 0

    assert dice(original, restored) >= 0.95
