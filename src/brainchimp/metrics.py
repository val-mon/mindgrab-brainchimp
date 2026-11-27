"""
Ref
    - https://karam-nus.github.io/metrics/docs/metrics/segmentation/dice
    - https://en.wikipedia.org/wiki/Hausdorff_distance
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html
"""

from time import perf_counter

import numpy as np
from scipy import ndimage


def dice(mask_truth, mask_pred):
    if mask_truth.shape != mask_pred.shape:
        raise ValueError("Mask have not the same shape")

    sum_t = np.count_nonzero(mask_truth)  # TP + FN
    sum_p = np.count_nonzero(mask_pred)  # TP + FP

    if sum_t == 0 and sum_p == 0:
        return 1

    tp = np.count_nonzero(mask_truth & mask_pred)
    den = sum_t + sum_p  # TP + FN + TP + FP = 2*TP + FN + FP

    return 2 * tp / den


def hd95(mask_truth, mask_pred, voxel_size):
    if mask_truth.shape != mask_pred.shape:
        raise ValueError("Mask have not the same shape")

    if len(voxel_size) != mask_truth.ndim:
        raise ValueError("Voxel size dimension do not correspond to mask dimension")

    surface_t = mask_truth & ~ndimage.binary_erosion(mask_truth)
    surface_p = mask_pred & ~ndimage.binary_erosion(mask_pred)

    if not surface_t.any() and not surface_p.any():
        return 0.0

    if not surface_t.any() or not surface_p.any():
        return float("inf")

    distance_to_t = ndimage.distance_transform_edt(~surface_t, sampling=voxel_size)
    distance_to_p = ndimage.distance_transform_edt(~surface_p, sampling=voxel_size)

    distances = np.concatenate((distance_to_p[surface_t], distance_to_t[surface_p]))

    return float(np.percentile(distances, 95))


def volume_error(mask_truth, mask_pred, voxel_size):
    if mask_truth.shape != mask_pred.shape:
        raise ValueError("Mask have not the same shape")

    if len(voxel_size) != mask_truth.ndim:
        raise ValueError("Voxel size dimension do not correspond to mask dimension")

    vol_t = mask_truth.sum() * np.prod(voxel_size)
    vol_p = mask_pred.sum() * np.prod(voxel_size)

    if vol_t == 0 and vol_p == 0:
        return (0.0, 0.0)

    if vol_t == 0:
        raise ZeroDivisionError

    error_mm3 = abs(vol_t - vol_p)
    error_cm3 = error_mm3 / 1000
    error_percent = error_mm3 * 100 / vol_t

    return float(error_percent), float(error_cm3)


if __name__ == "__main__":
    size = 256
    nb_iter = 100

    truth = np.zeros((size, size, size), dtype=bool)
    pred = np.zeros_like(truth)

    truth[0, 0, 0] = True
    truth[0, 0, 1] = True
    truth[0, 0, 2] = True

    pred[0, 0, 0] = True
    pred[0, 0, 1] = True

    tot_time = np.zeros(nb_iter)

    start = perf_counter()
    for i in range(nb_iter):
        t = perf_counter()
        res = dice(truth, pred)
        tot_time[i] = perf_counter() - t
        print(f"{i}: {res}")
        assert res == (4 / 5)

    print(f"avg time: {(np.average(tot_time) * 1000):.2}ms ± {(np.std(tot_time) * 1000):.3}ms")
    print(f"total time: {(perf_counter() - start):.2}s")
