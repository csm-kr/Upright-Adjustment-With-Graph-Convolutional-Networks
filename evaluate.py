import numpy as np


def angle_acc(gt_xyz, pred_xyz):
    """

    :param gt_xyz:
    :param pred_xyz:
    :return:
    """

    gt_dir = gt_xyz
    pred_dir =pred_xyz

    if pred_dir.ndim == 1:
        dot_prod = np.sum(np.multiply(pred_dir, gt_dir), axis=0)
    else:
        dot_prod = np.sum(np.multiply(pred_dir, gt_dir), axis=1)
    dot_prod = np.clip(dot_prod, -0.999999, 0.999999)
    angle = np.arccos(dot_prod) * 180 / np.pi
    angle = np.mean(angle)
    return angle

