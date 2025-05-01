import numpy as np

def hsi_cube_to_matrix(hsi_cube):
    """
    Convert HSI cube of shape (H, W, C) to 2D matrix of shape (C, H*W)
    """
    H, W, C = hsi_cube.shape
    X = hsi_cube.reshape(H * W, C).T  # shape: (C, H*W)
    return X, (H, W, C)

    def components_to_images(S, shape_hw):
    """
    Convert ICA components (k, N) to list of 2D images (H, W)
    """
    H, W = shape_hw
    k, N = S.shape
    assert N == H * W, "Shape mismatch during reconstruction"

    component_images = [S[i, :].reshape(H, W) for i in range(k)]
    return component_images  # list of H x W arrays

