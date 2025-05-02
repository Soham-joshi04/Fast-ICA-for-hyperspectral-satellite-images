import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

def hsi_cube_to_matrix(hsi_cube):
    """
    Convert HSI cube of shape (H, W, C) to 2D matrix of shape (C, H*W)
    """
    H, W, C = hsi_cube.shape
    X = hsi_cube.reshape(H * W, C).T  # shape: (C, H*W)
    return X, H, W

def fastica_output_to_hsi(S, H, W):
    """
    Convert FastICA output from shape (k, H*W) to H x W x k

    Parameters:
    - S: numpy array of shape (k, H*W)
    - H: original height
    - W: original width

    Returns:
    - 3D numpy array of shape (H, W, k)
    """
    k, N = S.shape
    assert N == H * W, "Mismatch between H*W and number of pixels"
    hsi_ica = S.T.reshape(H, W, k)
    return hsi_ica




# ───────────────────────── helpers ─────────────────────────
def _to_hwc(cube):
    """Ensure cube is (H, W, C); auto‑transpose if given as (C, H, W)."""
    if cube.ndim != 3:
        raise ValueError("Need a 3‑D array")
    return cube.transpose(1, 2, 0) if cube.shape[0] <= 8 else cube

def _stretch(img):
    """0‑1 stretch then convert to uint8 RGB."""
    img = img.astype(np.float32)
    img -= img.min((0, 1), keepdims=True)
    img /= img.max((0, 1), keepdims=True) + 1e-6
    return (img * 255).clip(0, 255).astype(np.uint8)


# ───────────── 1.  false‑colour RGB (raw bands) ────────────
def false_colour_img(cube, bands=(49, 26, 16)):
    cube = _to_hwc(cube)
    rgb  = cube[:, :, list(bands)]
    return _stretch(rgb)                       # returns (H,W,3) uint8


# ───────────── 2.  PCA→RGB  ────────────
def pca_rgb_img(cube, whiten=False):
    cube = _to_hwc(cube)
    H, W, C = cube.shape
    pcs = PCA(n_components=3, whiten=whiten, random_state=0
              ).fit_transform(cube.reshape(-1, C))
    return _stretch(pcs.reshape(H, W, 3))


# ───────────── 3.  reduced cube → RGB (first 3 comps) ─────────
def reduced_rgb_img(red_cube):
    cube = _to_hwc(red_cube)
    if cube.shape[2] < 3:
        raise ValueError("Need at least 3 components")
    return _stretch(cube[:, :, :3])


# ───────────── 4.  component gallery (greyscale) ─────────────
def component_gallery_img(cube, num=6, cols=3):
    """
    Assemble a rows×cols collage of grayscale component maps.
    Returns a uint8 RGB canvas.
    """
    cube = _to_hwc(cube)
    k = cube.shape[2]
    num = min(num, k)
    rows = -(-num // cols)            # ceiling division

    H, W = cube.shape[:2]
    canvas = np.full((rows * H, cols * W, 3), 255, np.uint8)

    for i in range(num):
        r, c = divmod(i, cols)

        tile = _stretch(cube[:, :, i])          # (H,W) uint8
        if tile.ndim == 2:                      # make it RGB
            tile = np.stack([tile]*3, axis=-1)  # (H,W,3)

        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = tile

    return canvas




# ───────────── 5.  side‑by‑side stitch ─────────────
def stitch_imgs(*imgs, titles=None, pad=3):
    """
    Concatenate RGB uint8 images horizontally with optional top‑left titles.
    """
    h = max(im.shape[0] for im in imgs)
    widths = [im.shape[1] for im in imgs]
    out = np.full((h, sum(widths) + pad * (len(imgs) - 1), 3), 255, np.uint8)

    x = 0
    for i, im in enumerate(imgs):
        out[:im.shape[0], x:x + im.shape[1]] = im
        if titles and titles[i]:
            plt.text(0, 0, "")  # placeholder; skip drawing in “radio” mode
        x += im.shape[1] + pad
    return out
