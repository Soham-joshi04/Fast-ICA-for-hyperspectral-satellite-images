# app.py  (only the differences from your last version are shown)

import gradio as gr
import numpy as np
from fastICA import fast_ica
from utils import (
    hsi_cube_to_matrix, fastica_output_to_hsi,
    false_colour_img, reduced_rgb_img,
    component_gallery_img, stitch_imgs,
)

# ── Load data once so we know band counts ──────────────────────────
hsi_data = {
    "Indian Pines": {
        "image": np.load("data/indian pine/indianpinearray.npy"),
        "mask" : np.load("data/indian pine/IPgt.npy"),
    },
    "Salinas": {
        "image": np.load("data/salinas/salinas_corrected_salinas_corrected.npy"),
        "mask" : np.load("data/salinas/salinas_gt_salinas_gt.npy"),
    },
}
for v in hsi_data.values():
    v["C"] = v["image"].shape[-1]      # number of spectral bands

# ── Helper: visualise mask (unchanged) ─────────────────────────────
def visualize_mask(mask):
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap("tab20", int(mask.max()) + 1)
    return (cmap(mask)[:, :, :3] * 255).astype(np.uint8)

# ── Pipeline ───────────────────────────────────────────────────────
def pipeline(dataset, k, g_fun,
             b1, b2, b3, ic1, ic2, ic3):
    data = hsi_data[dataset]
    img, mask, C = data["image"], data["mask"], data["C"]

    # clip band indices to valid range
    bands = tuple(int(np.clip(b, 0, C-1)) for b in (b1, b2, b3))

    # —— FastICA
    X, H, W = hsi_cube_to_matrix(img)
    k = int(np.clip(k, 3, C))                 # safe guard
    S = fast_ica(X, n_components=k, fun=g_fun)
    ica_cube = fastica_output_to_hsi(S, H, W)

    # clip IC indices
    ics = tuple(int(np.clip(i, 0, k-1)) for i in (ic1, ic2, ic3))

    # —— Build visuals
    raw_mosaic = stitch_imgs(
        false_colour_img(img, bands),
        component_gallery_img(img[:, :, :], num=3, cols=3),
    )

    mask_img = visualize_mask(mask)

    ica_rgb   = reduced_rgb_img(ica_cube[:, :, ics])   # choose ICs
    ica_mosaic = stitch_imgs(
        ica_rgb,
        component_gallery_img(ica_cube, num=3, cols=3),
    )

    return raw_mosaic, mask_img, ica_mosaic

# ── Gradio UI ──────────────────────────────────────────────────────
def build_interface():
    # pick one dataset just to set max slider; they all have C attr.
    max_bands = max(v["C"] for v in hsi_data.values())

    iface = gr.Interface(
        fn=pipeline,
        inputs=[
            gr.Radio(list(hsi_data.keys()), label="HSI Dataset"),

            # ICA parameters
            gr.Slider(3, max_bands, value=10, step=1, label="ICA: number of components (k)"),
            gr.Radio(["tanh", "cube", "gauss"], label="Non‑linearity g(u)"),

            # user‑chosen raw bands
            gr.Number(value=49, label="Raw Band R"),
            gr.Number(value=26, label="Raw Band G"),
            gr.Number(value=16, label="Raw Band B"),

            # user‑chosen ICA components
            gr.Number(value=0, label="IC index R"),
            gr.Number(value=1, label="IC index G"),
            gr.Number(value=2, label="IC index B"),
        ],
        outputs=[
            gr.Image(label="Original HSI • RGB (your bands) + 3 comps"),
            gr.Image(label="Ground‑truth Mask"),
            gr.Image(label="ICA Reduced • RGB (your ICs) + 3 comps"),
        ],
        title="HSI Viewer with Selectable Bands & ICA Visuals",
        description=(
            "Pick any three raw spectral bands and any three ICA components "
            "to map to RGB. Slider ‘k’ can be set up to the sensor’s full "
            "band count."
        ),
    )
    return iface

if __name__ == "__main__":
    build_interface().launch()
