import os
import numpy as np
from scipy.io import loadmat

# Update these paths to your directories
input_dir = '.'
output_dir = '.'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith('.mat'):
        mat_path = os.path.join(input_dir, filename)
        data = loadmat(mat_path)
        base = os.path.splitext(filename)[0]
        for var_name, array in data.items():
            if var_name.startswith('__'):
                continue
            print(array.shape)
            out_name = f"{base}_{var_name}.npy"
            out_path = os.path.join(output_dir, out_name)
            np.save(out_path, array)
            print(f"Saved {out_path}")
