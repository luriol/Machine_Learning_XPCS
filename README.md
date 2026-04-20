# Machine_Learning_XPCS

## Overview

This repository contains initial code for processing two-time XPCS data.

The current objective is to normalize the two-time correlation matrix so that the diagonal is equal to unity. The approach uses a rotated coordinate system aligned with the diagonal of the matrix to improve averaging and reduce noise.

This is an initial prototype. Additional analysis and refinement will be added.

---

## Contents

- reshape_funs.py  
  Contains functions for geometric transformations of the data.  
  The primary function, `diagonal_resample_square`, interpolates the matrix onto a coordinate system aligned with the diagonal.

- Jupyter notebook  
  Demonstrates loading data, performing the rotation, averaging, and normalization.

- Data files (.mat)  
  Example two-time XPCS datasets used for testing.

---

## Method

The normalization procedure implemented here consists of:

- Interpolating the 2D matrix onto a rotated coordinate system where one axis lies along the diagonal
- Averaging across the direction perpendicular to the diagonal to reduce noise
- Extracting a mean value near the diagonal center
- Dividing the original matrix by this value to normalize the diagonal

The rotation step allows averaging over many pixels that are expected to be equivalent, improving statistical stability.

---

## Example Code

```python
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from reshape_funs import diagonal_resample_square

matfile = "D2_A_170C_B1_batch001_q009_phi001_twotimeresult.mat"

M = scipy.io.loadmat(matfile)["C"]
Mp = resize(M, (256,256), anti_aliasing=True)

# Rotate into diagonal coordinates
Mrot, xp, yp, x0, y0, xpmax, corners, boundary = diagonal_resample_square(
    Mp, frac=0.80, half_size=20, dx=1.0
)

# Average across diagonal strip
yp = np.average(Mrot, 0)

nx = Mrot.shape[1]
xp = np.arange(nx) - nx/2

# Estimate central value
center_slice = yp[int(nx/2+1)-2 : int(nx/2+1)+2]
ymean = np.average(center_slice)

# Normalize original matrix
Mout = Mp / ymean

plt.imshow(Mout, origin='lower')
plt.colorbar()
plt.show()
