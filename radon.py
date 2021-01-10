# Matriculation number: 2227134

import numpy as np 
from scipy.misc import ascent
from skimage.transform import radon
# Add other imports here if needed
from numpy.fft import fft, ifft, fftshift

im = ascent()
n = im.shape[0]
padded = np.pad(im, n // 2, mode='constant')
# ANGLES is an array between 0 – 180 degrees, e.g. np.arange(0, 180, 1)
ANGLES = np.arange(0, 360, 1)
# You need to decide how many angles you need to obtain a good result
sino = radon(padded.astype(np.float), theta=ANGLES, circle=True)

# Your solution here
sino = np.transpose(sino)

#####

# |Omega| Filtering
im_size = im.shape[0]

omega_filter = [element * 0.55555 / n for element in np.absolute(list(range(-n, n)))]
for x in list(range(0, 355)) + list(range((n * 2)-355, n * 2)):
    omega_filter[x] = 0

omega_filter = fftshift(omega_filter)

y, x = np.mgrid[-n:n, -n:n]
array = np.zeros((n * 2,n * 2))

for theta in range(len(sino)):
    temp_array = np.zeros_like(array)
    for i in range(n * 2):
        temp_array[i] = temp_array[i] + (np.pi / 360 * ifft(np.abs(omega_filter) * fft(sino[theta])).real)
    
    # Backward Approach
    sa = np.sin(np.deg2rad(theta))
    ca = np.cos(np.deg2rad(theta))
    xx = np.clip(x * ca - y * sa + n, 0, n * 2 - 1).astype(np.int)
    yy = np.clip(x * sa + y * ca + n, 0, n * 2 - 1).astype(np.int)
    array = array + temp_array[yy, xx]

# Crop array to im form
solution = array[n // 2:n + n // 2, n // 2:n + n // 2]

#####

# MSE of your solution with respect to original
print(np.mean((im - solution) ** 2))
