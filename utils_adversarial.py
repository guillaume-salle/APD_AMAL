import numpy as np
from skimage.feature import peak_local_max

def get_centers(M, ratio_threshold = 0.6, min_distance = 20):
    # Find coordinates of local maxima
    coordinates = peak_local_max(M, num_peaks=3, exclude_border=False, min_distance=min_distance)

    if len(coordinates)==0:
      global_max_index = np.unravel_index(np.argmax(M), M.shape)
      return np.array([global_max_index])

    # Extract the values at these coordinates
    maxima_values = M[coordinates[:, 0], coordinates[:, 1]]

    # Determine the highest value among the maxima
    highest_maxima_value = max(maxima_values)

    # Filter coordinates based on the threshold
    filtered_coordinates = coordinates[maxima_values >= ratio_threshold * highest_maxima_value]
    return filtered_coordinates