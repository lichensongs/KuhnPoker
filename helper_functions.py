import numpy as np

def perturb_probs(probs, uncertainty):
    indices = np.where((probs > 0) & (probs < 1))[0]  # Indices of modifiable probabilities
    
    # Creating perturbed versions
    if len(indices) > 1:  # Ensure there are at least two modifiable probabilities
        lower = probs.copy()
        upper = probs.copy()

        # Decrease first modifiable prob and increase second
        lower[indices[0]] = max(0, lower[indices[0]] - uncertainty)
        lower[indices[1]] = min(1, lower[indices[1]] + uncertainty)

        # Increase first modifiable prob and decrease second
        upper[indices[0]] = min(1, upper[indices[0]] + uncertainty)
        upper[indices[1]] = max(0, upper[indices[1]] - uncertainty)

        return lower, upper
    else:
        return probs, probs  # Return unchanged if not enough elements to modify

def find_midpoint_overlap(interval1, interval2):
    interval1 = np.sort(interval1)
    interval2 = np.sort(interval2)
    
    a, b = interval1
    c, d = interval2
    
    start = max(a, c)
    end = min(b, d)
    
    if start <= end:
        midpoint = (start + end) / 2
        return midpoint
    else:
        return None  # No overlap
    

def intervals_overlap(interval1, interval2):
    interval1 = np.sort(interval1)
    interval2 = np.sort(interval2)
    
    return interval1[0] < interval2[1] and interval2[0] < interval1[1]