import numpy as np

def min_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id = np.Inf
    
    for linei in rows:
        
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id

    return np.array([min_id])

def max_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    max_id = -np.Inf
    
    for linei in rows:
        
        perm_id = permutation[linei]
        if perm_id > max_id:
            max_id = perm_id

    return np.array([max_id])

def minmax_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id,max_id = np.Inf,-np.Inf
    
    for linei in rows:
        
        perm_id = permutation[linei]
        if perm_id < min_id:
            min_id = perm_id
        if perm_id > max_id:
            max_id = perm_id

    return np.array([min_id,max_id])