import numpy as np


def minps_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id1,min_id2 = np.Inf, np.Inf
     
    row_count = len(rows)
    slot_size = row_count
    slot_size //= 2
     
    for i in range(slot_size):
         
        perm_id1 = permutation[rows[i]]
        perm_id2 = permutation[rows[i+slot_size]]
         
        if perm_id1 < min_id1:
            min_id1 = perm_id1

        if perm_id2 < min_id2:
            min_id2 = perm_id2
    
    '''
        continue from last i value
    '''        
    for i in range(i+slot_size,row_count):
        perm_id2 = permutation[rows[i]]
        if perm_id2 < min_id2:
            min_id2 = perm_id2

    return np.array([min_id1,min_id2])






def minmaxps_hashing(element_set, permutation):
    rows,_ = np.nonzero(element_set)
    min_id1,min_id2 = np.Inf, np.Inf
    max_id1,max_id2 = -np.Inf, -np.Inf
     
    row_count = len(rows)
    print ('row',row_count)
    slot_size = row_count
    slot_size //= 2
    print ('slot_size',slot_size)
     
    for i in range(slot_size):
         
        perm_id1 = permutation[rows[i]]
        perm_id2 = permutation[rows[i+slot_size]]
        print(i, '<', i+slot_size)
        rows[i+slot_size]
        if perm_id1 < min_id1:
            min_id1 = perm_id1

        if perm_id2 < min_id2:
            min_id2 = perm_id2

        if perm_id1 > max_id1:
            max_id1 = perm_id1

        if perm_id2 > max_id2:
            max_id2 = perm_id2
    
    '''
        continue from last i value
    '''
    print('\n\n\n\n\n#####################################################################################3')
    for i in range(i+slot_size,row_count):
        print('i', i)
        perm_id2 = permutation[rows[i]]

        if perm_id2 < min_id2:
            min_id2 = perm_id2
        
        if perm_id2 > max_id2:
            max_id2 = perm_id2

    return np.array([min_id1,min_id2,max_id1,max_id2])