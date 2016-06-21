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
    
    slot_size = row_count
    slot_size //= 2
    
     
    for i in range(slot_size):
         
        perm_id1 = permutation[rows[i]]
        perm_id2 = permutation[rows[i+slot_size]]
      
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

    for i in range(i+slot_size,row_count):
      
        perm_id2 = permutation[rows[i]]

        if perm_id2 < min_id2:
            min_id2 = perm_id2
        
        if perm_id2 > max_id2:
            max_id2 = perm_id2

    return np.array([min_id1,min_id2,max_id1,max_id2])


def minmax_hashing_two_slot_asymetric(element_set, permutation):
    
    rows,_ = np.nonzero(element_set)
    row_count = element_set.shape[0]
    slot_size = row_count
    slot_size //= 2
    slot_remainder = slot_size % 2
    
    min_id,min_id2 = np.Inf,np.Inf
    max_id,max_id2 = -np.Inf,-np.Inf
    
    slot_range = slot_size
    if(slot_remainder > 0):
        slot_range += 1    
   
    for i in range(len(rows)):
        perm_id = permutation[rows[i]]

        if(slot_range < rows[i]):         
            if perm_id < min_id:
                min_id = perm_id
      
            if perm_id > max_id:
                max_id = perm_id
        else:
            if perm_id < min_id2:
                min_id2 = perm_id
      
            if perm_id > max_id2:
                max_id2 = perm_id
       
    #print('result', np.concatenate((min_id,max_id),axis= 0) )
    return np.concatenate(([min_id,min_id2],[max_id,max_id2]),axis= 0)