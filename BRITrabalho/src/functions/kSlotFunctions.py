import numpy as np

def minmaxps_hashing_n(element_set, permutation,n):
    rows,_ = np.nonzero(element_set)
    row_count = len(rows)
    slot_size = row_count
    slot_size //= n
    min_id = []
    max_id = []
    for k in range(n): 
        min_id.append(np.Inf)
        max_id.append(-np.Inf)

     
    
    
    for j in range(n):
        for i in range((slot_size *j),(slot_size *(j+1))):
            perm_id = permutation[rows[i]]            
             
            if perm_id < min_id[j]:
                min_id[j] = perm_id
          
            if perm_id > max_id[j]:
                max_id[j] = perm_id


        '''
        continue from last i value
        '''
        for i in range((slot_size *j + i),(slot_size *(j+1))):
            
            perm_id = permutation[rows[i]]            
             
            if perm_id < min_id[j]:
                min_id[j] = perm_id
          
            if perm_id > max_id[j]:
                max_id[j] = perm_id
         
    #print('result', np.concatenate((min_id,max_id),axis= 0) )
    return np.concatenate((min_id,max_id),axis= 0) 