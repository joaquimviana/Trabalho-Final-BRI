import numpy as np

def minmaxps_hashing_n(element_set, permutation,k):
    rows,_ = np.nonzero(element_set)
    row_count = len(rows)
    slot_size = row_count
    slot_size //= k
    min_id = []
    max_id = []
    for y in range(k): 
        min_id.append(np.Inf)
        max_id.append(-np.Inf)

     
    
    
    for j in range(k):
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




def minmax_hashing_k_slot_asymetric(element_set, permutation,k):
    
    rows,_ = np.nonzero(element_set)
    row_count = element_set.shape[0]
    slot_size = row_count
    slot_size //= k
    slot_remainder = row_count % k
    
    min_id = []
    max_id = []
    slot_ranges = []
    for y in range(k): 
        min_id.append(np.Inf)
        max_id.append(-np.Inf)
        
        slot_ranges.append(slot_size)
        if(slot_remainder > 0):
            slot_ranges[y] += 1
            slot_remainder -= 1
    
        
    qtd = 0  #var only needs for debug
    
    slot_pos = 0
    slot_range = slot_ranges[slot_pos]
    for i in range(len(rows)):
        perm_id = permutation[rows[i]]

        while(slot_range < rows[i]):                        
            slot_pos += 1
            slot_range += slot_ranges[slot_pos]
            qtd = 0
        
        qtd = qtd + 1
        if perm_id < min_id[slot_pos]:
            min_id[slot_pos] = perm_id
      
        if perm_id > max_id[slot_pos]:
            max_id[slot_pos] = perm_id
       
    #print('result', np.concatenate((min_id,max_id),axis= 0) )
    return np.concatenate((min_id,max_id),axis= 0) 



