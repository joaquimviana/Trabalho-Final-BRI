'''
Created on 24 de mai de 2016

@author: Fellipe

'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import numpy as np
from time import time
from math import floor
#from duartefellipe.datasets.extractors.meter_extractor import extract_meter_to_corpus_ranking_task
from sklearn.metrics.pairwise import pairwise_distances
from short_plagiarised_answers_extractor import extract_short_plagiarized_answers_to_ranking
from sklearn.externals.joblib.parallel import Parallel, delayed
from scipy.sparse.csr import csr_matrix
from functions.hashingFunctions import *
from functions.kSlotFunctions import *
from functions.twoSlotFunctions import *




def prepare_results(results_dict, perm_repetition, perm_count, perm_true_jaccard, approach_name, approach_time,approach_jaccard ):
    try:
        perm_dict = results_dict[perm_count]
    except KeyError:
        results_dict[perm_count] = {}
        perm_dict = results_dict[perm_count]

    try:
        approach_dict = perm_dict[approach_name]
    except KeyError:
        perm_dict[approach_name] = {'approach_time':[],'errors':[]}
        approach_dict = perm_dict[approach_name]
    
    approach_dict['perm_repetition'] = perm_repetition
    approach_dict['approach_time'].append(approach_time)
    approach_dict['errors'].append(approach_jaccard - perm_true_jaccard)


def jaccard_similarity(row_index1,row_index2,set_per_row):
    set1 = np.unique(set_per_row[row_index1,:])
    set2 = np.unique(set_per_row[row_index2,:])

    sim = len(np.intersect1d(set1, set2, True)) / len(np.union1d(set1, set2))
    return sim

def pairwise_jaccard_similarity(set_per_row):
#    print (set_per_row)
    
    results = Parallel(
        n_jobs=-1,backend='threading'
    )(
            delayed(jaccard_similarity)(i,j,set_per_row)
                for i in range(set_per_row.shape[0])
                for j in range(set_per_row.shape[0])
    )
    
    results = np.array(results)
    
    return results.reshape((set_per_row.shape[0],set_per_row.shape[0]))


if __name__ == "__main__":
    
    '''
        dataset extraction
    '''
#     corpus_name, (queries_, corpus_index, target, labels) = "meter", extract_meter_to_corpus_ranking_task()
    corpus_name, (queries_, corpus_index,target, labels) = "psa",extract_short_plagiarized_answers_to_ranking()
    
    queries = [qi['content'] for qi in queries_]
    documents = [dj['content'] for dj in corpus_index]
    
    del queries_, corpus_index
    
    print("Queries: %d X Indexed Documents: %d"%(len(queries),len(documents)))
    
    '''
        using scikit-learn : tokenization
    '''    
    vectorizer = CountVectorizer(binary=True)
    all_fingerprints = vectorizer.fit_transform(queries+documents, None).T
    print('a', all_fingerprints.shape)
    
    vocabulary_indexes = [di for di in vectorizer.vocabulary_.values()]
    print("all_fingerprints: ",all_fingerprints.shape[0])



    
    true_jaccard_sim = pairwise_jaccard_similarity(set_per_row = np.vstack([i*all_fingerprints[i,:].toarray() for i in range(all_fingerprints.shape[0])]).T)
#     true_jaccard_sim = pairwise_jaccard_similarity(set_per_row = all_fingerprints.T)
    true_jaccard_sim_mean, true_jaccard_sim_std = true_jaccard_sim.mean(), true_jaccard_sim.std()
    print('true_jaccard_sim (mean,std) =(',true_jaccard_sim_mean,',',true_jaccard_sim_std,')')

    results = {}
    '''
        using scikit-learn : permutation
            each permutation has one term-document matrix
    '''    
    permutation_repetition = 1
    n_slots=3
#    permutation_count_list = [i for i in range(100,501, 100)]
    permutation_count_list = [i for i in range(10,21, 10)]

    results_file_name = "%s%sx%d"%(corpus_name,str(permutation_count_list),permutation_repetition)


    for permutation_count in permutation_count_list:
        for permutation_repetitioni in range(permutation_repetition):
            indexes_permutations = [shuffle([i for i in range(all_fingerprints.shape[0])]) for j in range(permutation_count)]          
            
            
            '''
                min_hashing and minmax_hashing for the same permutations 
                (time evaluated for each permutation)
            '''
            half_permutation_count = floor(permutation_count/n_slots) ###############################Divide em 2
            ps_permutation_count = floor(permutation_count/n_slots)
            ps_half_permutation_count = floor(permutation_count/(n_slots*2))
            
            all_min_finger    = np.empty((1,all_fingerprints.shape[1],permutation_count),np.int)
            print('a', (1,all_fingerprints.shape[1],permutation_count))
            #print('a', all_min_finger)
            all_minmax_finger = np.empty((2,all_fingerprints.shape[1],half_permutation_count),np.int)
            all_minps_finger = np.empty((2,all_fingerprints.shape[1],ps_permutation_count),np.int)
            all_minmaxps_finger = np.empty(((n_slots*2),all_fingerprints.shape[1],ps_half_permutation_count),np.int)
            
            min_time = np.zeros((permutation_count,))
            
            minmax_time = np.zeros((half_permutation_count))            
            minps_time = np.zeros((ps_permutation_count))
            minmaxps_time = np.zeros((ps_half_permutation_count))
            
            
            for i in range(permutation_count):
                t0 = time() 
                for j in range(all_fingerprints.shape[1]):
                    all_min_finger[:,j,i] = min_hashing(all_fingerprints[:,j], indexes_permutations[i])
                    
                min_time[i] = time()- t0
    
            
            for i in range(half_permutation_count):
                t0 = time() 
                for j in range(all_fingerprints.shape[1]):
                    all_minmax_finger[:,j,i] = minmax_hashing(all_fingerprints[:,j], indexes_permutations[i])
        
                minmax_time[i] = time()- t0
    
            for i in range(ps_permutation_count):
                t0 = time() 
                for j in range(all_fingerprints.shape[1]):
                    all_minps_finger[:,j,i] = minps_hashing(all_fingerprints[:,j], indexes_permutations[i])
        
                minps_time[i] = time()- t0
    
            for i in range(ps_half_permutation_count):
                t0 = time() 
                for j in range(all_fingerprints.shape[1]):
                    all_minmaxps_finger[:,j,i] = minmax_hashing_k_slot_asymetric(all_fingerprints[:,j], indexes_permutations[i], n_slots)
        
                minmaxps_time[i] = time()- t0
        
            min_jaccard_sim = pairwise_jaccard_similarity(set_per_row = all_min_finger[0,:,:])
            minmax_jaccard_sim = pairwise_jaccard_similarity(set_per_row = np.hstack([all_minmax_finger[i,:,:] for i in range(all_minmax_finger.shape[0])]))
            minps_jaccard_sim = pairwise_jaccard_similarity(set_per_row = np.hstack([all_minps_finger[i,:,:] for i in range(all_minps_finger.shape[0])]))
            minmaxps_jaccard_sim = pairwise_jaccard_similarity(set_per_row = np.hstack([all_minmaxps_finger[i,:,:] for i in range(all_minmaxps_finger.shape[0])]))
    
            prepare_results(approach_name="min              ", approach_time=min_time.sum(), approach_jaccard=min_jaccard_sim, results_dict=results, perm_repetition=permutation_repetition, perm_count=permutation_count, perm_true_jaccard=true_jaccard_sim)
            prepare_results(approach_name="minmax           ", approach_time=minmax_time.sum(), approach_jaccard=minmax_jaccard_sim, results_dict=results, perm_repetition=permutation_repetition, perm_count=permutation_count, perm_true_jaccard=true_jaccard_sim)
            prepare_results(approach_name="min   (voc. slot)", approach_time=minps_time.sum(), approach_jaccard=minps_jaccard_sim, results_dict=results, perm_repetition=permutation_repetition, perm_count=permutation_count, perm_true_jaccard=true_jaccard_sim)
            prepare_results(approach_name="minmax(voc. slot)", approach_time=minmaxps_time.sum(), approach_jaccard=minmaxps_jaccard_sim, results_dict=results, perm_repetition=permutation_repetition, perm_count=permutation_count, perm_true_jaccard=true_jaccard_sim)

            print("%d permutations[%d/%d]:"%(permutation_count,permutation_repetitioni,permutation_repetition))
            for approach_name, ap_dict in  results[permutation_count].items():
                squared_errors = ap_dict['errors'][permutation_repetitioni]**2
                absolute_errors = np.sqrt(squared_errors)
                print("[%s] MSE:%2.2e[+/- %2.2e] MAE:%2.2e[+/- %2.2e](in %4.2fs) "%(
                        approach_name, 
                        np.mean(squared_errors), np.std(squared_errors),
                        np.mean(absolute_errors), np.std(absolute_errors),
                        ap_dict['approach_time'][permutation_repetitioni]))
        
'''
    import matplotlib.pyplot as plt
    results_ae  = []
    results_se  = []
    results_names = []
            
    for permutation_count in results.keys():
        print("%d permutations:"%(permutation_count))
        for approach_name, ap_dict in  results[permutation_count].items():
            squared_errors = np.array(ap_dict['errors'])**2
            absolute_errors = np.sqrt(squared_errors)
            print("[%s] MSE:%2.2e[+/- %2.2e] MAE:%2.2e[+/- %2.2e](in %4.2fs) "%(
                approach_name, 
                np.mean(squared_errors), np.std(squared_errors),
                np.mean(absolute_errors), np.std(absolute_errors),
                np.array(ap_dict['approach_time']).mean()))

            results_ae.append(np.abs(ap_dict['errors']))
            results_se.append(np.array(ap_dict['errors'])**2)

            results_names.append("%d %s perm. (in %4.2fs)"%(permutation_count,approach_name.replace(' ',''),np.array(ap_dict['approach_time']).mean()))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    plt.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.4)
    
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of LSH approaches')
#     ax1.set_xlabel('Distribution')
    ax1.set_ylabel('Absolute Error')
    ax1.set_ylim(-0.1, np.array(results_ae).max()+0.1)

    # multiple box plots on one figure
    plt.boxplot(results_ae, showmeans=True)
    xtickNames = plt.setp(ax1, xticklabels=results_names)

    plt.setp(xtickNames, rotation=-90, fontsize=10)    

    plt.savefig('%s_absolute_errors'%(results_file_name))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    plt.subplots_adjust(left=0.075, right=0.95, top=0.95, bottom=0.4)
    
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of LSH approaches')
#     ax1.set_xlabel('Distribution')
    ax1.set_ylabel('Squared Error')
    ax1.set_ylim(-0.1, np.array(results_se).max()+0.1)
    
    # multiple box plots on one figure
    plt.boxplot(results_se, showmeans=True)
    xtickNames = plt.setp(ax1, xticklabels=results_names)
    plt.setp(xtickNames, rotation=-90, fontsize=10)    

    plt.savefig('%s_squared_errors'%(results_file_name))
#     plt.show()
'''
