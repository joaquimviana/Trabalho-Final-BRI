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
from pan_extractor import recoverOnlyText
from pairwise_jaccard_comparison import *


def generateResult(permutation_count,all_fingerprints,function_out_size,indexes_permutations,hasing_function,n_slots = -1):
   
    all_minps_finger = np.empty((function_out_size,all_fingerprints.shape[1],permutation_count),np.int)
    minps_time = np.zeros((permutation_count))
   
    for i in range(permutation_count):
                t0 = time()
                for j in range(all_fingerprints.shape[1]):
                    if(n_slots == -1):
                        all_minps_finger[:,j,i] = hasing_function(all_fingerprints[:,j], indexes_permutations[i])
                    else:
                        all_minps_finger[:,j,i] = hasing_function(all_fingerprints[:,j], indexes_permutations[i],n_slots)                   
                   
                minps_time[i] = time()- t0
   
    return all_minps_finger,minps_time


def printResult(resultName,all_minmaxps_finger,minps_time,results,permutation_count):
    minmaxps_jaccard_sim = pairwise_jaccard_similarity(set_per_row = np.hstack([all_minmaxps_finger[i,:,:] for i in range(all_minmaxps_finger.shape[0])]))
    prepare_results(approach_name=resultName, approach_time=minps_time.sum(), approach_jaccard=minmaxps_jaccard_sim, results_dict=results, perm_repetition=permutation_repetition, perm_count=permutation_count, perm_true_jaccard=true_jaccard_sim)   



if __name__ == "__main__":
    
   
    #all_fingerprints,vectorizer = panDatasetToFingerprint()
    all_fingerprints,vectorizer = short_plagiarizedToFingerprint()
    
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
    n_slots=4
#    permutation_count_list = [i for i in range(100,501, 100)]
    permutation_count_list = [i for i in range(1000,1001, 10)]

    #results_file_name = "%s%sx%d"%(corpus_name,str(permutation_count_list),permutation_repetition)
    slots_range = [2,4,8,16,20,22,24]
    for n_slots in slots_range:        
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
                
                all_minhash_finger,minhash_time = generateResult(ps_permutation_count,all_fingerprints,2,indexes_permutations,min_hashing)
                printResult("min_hashing                        ",all_minhash_finger,minhash_time,results,permutation_count)
                del all_minhash_finger,minhash_time
                
                all_maxhash_finger,maxhash_time = generateResult(ps_permutation_count,all_fingerprints,2,indexes_permutations,max_hashing)
                printResult("max_hashing                        ",all_maxhash_finger,maxhash_time,results,permutation_count)
                del all_maxhash_finger,maxhash_time
                
                all_minmaxhash_finger,minmaxhash_time = generateResult(ps_permutation_count,all_fingerprints,2,indexes_permutations,minmax_hashing)
                printResult("minmax_hashing                     ",all_minmaxhash_finger,minmaxhash_time,results,permutation_count)
                del all_minmaxhash_finger,minmaxhash_time
                
                all_minps_finger,minps_time = generateResult(ps_permutation_count,all_fingerprints,2,indexes_permutations,minps_hashing)
                printResult("minps                              ",all_minps_finger,minps_time,results,permutation_count)
                del all_minps_finger,minps_time
                
                all_minmaxps_finger,minmaxps_time = generateResult(ps_half_permutation_count,all_fingerprints,4,indexes_permutations,minmaxps_hashing)
                printResult("minmaxps                           ",all_minmaxps_finger,minmaxps_time,results,permutation_count)
                del all_minmaxps_finger,minmaxps_time
                
                all_minmaxps_n_finger,minmaxps_n_time = generateResult(ps_half_permutation_count,all_fingerprints,n_slots*2,indexes_permutations,minmaxps_hashing_n,n_slots)
                printResult("minmaxps_n                         ",all_minmaxps_n_finger,minmaxps_n_time,results,permutation_count)
                del all_minmaxps_n_finger,minmaxps_n_time
                
                all_minmax_hashing_k_slot_asymetric_finger,all_minmax_hashing_k_slot_asymetric_time = generateResult(ps_half_permutation_count,all_fingerprints,n_slots*2,indexes_permutations,minmax_hashing_k_slot_asymetric,n_slots)  
                printResult("all_minmax_hashing_k_slot_asymetric",all_minmax_hashing_k_slot_asymetric_finger,all_minmax_hashing_k_slot_asymetric_time,results,permutation_count)
                del all_minmax_hashing_k_slot_asymetric_finger,all_minmax_hashing_k_slot_asymetric_time            
                
                all_minmax_hashing_two_slot_asymetric_finger,all_minmax_hashing_two_slot_asymetric_time = generateResult(ps_half_permutation_count,all_fingerprints,4,indexes_permutations,minmax_hashing_two_slot_asymetric) 
                printResult("minmax_hashing_two_slot_asymetric  ",all_minmax_hashing_two_slot_asymetric_finger,all_minmax_hashing_two_slot_asymetric_time,results,permutation_count)
                del all_minmax_hashing_two_slot_asymetric_finger,all_minmax_hashing_two_slot_asymetric_time
                
                
                
                
    
                
                print("%d permutations[%d/%d]:"%(permutation_count,permutation_repetitioni,permutation_repetition))
                for approach_name, ap_dict in  results[permutation_count].items():
                    squared_errors = ap_dict['errors'][permutation_repetitioni]**2
                    absolute_errors = np.sqrt(squared_errors)
                    print("[%s] MSE:%2.2e[+/- %2.2e] MAE:%2.2e[+/- %2.2e](in %4.2fs) "%(
                            approach_name, 
                            np.mean(squared_errors), np.std(squared_errors),
                            np.mean(absolute_errors), np.std(absolute_errors),
                            ap_dict['approach_time'][permutation_repetitioni]))
            
    
        import matplotlib.pyplot as plt
        results_ae  = []
        results_se  = []
        results_names = []
        results_file_name = "../results/plot_" + str(n_slots)         
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
