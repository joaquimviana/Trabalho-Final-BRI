import os
from pprint import pprint
import json
from numpy import zeros, nonzero
from config import *

def extract_short_plagiarized_answers_to_ranking():
    path = datasets_extractors['DATASETS_PATH']['short_plagiarised_answers_dataset']

    json_data=open(os.path.join(path,'query_answer_json'))

    data = json.load(json_data)
#     pprint(data)
    json_data.close()
    
    files_by_task = {}
    
    doc_count = 0
    for datai in data:
        if datai['plag_type']=='non':
            continue
        else:
            doc_count += 1
        try:
            task_files = files_by_task[datai['task']]
        except:
            task_files = []
            files_by_task[datai['task']] = task_files 
            
        filei_path = os.path.join(path,datai['plag_type'],datai['document'])
        f = open(filei_path ,'rb')
        task_files.append(r"%s"%f.read())
        f.close()
        
    corpus_index = []
    queries = []
    target = zeros((doc_count,len(files_by_task.keys()) ),)
    labels = []
    for keyi in files_by_task.keys():
        labels.append(keyi)
        source_path = os.path.join(path,"source","orig_task%s.txt"%(keyi))
        f = open(source_path ,'rb')
        doci = {'content':r"%s"%f.read(), 'extra_fields':{'task':keyi,'_id':len(corpus_index)},}
        corpus_index.append(doci)
        f.close()
        
        last_pos = len(queries)
        for query_contenti in files_by_task[keyi]:            
            queries.append({'content':query_contenti, 'extra_fields':{'_id':len(queries)}})
            
        for i in range(last_pos, len(queries)):
            target[i][len(corpus_index)-1] += 1
            
    return queries, corpus_index, target, labels

def load_to_RetrievalDatasets(retrieval_dataset):
    queries, corpus_index,target , labels = extract_short_plagiarized_answers_to_ranking()
#     print(target.shape)
        
    doc_ids = retrieval_dataset.addDocuments(corpus_index)
    
    del corpus_index
    
    for i in range(0,target.shape[0]):
        queryi = queries[i]
        old_indexes = nonzero(target[i])[0]
        relevants = []
        try:
            for indexi in old_indexes:
                relevants.append(doc_ids[indexi])
        except:
            relevants += [doc_ids[old_indexes]]
        query_id = retrieval_dataset.addQuery(queryi,relevants)

    del queries
    del target
    
if __name__ == "__main__":
    queries, corpus_index,target, labels = extract_short_plagiarized_answers_to_ranking()
    print(len(queries))
    print(len(corpus_index))
    print(target.shape)
    print(target)
