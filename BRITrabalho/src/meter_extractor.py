'''
Created on 21/03/2014

@author: fellipe
'''
import os
import csv
import xml.dom.minidom
from pprint import pprint
from numpy import shape, nonzero, zeros, array
import re
from sklearn import metrics
from config import datasets_extractors

def extract_meter(content_type = 'rawtexts'):
    root = datasets_extractors['DATASETS_PATH']['meter_corpus']
    filenames_list =[
                ('courts','PA_court_files.txt'),
                ('showbiz','PA_showbiz_files.txt')
        ]
    content_type_extensions = {'rawtexts': '.txt','annotated':'.sgml'}
    
#     root = "/media/dados/Colecoes de Dados/meter_corpus"
    
    papers = {}
    
    for filenamesi in filenames_list:
        folder_path = os.path.join(root,'file_index',filenamesi[0])
        pa_file = open(os.path.join(folder_path,filenamesi[1]), 'r', encoding="utf-8")
        
        for linei in pa_file.readlines():
            linei = linei.replace('\n','')
            features = linei.split('/')
            paperi = {'path':linei, 'newspaper':features[2], 'domain':features[4], 'date':features[5], 'catchline':features[6], 'filename':features[7],'classification':'source'}
#             pprint(paperi)
# 
#             exit(0)
            with open(root.replace('/meter_corpus', linei),'r', encoding ='latin1') as content_file:
#                 content = content_file.read()
                content = re.sub(r"[0-9]","",content_file.read())
                paperi['content'] = content
                
            papers[linei] = paperi
        
#         print("folder_path:%s"%folder_path)
        
        for classification_file in ('partially_derived', 'wholly_derived', 'non_derived'):
            pa_file = open(os.path.join(folder_path,classification_file+'.txt'), 'r')
            for linei in pa_file.readlines():
                linei = linei.replace('\n','').replace('rawtexts',content_type).replace('.txt',content_type_extensions[content_type])
                features = linei.split('/')
                #meter_corpus/newspapers/rawtexts/showbiz/27.09.99/beegees/beegees119_times.txt
                newspaper_name = features[6].split('_')[1].replace('.txt','') 
                paperi = {'path':linei, 'newspaper':newspaper_name, 'domain':features[3], 'date':features[4], 'catchline':features[5], 'filename':features[6],'classification':classification_file}
    
#                 print(classification_file)
#                 print(root.replace('meter_corpus', linei))
#                 exit(0) 
                with open(root.replace('meter_corpus', linei).replace(' ',""),'r', encoding ='latin1') as content_file:
#                     content = content_file.read()
                    content = re.sub(r"[0-9]","",content_file.read())
                    paperi['content'] = content
                    
                papers[linei] = paperi
    return papers


def extract_meter_to_corpus_target_labels():
    papers = extract_meter()
    corpus = []
    target = []
    labels = [] 
    
    for keyi in papers:
#         print(keyi)
        class_name = papers[keyi]["domain"] +"_"+ papers[keyi]["classification"]
        if class_name in labels:
            class_index = labels.index(class_name, )
        else:
            class_index = len(labels)
            labels.append(class_name)
        target.append(class_index)
        corpus.append(papers[keyi]['content'])
        
#         print(X)
#         print(target)
#         print(labels)
#         exit()    

          
    return corpus, target, labels

def extract_meter_to_corpus_ranking_task():
    papers = extract_meter()
    queries = []
    corpus_index = []
    
    labels = ['index','queries']
    path_to_index = {'index':{},'queries':{}} 
    
    for keyi in papers:
        
#         if (papers[keyi]['classification'] == 'non_derived'):
#             continue
#             print(papers[keyi])
#             exit()
            
        if (papers[keyi]["newspaper"] == 'PA'):
            indexi = len(corpus_index) 
            documenti = {'extra_fields':{}}
            for doc_fields_keyi in papers[keyi].keys():
                if doc_fields_keyi == 'content':
                    documenti[doc_fields_keyi] = papers[keyi][doc_fields_keyi]
                else:
                    documenti['extra_fields'][doc_fields_keyi] = papers[keyi][doc_fields_keyi]
                    
            documenti['extra_fields']['_id'] = len(corpus_index)
            documenti['extra_fields']['root_path'] = '/'.join(keyi.split('/')[-4:-1])                    
            corpus_index.append(documenti)
            doc_type = 'index'
            path_to_index[doc_type][keyi] = indexi  
        else:
            if papers[keyi]['classification'] != 'non_derived':
                indexi = len(queries)  
                queryj = {'extra_fields':{}}
                for doc_fields_keyi in papers[keyi].keys():
                    if doc_fields_keyi == 'content':
                        queryj[doc_fields_keyi] = papers[keyi][doc_fields_keyi]
                    else:
                        queryj['extra_fields'][doc_fields_keyi] = papers[keyi][doc_fields_keyi]
                queryj['extra_fields']['_id'] = len(queries)
                queryj['extra_fields']['root_path'] = '/'.join(keyi.split('/')[-4:-1])
                queries.append(queryj)
                doc_type = 'queries'
            
                path_to_index[doc_type][keyi] = indexi
#     print(path_to_index)
#     print(len(path_to_index['index']),len(path_to_index['queries']))
#     print(shape(queries))
#     print(shape(corpus_index))
    
    target = zeros((len(queries),len(corpus_index))) #[[0]*len(corpus_index)]*len(queries)
#     print(shape(target))
    
    for query_keyi in path_to_index['queries']:
        relevant_pathi = query_keyi.replace(papers[query_keyi]["filename"],'').replace('newspapers','PA')
        query_indexi = path_to_index['queries'][query_keyi]
        count = 0
        for index_keyi in path_to_index['index']:
            if relevant_pathi in index_keyi:
#                 print('match : ', query_keyi, "=>",relevant_pathi,' x ', index_keyi)                
                relevant_indexi = path_to_index['index'][index_keyi]
                target[query_indexi][relevant_indexi] = 1
                count += 1
#         print(nonzero(target[query_indexi]))
#         print("%s = %d != %s ? total(%s)"%(query_keyi,count,shape(nonzero(target[query_indexi])),shape(target[query_indexi])))
            
#     exit(0)
    return queries, corpus_index, target, labels

def extract_meter_news_relations(leave_out = None):
    """
        leave_out : "courts" will leave out courts news, "showbiz" will leave out showbiz news otherwise nothing will be left out
         
        groups PA news from the same folder to form the source news from which the suspect news (newspapers news) must be matched searching for journalistic plagiarism.
        
        265 source news, 739 suspect news
        
        123820 court pairs + 8100 showbiz pairs = 131920 pairs to compare
    """
    
    queries, corpus_index, target, labels = extract_meter_to_corpus_ranking_task()
    
    source_news = {}
    pairs_to_compare = []
    
    for i in range(target.shape[1]):
        ci = corpus_index[i]
        if ci['extra_fields']['root_path'].split('/')[0] == leave_out:
            continue
        
        if not ci['extra_fields']['root_path'] in source_news.keys():
            source_news[ci['extra_fields']['root_path']] = ""

            """
                compairing with the joined source news (all queries will be compared just once)
            """
            for j in range(target.shape[0]):
                qj = queries[j]
                contentj = qj['content']
                if ci['extra_fields']['root_path'].split('/')[0] == qj['extra_fields']['root_path'].split('/')[0]:
                    pairs_to_compare.append([qj['extra_fields']['root_path'],contentj,target[j,i]])

        source_news[ci['extra_fields']['root_path']] += ci['content']
        
    del queries, corpus_index, target, labels
    
    for i in range(len(pairs_to_compare)):
        pairs_to_compare[i][0] = source_news[pairs_to_compare[i][0]]                        
    
    return pairs_to_compare

def load_to_RetrievalDatasets(retrieval_dataset):
    queries, corpus_index,target , labels = extract_meter_to_corpus_ranking_task()
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
    
    
            
if __name__ == '__main__':
    
    news_relations = extract_meter_news_relations()
    exit()
    
    corpus, target, labels =  extract_meter_to_corpus_target_labels()
    print(labels)
    
    for i in range(0,3):
        print('o documento %d pertence a classe[%d] = %s '%(i,target[i],labels[target[i]]))
    print(len(target))
    print(len(corpus))
    
        