from os.path import os
mongo_config = {
    'SAFE_MONOGOCALL_CONFIG' : {
                   'thread_sleep_time':1, # seconds!!
                   'retry_attempts':60
    },
    'server_info':{
        'database_server':"localhost",#"146.164.35.51",
        'server_port': 27017, 
        'max_pool_size' :200               
    }
}

# datasets_root_path = "/home/duartefellipe/Documents/Datasets"
# datasets_root_path = "/home/fellipe/Documents/Datasets"
path = os.path.dirname(os.path.realpath(__file__))
datasets_root_path = path+"/../Datasets"

datasets_extractors = {
    "DATASETS_PATH" : {
            'short_plagiarised_answers_dataset': os.path.join(datasets_root_path,'short plagiarised answers corpus/corpus-20090418'),
            'meter_corpus':os.path.join(datasets_root_path,'meter_corpus'),
             'co_derivative':os.path.join(datasets_root_path,'PAN/co-derivative/wik-coderiv-corpus-original'),
             'pan_plagiarism_corpus_2011': os.path.join(datasets_root_path,'PAN/Plagiarism detection/pan-plagiarism-corpus-2011'),
             'cf': os.path.join(datasets_root_path,'Common IR collections /Cystic Fibrosis/cfc-xml'),
             'cranfield':os.path.join(datasets_root_path,'Common IR collections /cranfield'),
             'P4P':os.path.join(datasets_root_path,'P4P_corpus'),
             'cpc-11':os.path.join(datasets_root_path,'corpus-webis-cpc-11'),
             'MSRParaphrase':os.path.join(datasets_root_path,'MSRParaphraseCorpus'),
             'NerACL2010':os.path.join(datasets_root_path,'NerACL2010_Experiments/Data/WordEmbedding'),
    }
}

STANFORD_PARSER ={
    "encoding": 'utf-8',  # linux
#     "encoding": 'utf-16',  # linux
#     "encoding": 'cp1252', # windows
    "java_options":'-mx3000m'              
}


