from __future__ import division
import os
import re, sys
import random
import time
import binascii
import json
import ConfigParser

import sys, time
ini = time.time() ### Utilizo para contar o tempo de execucao
sys.path.append('/home/joaquim/Área de Trabalho/Mestrado/BRI/Plagio/modulo/functions')
from lsh import WeightedMinHashLSH, MinHashLSH
from hashlib import sha1
import numpy as np
from minhash import MinHash


resultado = [] # Vetor onde o resultado sera armazenado

pasta_master = unicode('/home/joaquim/Área de Trabalho/Mestrado/BRI/Plagio', 'utf-8') #Pasta Master do projeto


############## Leitura do arquivo BUSCA.CFG #######################################################################################

caminho_arq_conf = pasta_master+ '/modulo/PARAMETROS.CFG'               # Obtem caminho do arquivo de configuracao    
config = ConfigParser.ConfigParser()
config.read(caminho_arq_conf)                                           # Executa a leitura do arquivo PARAMETROS.CFG

# input files 
numHashes = int(config.get('Parametros','NUM_HASHES'))                     # Numero de funcoes Hashes (K)
threshold = float(config.get('Parametros','THRESHOLD'))                        # Limete de corte
metrica = config.get('Parametros','METRICA')                            # Mininimo ou maximo ou mediana hashing
pasta_leitura = config.get('InputFiles','PASTA_LEIA')                   # Pasta que estao os arquivos a serem carregados
nome_catalogo_arquivos = config.get('InputFiles','LEIA')                # Arquivo que armazena todas os arquivos a serem lidos



############## Obtem file-names dos arquivo que contem os dados ###################################################################

cat_file_name = pasta_master+pasta_leitura+nome_catalogo_arquivos
with open(cat_file_name) as json_file:
    json_data = json.load(json_file)    

qtdeDocs = len(json_data)                                               # Obtem a quantidade de documentos


############## Executa a leitura dos arquivos e faz o Shingling ####################################################################
print "Executa a leitura dos arquivos e faz o Shingling"
t0 = time.time()                                                        # Salvo para calcular o tempo de execucao do Shingling
totalShingles = 0                                                       # Salvo a quantidade total de Shingling em todos os documentos

sets_shingle_docs = {}
docIDs = []
setArquivos = {}
dados = {}

for docID, line in enumerate(json_data):                                  # Loop ate obter todos os arquivos
    
    file_name = ( str(line['plag_type'])+'/'+ line['document'])
    dataFile =(pasta_master+pasta_leitura+file_name)                    # Gera o file-name de cada arquivo
          
    f = open(dataFile, "rU")                                            # Executa a leitura dos dados
    words = f.read().replace('\n', '').split(" ")                       # Limpa o texto e mantem apenas as palavras    
                                                          
    setArquivos[docID]= line['document']                                # Salvo o nome dos arquivos
    docIDs.append(docID)       
    
    shinglesInDoc = set()                                               # Armazena todos os Shingles
    palavras = set()
    
    for index in range(0, len(words) - 2):                              # Para cada palavra no documento...
        
        shingle = words[index] + " " + words[index + 1] + " " + words[index + 2] # Shingle composto por 3 palavras
        palavras.add(shingle)
        crc = binascii.crc32(shingle) & 0xffffffff                      # Hash o shingle para um inteiro de 32-bit.
        shinglesInDoc.add(crc)
    
    sets_shingle_docs[docID] = shinglesInDoc                            # Armazena a lista de shingles
    dados[docID] = palavras
    
    totalShingles = totalShingles + (len(words) - 2)
    
    f.close()                                                           # Fecha o arquivo
    
# Informa dados da execucao do shingling.
print '\nO shingling de ' + str(qtdeDocs) + ' documentos levou %.2f sec.' % (time.time() - t0)
print '\nMedia de shingles por documentos: %.2f' % (totalShingles / qtdeDocs)
print 'Shingling Finalizado com sucesso'


####################################################################################################
################LSH 

lista = []

lsh = MinHashLSH(threshold=0.1, num_perm=numHashes) # Threshold usado para gerar o index e encontrar possiveis similares

for idx in range(len(dados)):
    m = MinHash(num_perm=numHashes, tipo=metrica)
    lista.append(m)
    for d in dados[idx]:
        lista[idx].update(d)
    #print lista[idx].hashvalues
#   print lista[1]
# Create LSH index
    lsh.insert(idx, lista[idx])
    
    #result = lsh.query(lista[idx], setArquivos[idx])
    #index_lsh[idx] = result
    #print(setArquivos[idx],"Approximate neighbours with Jaccard similarity > 0.4", result)
    #print(setArquivos[idx],"", result)


for idx in range(0, len(dados)):
    result = lsh.query(lista[idx], idx)
    print '\n##########################################################'
    print idx, ' --> ', result









































  
############################################################################################################################################
# =============================================================================
#                   Calculando a similaridade entre os documentos
# =============================================================================  
ini = time.time()
print "Similaridade de Jaccard.\n"
print "\nObtem a lista de documentos com J(d1,d2) maior ou igual a que ", threshold


for i in range(0, qtdeDocs):                                            # Loop para cada documento...
  for j in range(i + 1, qtdeDocs):
    # Calcula a Similaridade de Jaccard
    s1 = sets_shingle_docs[docIDs[i]]
    s2 = sets_shingle_docs[docIDs[j]]
    jaccard = (len(s1.intersection(s2)) / len(s1.union(s2)))
    
    if jaccard >= threshold:
        #print "  %10s <--> %10s        %.2f" % (setArquivos[i], setArquivos[j], J )
        #resultado.append((setArquivos[i], setArquivos[j], jaccard ))
        resultado.append((i, j, jaccard ))
          
print "Jaccard Similaridade finalizada em ", round((time.time() - ini),2), ' segundos'



############################################################################################################################################
# =============================================================================
#                   Salvando os resultados
# =============================================================================
print '\nExibindo os resultados'
resultado = sorted(resultado,key = lambda elem: elem[2], reverse=True) # Ordena os resultados
for doc1, doc2, jaccard in resultado:
    print "  %10s <--> %10s        %.2f" % (doc1, doc2, jaccard )
    a=1
    










