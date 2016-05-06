from __future__ import division
import os
import re, sys
import random
import time
import binascii
import json
import ConfigParser

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

for docID, line in enumerate(json_data):                                  # Loop ate obter todos os arquivos
    
    file_name = ( str(line['plag_type'])+'/'+ line['document'])
    dataFile =(pasta_master+pasta_leitura+file_name)                    # Gera o file-name de cada arquivo
          
    f = open(dataFile, "rU")                                            # Executa a leitura dos dados
    words = f.read().replace('\n', '').split(" ")                       # Limpa o texto e mantem apenas as palavras    
                                                          
    setArquivos[docID]= line['document']                                # Salvo o nome dos arquivos
    docIDs.append(docID)                                                # Salvo o index no vetor de documentos
    
    
    shinglesInDoc = set()                                               # Armazena todos os Shingles
    
    for index in range(0, len(words) - 2):                              # Para cada palavra no documento...
        
        shingle = words[index] + " " + words[index + 1] + " " + words[index + 2] # Shingle composto por 3 palavras
        
        crc = binascii.crc32(shingle) & 0xffffffff                      # Hash o shingle para um inteiro de 32-bit.
        shinglesInDoc.add(crc)
    
    sets_shingle_docs[docID] = shinglesInDoc                            # Armazena a lista de shingles

    
    totalShingles = totalShingles + (len(words) - 2)
    
    f.close()                                                           # Fecha o arquivo
    
# Informa dados da execucao do shingling.
print '\nO shingling de ' + str(qtdeDocs) + ' documentos levou %.2f sec.' % (time.time() - t0)
print '\nMedia de shingles por documentos: %.2f' % (totalShingles / qtdeDocs)
print 'Shingling Finalizado com sucesso'



######################################################################################################################
# =============================================================================
#                 Generate MinHash Signatures
# =============================================================================
print '\nGerando funcoes Aleatorias...'
t0 = time.time()                                                        # Salvo para calcular o tempo de Geracao das Assinaturas

maxShingleID = 2**32-1                                                  # Armazena o valor maximo de shingle ID
nextPrime = 4294967311                                                  # Proximo valor primo maior que o maxShingleID


# Our random hash function will take the form of:
#   h(x) = (a*x + b) % c
# Where 'x' is the input value, 'a' and 'b' are random coefficients, and 'c' is
# a prime number just greater than maxShingleID.

### Gera uma lista de K coeficientes aleatorios para as funcoes aleatorias de hash
def pickRandomCoeffs(k):
  randList = []                                                         # Lista de 'K' valores aleatorios.
  
  while k > 0:    
    randIndex = random.randint(0, maxShingleID)                         # Obtem um shingle ID aleatorio.
  
    while randIndex in randList:                                        # Garante que cada ID aleatorio e unico
      randIndex = random.randint(0, maxShingleID) 
    
    randList.append(randIndex)                                          # Adiciona um numero aleatorio a lista
    k -= 1    
  return randList

# Para cada 'numHashes' functions e gerado um coeficiente diferente 'a' e 'b'.
coeffA = pickRandomCoeffs(numHashes)
coeffB = pickRandomCoeffs(numHashes)

print '\nGerando assinaturas para todos os documentos...'
print '\nMetrica utilizada: '+metrica

# List of documents represented as signature vectors
signatures = []                                                         # Armazena as assinaturas da Listas de documentos 

for docID in docIDs:                                                    # Loop em todos os documentos
  
  shingleIDSet = sets_shingle_docs[docID]                               # Obtem o conjunto de shingles de cada documento
  signature = []                                                        # Obtem o resultado da assinatura para cada documento
  
  for i in range(0, numHashes):                                         # Loop em todas as funcoes aleatorias

    
    # For each of the shingles actually in the document, calculate its hash code
    # using hash function 'i'. 
    
    # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
    # the maximum possible value output by the hash.
    metricaHashCode = nextPrime + 1                                     # Inicialmente armazena o valor maximo possivel de hash
    if metrica == 'MAXIMO': metricaHashCode = 0                         # Caso seja escolhido o Maximo hash code
    
    for shingleID in shingleIDSet:                                      # Loop em todos os Shingles      
      hashCode = (coeffA[i] * shingleID + coeffB[i]) % nextPrime        # Gera as Funcoes Aleatorias
      # Track the lowest hash code seen.
      
      if(metrica == 'MINIMO'):
          if hashCode < metricaHashCode:
              metricaHashCode = hashCode
      elif(metrica == 'MAXIMO'):
          if hashCode > metricaHashCode:
              metricaHashCode = hashCode
      else:
          print 'Parar implementar a mediana'
    
    signature.append(metricaHashCode)
  
  signatures.append(signature)                                          # Armazena a assinatura da metrica de Hashs para todos os documentos.

print "\nA geracao das assinaturas levou %.2fsec" % (time.time() - t0)   
  
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
          #print "  %10s --> %10s        %.2f" % (setArquivos[i], setArquivos[j], J )
          resultado.append((setArquivos[i], setArquivos[j], jaccard ))
          
print "Jaccard Similaridade finalizada em ", round((time.time() - ini),2), ' segundos'



############################################################################################################################################
# =============================================================================
#                   Salvando os resultados
# =============================================================================
print '\nExibindo os resultados'
resultado = sorted(resultado,key = lambda elem: elem[2], reverse=True) # Ordena os resultados
for doc1, doc2, jaccard in resultado:
    print "  %10s --> %10s        %.2f" % (doc1, doc2, jaccard )










