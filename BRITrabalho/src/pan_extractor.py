from xml.dom import minidom
import glob
import os

class Feature:
    
    def __init__(self,xmlFeature):
        self.offset = -1
        self.length = -1

        self.offset = xmlFeature.getAttribute("this_offset")
        self.length = xmlFeature.getAttribute("this_length")


class Document:
    

    def __init__(self,xmlDocument,path):
        self.text = ''
        self.features = []
        xmlFeatures = xmlDocument.getElementsByTagName("feature")
        for xmlFeature in xmlFeatures:
            name = xmlFeature.getAttribute("name")            
            if name == "plagiarism":
                self.features.append(Feature(xmlFeature))

        fileTextpath = xmlDocument.getAttribute("reference")
        
    

        fileText = open(path+"/"+fileTextpath, 'rb')
        self.text = fileText.read()

def gerarDocuments(pathFileXml,path):
    fileXml = minidom.parse(pathFileXml)
    xmlDocuments = fileXml.getElementsByTagName("document")
    documents = []
    for xmlDocument in xmlDocuments:
        documents.append(Document(xmlDocument,path))

    return documents

def gerarDocument(pathFileXml,path):
    return gerarDocuments(pathFileXml,path)[0]




#TESTE
pathDataset =  os.path.dirname(os.path.realpath(__file__))+"/../../../pan-plagiarism-corpus-2011"

pathSuspicius = pathDataset+ "/external-detection-corpus/suspicious-document/**"
pathFileXmls = glob.glob(pathSuspicius+"/*.txt")

documents = []
suspicius = []
for pathFileXml in pathFileXmls:
    fileText = open(pathSuspicius+"/"+pathFileXml, 'rb')
    suspicius.append(fileText.read())
    
pathDocument = pathDataset + "external-detection-corpus/source-document/**"
pathFileXmls = glob.glob(pathDocument+"/*.txt")

for pathFileXml in pathFileXmls:
    fileText = open(pathSuspicius+"/"+pathFileXml, 'rb')
    documents.append(fileText.read())

