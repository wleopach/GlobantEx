import glob
import sys, csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from scipy.special import psi
import xml.etree.ElementTree as ET
import numpy as n
import matplotlib.pyplot as plt

#Cargamos los nombres de todos los archivos que estan en la carpeta AbstractsClass
filenames = glob.glob("AbstractsClassS\*.xml")


#La siguiente funcion devuelve el abstract de un archivo que se le introduzca.
def getabstract(filename):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~123456789'''
    # print "getting file " + filename
    tree = ET.parse(filename, parser= ET.XMLParser(encoding = 'iso-8859-5'))
    root = tree.getroot()
    doc = " "
    for i in root[0]:
        if i.tag == 'AbstractNarration' and i.text is not None:
            doc = i.text
            doc = doc.replace('<br/>', ' ')
            doc = doc.replace('т\x80\x99',' ')
            doc = doc.replace('т\x80\x94', ' ')
            doc = doc.replace('т\x80\x9c', ' ')
            doc = doc.replace('т\x80\x9d', ' ')
            no_punt = ""
            for char in doc:
                if char not in punctuations:
                    no_punt = no_punt + char
            doc = no_punt
            doc = doc.lower()
            doc = ' '.join([word for word in doc.split() if word not in stopwords.words('english')])
    # print doc
    return doc
#la siguiente funci'on devuelve el título de un artícilo dado


def gettitle(filename):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~123456789'''
    tree = ET.parse(filename, parser=ET.XMLParser(encoding='iso-8859-5'))
    root = tree.getroot()
    doc = " "
    for i in root[0]:
        if i.tag == 'AwardTitle' and i.text is not None:
            doc = i.text
            doc = doc.lower()
            doc = ' '.join([word for word in doc.split() if word not in stopwords.words('english')])
            no_punt = ""
            for char in doc:
                if char not in punctuations:
                    no_punt = no_punt + char
            doc = no_punt
    # print doc
    return doc

#La siguiente funcion  devuelve una lista con todos los abstracts no vacios
def getalldocs():
    files = filenames
    docs = []
    for file in files:
        doc = getabstract(file)
        # print doc
        if doc != " ":
            docs.append(doc)
    # print docs
    return docs


#La siguiente funci'on devuelve todos los titulos del premio de un articulo dado

def getalltitles():
    files = filenames
    titles = []
    for file in files:
        tit = gettitle(file)
        doc = getabstract(file)
        # print doc
        if doc != " ":
            titles.append(tit)
    # print docs
    return titles

def dirichlet_expectation(alpha):
    '''see onlineldavb.py by Blei et al'''
    if (len(alpha.shape) == 1):
        return (psi(alpha) - psi(n.sum(alpha)))
    return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])


def beta_expectation(a, b, k):
    mysum = psi(a + b)
    Elog_a = psi(a) - mysum
    Elog_b = psi(b) - mysum
    Elog_beta = n.zeros(k)
    Elog_beta[0] = Elog_a[0]
    # print Elog_beta
    for i in range(1, k):
        Elog_beta[i] = Elog_a[i] + n.sum(Elog_b[0:i])
    # print Elog_beta
    # print Elog_beta
    return Elog_beta


def parseDocument(doc, vocab):
    wordslist = list()
    countslist = list()
    doc = doc.lower()
    tokens = wordpunct_tokenize(doc)

    dictionary = dict()
    for word in tokens:
        if word in vocab:
            wordtk = vocab[word]
            if wordtk not in dictionary:
                dictionary[wordtk] = 1
            else:
                dictionary[wordtk] += 1

    wordslist.append(dictionary.keys())
    countslist.append(dictionary.values())
    return (wordslist[0], countslist[0])


def getVocab(file):
    '''getting vocab dictionary from a csv file (nostopwords)'''
    vocab = dict()
    with open(file, 'r') as infile:
        reader = csv.reader(infile)
        for index, row in enumerate(reader):
            vocab[row[0]] = index

    return vocab


def plottrace(x, Y, K, n, perp):
    for i in range(K):
        plt.plot(x, Y[i], label="Topic %i" % (i + 1))

    plt.xlabel("Number of Iterations")
    plt.ylabel("Probability of Each topic")
    plt.legend()
    plt.title("Trace plot for topic probabilities")
    plt.savefig("temp/plot_%i_%i_%f.png" % (K, n, perp))
