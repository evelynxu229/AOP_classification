import math
import csv
from sklearn.model_selection import cross_val_score,LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix

def get_base(k):
    nucle_com = []
    # chars = ['A' ,'F' ,'C' ,'U', 'D', 'N', 'E', 'Q', 'G', 'H', 'L', 'I', 'K', 'O', 'M', 'P', 'R' ,'S', 'T', 'V', 'W' ,'Y']
    chars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    base = len(chars)
    end = len(chars)**k
    for i in range(0, end):
        n = i
        add = ''
        for j in range(k):
            ch = chars[n % base]
            n = int(n/base)
            add += ch
        nucle_com.append(add)
    return nucle_com


def get_kmer(seq,k):
    seq = seq.replace('U','')
    seq = seq.replace('O','')
    sequence = seq
    # print(sequence)
    
    # print(len(sequence))
    kmerbases = get_base(k)

    kmermap = {}
    for kmer in  kmerbases:
        kmermap[kmer] = 0

    for index in range(len(sequence)-k+1):
        kmermap[sequence[index:index+k]] += 1

    result = []
    for kmer in kmermap:
        result.append(kmermap[kmer])
    return result





def gram1(seq):
    feature = get_kmer(seq,1)
    return feature

def gram2(seq):
    feature = get_kmer(seq,2)
    return feature



def readseq(dataset,category,filename):
    fr = open('./dataset/' + dataset + '/' + 'result/' + category + '/sequence/' + filename,'r')
    data = fr.readlines()
    fr.close()
    seq = data[1][:-1]
    # seq = np.array(seq)
    # print(seq)
    # print(seq.shape)
    return seq

def getseqdata(dataset):
    classtarget=[]
    feature=[]
    if (dataset == 'Feng'):
        for i in range(0, 1552):
            print(dataset+' nonanti '+ str(i))
            feature1=[]
            seq=readseq(dataset,'nonanti',str(i))
            feature1.extend(gram2(seq))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 253):
            print(dataset+' anti '+ str(i))
            feature1 = []
            seq = readseq(dataset, 'anti', str(i))
            feature1.extend(gram2(seq))
            feature.append(feature1)
            classtarget.append(1)
    if (dataset == 'ZhangTrain'):
        for i in range(0, 100):
            print(dataset+' nonanti '+ str(i))
            feature1=[]
            seq=readseq(dataset,'nonanti',str(i))
            feature1.extend(gram2(seq))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 100):
            print(dataset+' anti '+ str(i))
            feature1 = []
            seq = readseq(dataset, 'anti', str(i))
            feature1.extend(gram2(seq))
            feature.append(feature1)
            classtarget.append(1)
    if (dataset == 'ZhangTest'):
        for i in range(0, 392):
            print(dataset+' nonanti '+ str(i))
            feature1 = []
            seq = readseq(dataset, 'nonanti', str(i))
            feature1.extend(gram2(seq))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 74):
            print(dataset+' anti '+ str(i))
            feature1 = []
            seq = readseq(dataset, 'anti', str(i))
            feature1.extend(gram2(seq))
            feature.append(feature1)
            classtarget.append(1)
    return feature,classtarget

def getfeature_gram1():
    feature, target = getseqdata('Feng')
    strlen = len(feature)
    file = open('./feature_data/Feng/Fengseqgram2.csv', 'a', newline='') 
    content = csv.writer(file, dialect='excel')  
    for i in range(0, strlen):
        content.writerow(feature[i])
    feature, target = getseqdata('ZhangTrain')
    strlen = len(feature)
    file = open('./feature_data/ZhangTrain/ZhangTrainseqgram2.csv', 'a', newline='') 
    content = csv.writer(file, dialect='excel') 
    for i in range(0,strlen):
        content.writerow(feature[i])
    feature, target = getseqdata('ZhangTest')
    strlen = len(feature)
    file = open('./feature_data/ZhangTest/ZhangTestseqgram2.csv', 'a', newline='')  
    content = csv.writer(file, dialect='excel')  
    for i in range(0, strlen):
        content.writerow(feature[i])


getfeature_gram1()

