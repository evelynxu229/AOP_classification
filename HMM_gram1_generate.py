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

def gram1(pssm):
    feature1=np.zeros(20)
    for i in range(0,20):
        feature1[i]=pssm[:,i].mean()
    feature1=np.round(feature1,6)
    feature=[]
    for i in range(0,20):
        num=[]
        num.append(feature1[i])
        feature.extend(num)
    return feature

def gram2(pssm):
    feature2=np.zeros(400)
    L=len(pssm)
    for j in range(0,20):
        for k in range(0,20):
            num=0
            for i in range(0,L-1):
                num=num+pssm[i,j]*pssm[i+1,k]
            num=num/(L-1)
            index=20*j+k
            feature2[index]=num
    feature2=np.round(feature2,6)
    feature=[]
    for i in range(0,400):
        num=[]
        num.append(feature2[i])
        feature.extend(num)
    return feature


def readhmm(dataset,category,filename):
    L=0
    hmm=[]
    fr = open('./dataset/' + dataset + '/' + 'result/' + category + '/phmm_profile/' + filename+'.hhm')
    arryOlines=fr.readlines()
    filelen = len(arryOlines)

    for i in range(0, filelen):
        str = arryOlines[i]
        if (str.strip() == '#'):
            break
        L = L + 1
    L = L + 5

    for i in range(L, filelen - 3, 3):
        strhmm = arryOlines[i].strip()
        strhmm = strhmm + arryOlines[i + 1].strip()
        strhmm = strhmm + arryOlines[i + 1].strip()
        strhmm = strhmm.split()
        num = strhmm[2:22]
        for j in range(0, 20):
            if (num[j] == '*'):
                num[j] = 0
        num = [float(x) for x in num]
        for j in range(0, 20):
            if (num[j] != 0):
                num[j] = math.pow(2, (-num[j]) / 1000)
                num[j] = round(num[j], 6)
        hmm.append(num)
    hmm = np.array(hmm)
    return hmm

def gethmmdata(dataset):
    classtarget=[]
    feature=[]
    if (dataset == 'Feng'):
        for i in range(0, 1552):
            print(dataset+' nonanti '+ str(i))
            feature1=[]
            hmm=readhmm(dataset,'nonanti',str(i))
            feature1.extend(gram1(hmm))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 253):
            print(dataset+' anti '+ str(i))
            feature1 = []
            hmm = readhmm(dataset, 'anti', str(i))
            feature1.extend(gram1(hmm))
            feature.append(feature1)
            classtarget.append(1)
    if (dataset == 'ZhangTrain'):
        for i in range(0, 100):
            print(dataset+' nonanti '+ str(i))
            feature1=[]
            hmm=readhmm(dataset,'nonanti',str(i))
            feature1.extend(gram1(hmm))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 100):
            print(dataset+' anti '+ str(i))
            feature1 = []
            hmm = readhmm(dataset, 'anti', str(i))
            feature1.extend(gram1(hmm))
            feature.append(feature1)
            classtarget.append(1)
    if (dataset == 'ZhangTest'):
        for i in range(0, 392):
            print(dataset+' nonanti '+ str(i))
            feature1 = []
            hmm = readhmm(dataset, 'nonanti', str(i))
            feature1.extend(gram1(hmm))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 74):
            print(dataset+' anti '+ str(i))
            feature1 = []
            hmm = readhmm(dataset, 'anti', str(i))
            feature1.extend(gram1(hmm))
            feature.append(feature1)
            classtarget.append(1)
    return feature,classtarget

def getfeature_gram1():
    feature, target = gethmmdata('Feng')
    strlen = len(feature)
    file = open('./feature_data/Feng/Fenggram1.csv', 'a', newline='') 
    content = csv.writer(file, dialect='excel')  
    for i in range(0, strlen):
        content.writerow(feature[i])
    feature, target = gethmmdata('ZhangTrain')
    strlen = len(feature)
    file = open('./feature_data/ZhangTrain/ZhangTraingram1.csv', 'a', newline='')  
    content = csv.writer(file, dialect='excel')  
    for i in range(0,strlen):
        content.writerow(feature[i])
    feature, target = gethmmdata('ZhangTest')
    strlen = len(feature)
    file = open('./feature_data/ZhangTest/ZhangTestgram1.csv', 'a', newline='')  
    content = csv.writer(file, dialect='excel')  
    for i in range(0, strlen):
        content.writerow(feature[i])




getfeature_gram1()

