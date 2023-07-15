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
# from imblearn.over_sampling import SMOTE
import os

    
def getACCfeature(hmm,g):
    strlen=len(hmm)
    ACCfeature=[]
    for i in range(0,20):  
         avg1=sum(hmm[:,i])/strlen
         for k in range(0,20):
             feature=[]
             avg2=sum(hmm[:,k])/strlen
             for d in range(1,g):
                 num=0
                 f=[]
                 for j in range(0,strlen-d):  
                     num=num+(hmm[j][i]-avg1)*(hmm[j+d][k]-avg2)
                 num=num/(strlen-d)
                 f.append(num)
                 feature.extend(f)
             ACCfeature.extend(feature)
    return ACCfeature

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

def gethmmdata(dataset,d):
    classtarget=[]
    feature=[]
    if (dataset == 'Feng'):
        for i in range(0, 1552):
            print(dataset+' nonanti '+ str(i))
            feature1=[]
            hmm=readhmm(dataset,'nonanti',str(i))
            feature1.extend(getACCfeature(hmm,d))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 253):
            print(dataset+' anti '+ str(i))
            feature1 = []
            hmm = readhmm(dataset, 'anti', str(i))
            feature1.extend(getACCfeature(hmm,d))
            feature.append(feature1)
            classtarget.append(1)
    if (dataset == 'ZhangTrain'):
        for i in range(0, 100):
            print(dataset+' nonanti '+ str(i))
            feature1=[]
            hmm=readhmm(dataset,'nonanti',str(i))
            feature1.extend(getACCfeature(hmm,d))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 100):
            print(dataset+' anti '+ str(i))
            feature1 = []
            hmm = readhmm(dataset, 'anti', str(i))
            feature1.extend(getACCfeature(hmm,d))
            feature.append(feature1)
            classtarget.append(1)
    if (dataset == 'ZhangTest'):
        for i in range(0, 392):
            print(dataset+' nonanti '+ str(i))
            feature1 = []
            hmm = readhmm(dataset, 'nonanti', str(i))
            feature1.extend(getACCfeature(hmm,d))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 74):
            print(dataset+' anti '+ str(i))
            feature1 = []
            hmm = readhmm(dataset, 'anti', str(i))
            feature1.extend(getACCfeature(hmm,d))
            feature.append(feature1)
            classtarget.append(1)
    return feature,classtarget

def getfeature(d):
    feature, target = gethmmdata('Feng',d)
    strlen = len(feature)
    file = open('./feature_data/Feng/'+str(d)+'Fengacc.csv', 'a', newline='') 
    content = csv.writer(file, dialect='excel') 
    for i in range(0, strlen):
        content.writerow(feature[i])
    file.close()
    
    feature, target = gethmmdata('ZhangTrain',d)
    strlen = len(feature)
    file = open('./feature_data/ZhangTrain/'+str(d)+'ZhangTrainacc.csv', 'a', newline='') 
    content = csv.writer(file, dialect='excel')  
    for i in range(0,strlen):
        content.writerow(feature[i])
    file.close()

    feature, target = gethmmdata('ZhangTest',d)
    strlen = len(feature)
    file = open('./feature_data/ZhangTest/'+str(d) + 'ZhangTestacc.csv', 'a', newline='')  
    content = csv.writer(file, dialect='excel')  
    for i in range(0, strlen):
        content.writerow(feature[i])
    file.close()
    
for d in range(1,12):
    getfeature(d)