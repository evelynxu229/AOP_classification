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
#计算单联体的频率
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
#计算二联体的频率
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
#计算三联体频率
def gram3(pssm):
    feature3=np.zeros(8000)
    L=len(pssm)
    for i in range(0,20):
        for j in range(0,20):
            for k in range(0,20):
                num=0
                for s in range(0,L-2):
                    num=num+pssm[s,i]*pssm[s+1,j]*pssm[s+2,k]
                num=num/(L-2)
                index=400*i+20*j+k
                feature3[index]=num
    feature3=np.round(feature3,6)
    feature=[]
    for i in range(0,8000):
        num=[]
        num.append(feature3[i])
        feature.extend(num)
    return feature
#读取PSSM矩阵
def readpssm(dataset,category,filename):
    L=0
    pssm=[]
    fr = open('D:/datasets/GPTpred/' + dataset + '/' + 'result/' + category + '/pssm_profile_uniref50/' + filename)
    arryOlines=fr.readlines()#读取多行文件
    #判断序列长度
    for i in range(3,len(arryOlines)):
        str=arryOlines[i]
        if(str.strip()==""):
            break
        L=L+1
    for i in range(3,3+L):
       strpssm=arryOlines[i].strip()
       strpssm=strpssm.split()
       num=strpssm[22:42]
       num=[float(x) for x in num]
       pssm.append(num)
    pssm=np.array(pssm)
    pssm=pssm/100
    return pssm
def getpssmdata(dataset):
    classtarget=[]
    feature=[]
    if (dataset == 'D1'):
        for i in range(0, 42):
            feature1=[]
            pssm=readpssm(dataset,'cis',str(i))
            feature1.extend(gram1(pssm))
            feature1.extend(gram2(pssm))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 94):
            feature1 = []
            pssm = readpssm(dataset, 'trans', str(i))
            feature1.extend(gram1(pssm))
            feature1.extend(gram2(pssm))
            feature.append(feature1)
            classtarget.append(1)
    if (dataset == 'D2'):
        for i in range(0, 87):
            feature1=[]
            pssm=readpssm(dataset,'cis',str(i))
            feature1.extend(gram1(pssm))
            feature1.extend(gram2(pssm))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 217):
            feature1 = []
            pssm = readpssm(dataset, 'trans', str(i))
            feature1.extend(gram1(pssm))
            feature1.extend(gram2(pssm))
            feature.append(feature1)
            classtarget.append(1)
    if (dataset == 'D3'):
        for i in range(0, 13):
            feature1 = []
            pssm = readpssm(dataset, 'cis', str(i))
            feature1.extend(gram1(pssm))
            feature1.extend(gram2(pssm))
            feature.append(feature1)
            classtarget.append(0)
        for i in range(0, 51):
            feature1 = []
            pssm = readpssm(dataset, 'trans', str(i))
            feature1.extend(gram1(pssm))
            feature1.extend(gram2(pssm))
            feature.append(feature1)
            classtarget.append(1)
    return feature,classtarget

