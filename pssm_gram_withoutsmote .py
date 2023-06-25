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
def test():
    feature,target=getpssmdata('D2')
    train_x=feature
    train_y=np.array(target)
    biaozhun = RobustScaler()
    train_x = biaozhun.fit_transform(train_x)
    from sklearn.svm import SVC
    '''
    maxscore = []
    for i in range(3, -15, -2):
        for j in range(-3, 15, 2):
            g = math.pow(2, i)
            c = math.pow(2, j)
            clf = SVC(C=c, gamma=g, kernel='rbf', probability=True, random_state=7)
            score = cross_val_score(clf, train_x, train_y, cv=10).mean()
            print(i, j, score)
            maxscore.append(score)
    print(max(maxscore))
    '''

    loo = LeaveOneOut()
    g = math.pow(2, -9)
    c = math.pow(2, 1)
    clf = SVC(C=c, gamma=g, kernel='rbf', probability=True, random_state=7)
    m = np.zeros((2, 2))
    tests_y = []
    probass_y = []
    for train_index, test_index in loo.split(train_x):
        x_train, x_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y[train_index], train_y[test_index]
        clf.fit(x_train, y_train)
        predicted = clf.predict(x_test)
        if (predicted == [1] and y_test == [1]):
            cm = [[1, 0], [0, 0]]
        elif (predicted == [1] and y_test == [0]):
            cm = [[0, 0], [1, 0]]
        elif (predicted == [0] and y_test == [0]):
            cm = [[0, 0], [0, 1]]
        else:
            cm = [[0, 1], [0, 0]]
        m = m + cm
        probass_y.extend(clf.predict_proba(x_test)[:, 1])
        tests_y.extend(y_test)

    print(m)
    TP = m[0, 0]
    FN = m[0, 1]
    FP = m[1, 0]
    TN = m[1, 1]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (TP + FN)
    Specifity = TN / (TN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    print('准确率:', accuracy)
    print('敏感性：', Sensitivity)
    print('特异性:', Specifity)
    print('马修斯系数', MCC)
    print("ROC曲线下面积评分为：", roc_auc_score(tests_y, probass_y))
    fpr, tpr, thresholds = roc_curve(tests_y, probass_y)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()





test()




