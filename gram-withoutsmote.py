import csv
import math
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV,LeaveOneOut
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
amino=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
#读取蛋白质序列字符串

def getsequencedata(dataset):
    fr=open(dataset)
    arryOlines=fr.readlines()#读取多行文件
    numberOflines=len(arryOlines)
    classstr=[]
    classtarget=[]
    if(dataset=='D1data.txt'):
        for i in range(42):
            classtarget.append(0.0)
        for i in range(95):
            classtarget.append(1.0)
    elif(dataset=='D2data.txt'):
        for i in range(87):
            classtarget.append(0.0)
        for i in range(217):
            classtarget.append(1.0)
    elif (dataset == 'D3data.txt'):
        for i in range(13):
            classtarget.append(0.0)
        for i in range(51):
            classtarget.append(1.0)
    for i in range(1,numberOflines,2):
        str=arryOlines[i].strip()
        classstr.append(str)
    return classstr,classtarget
#得到特征向量和标志
def gram1(dataset):
    feature=[]
    classstr,target=getsequencedata(dataset)
    L=len(classstr)
    for i in range(0,L):
        str=classstr[i]
        str=str.strip()
        strlen=len(str)
        a = [0 for _ in range(20)]
        for j in range(0, strlen):
            if(str[j]!='X'):
                k = amino.index(str[j])
                a[k] = a[k] + 1 / strlen
        feature.append(a)
    return feature,target
def gapgram2(dataset,g):
    feature = []
    classstr, target = getsequencedata(dataset)
    L = len(classstr)
    for i in range(0, L):
        str = classstr[i]
        str = str.strip()
        strlen = len(str)
        a = [0 for _ in range(400)]
        for j in range(0, strlen-g):
            if (str[j] != 'X'):
                k1= amino.index(str[j])
            if(str[j+g]!='X'):
                k2=amino.index(str[j+g])
                index=k1*20+k2
                a[index]=a[index]+1/(strlen-g)
        feature.append(a)
    return feature, target
def test(g):

    feature1,target=gram1('D2data.txt')
    feature2,target=gapgram2('D2data.txt',g)
    feature=[]
    for i in range(0,304):
        num=[]
        num.extend(feature1[i])
        num.extend(feature2[i])
        feature.append(num)

    train_x=feature
    train_y=np.array(target)
    biaozhun = RobustScaler()
    train_x = biaozhun.fit_transform(train_x)
    from sklearn.svm import SVC
    '''
    maxscore = []
    for i in range(-3, -15, -2):
        for j in range(-3, 15, 2):
            g = math.pow(2, i)
            c = math.pow(2, j)
            clf = SVC(C=c, gamma=g, kernel='rbf', probability=True, random_state=7)
            score = cross_val_score(clf, train_x, train_y, cv=LeaveOneOut()).mean()
            print(i, j, score)
            maxscore.append(score)
    print(max(maxscore))
    '''


    loo = LeaveOneOut()
    g = math.pow(2, -13)
    c = math.pow(2, 3)
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



test(1)

