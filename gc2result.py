import csv
import math
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
# from imblearn.over_sampling import SMOTE
# from imblearn.combine import SMOTEENN
from feature_selection.method.ANOVA import run1
from feature_selection.method.MRMD import run2

def gettarget(dataset):
    classtarget=[]
    classtarget=[]
    if (dataset == 'Feng'):
        for i in range(0, 1552):
            classtarget.append(0)
        for i in range(0, 253):
            classtarget.append(1)
    if (dataset == 'ZhangTrain'):
        for i in range(0, 100):
            classtarget.append(0)
        for i in range(0, 100):
            classtarget.append(1)
    if (dataset == 'ZhangTest'):
        for i in range(0, 392):
            classtarget.append(0)
        for i in range(0, 74):
            classtarget.append(1)
    return classtarget
def getdata(filename,X):
    feature = []
    with open(filename + ".csv") as fs:
        data = csv.reader(fs)
        for line in data:
            line = [float(x) for x in line]
            feature.append(line)
    feature = np.array(feature)
    if X == []:
        res = feature
    else:
        res = np.concatenate((X,feature),axis = 1)
    return res


def main(name,feature_extraction,feature_selection,standard,G,k,pg,pc,over_sampling):
    if name == 'Feng':
        x_train = []
        y_train=gettarget('Feng')
        y_train=np.array(y_train)
        if 'gram1' in feature_extraction:
            feature=getdata('./feature_data/Feng/Fenggram1',x_train)
            x_train=np.array(feature)
        if 'gram2' in feature_extraction:
            feature=getdata('./feature_data/Feng/Fenggram2',x_train)
            x_train=np.array(feature)
        if 'seqgram1' in feature_extraction:
            feature=getdata('./feature_data/Feng/Fengseqgram1',x_train)
            x_train=np.array(feature)
        if 'seqgram2' in feature_extraction:
            feature=getdata('./feature_data/Feng/Fengseqgram2',x_train)
            x_train=np.array(feature)
        if 'ACC' in feature_extraction:
            feature=getdata('./feature_data/Feng/'+str(G)+'Fengacc',x_train)
            x_train=np.array(feature)
        if '3gram1' in feature_extraction:
            feature=getdata('./feature_data/Feng/3Fenggram1',x_train)
            x_train=np.array(feature)
            print(x_train.shape)
            print(y_train.shape)
    elif name == 'ZhangTrain':
        x_train = []
        y_train=gettarget('ZhangTrain')
        y_train=np.array(y_train)
        if 'gram1' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/ZhangTraingram1',x_train)
            x_train=np.array(feature)
        if 'gram2' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/ZhangTraingram2',x_train)
            x_train=np.array(feature)
        if 'ACC' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/'+str(G)+'ZhangTrainacc',x_train)
            x_train=np.array(feature)
        if '3gram1' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/3ZhangTraingram1',x_train)
            x_train=np.array(feature)
            print(x_train.shape)
            print(y_train.shape)
    elif name == 'ZhangTest':
        x_train = []
        x_test = []
        # get label
        y_train=gettarget('ZhangTrain')
        y_train=np.array(y_train)
        y_test=gettarget('ZhangTest')
        y_test=np.array(y_test)
        if 'gram1' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/ZhangTraingram1',x_train)
            x_train=np.array(feature)
            feature=getdata('./feature_data/ZhangTest/ZhangTestgram1',x_test)
            x_test=np.array(feature)
        if 'gram2' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/ZhangTraingram2',x_train)
            x_train=np.array(feature)
            feature=getdata('./feature_data/ZhangTest/ZhangTestgram2',x_test)
            x_test=np.array(feature)
        if 'seqgram1' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/ZhangTrainseqgram1',x_train)
            x_train=np.array(feature)
            feature=getdata('./feature_data/ZhangTest/ZhangTestseqgram1',x_test)
            x_test=np.array(feature)
        if 'seqgram2' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/ZhangTrainseqgram2',x_train)
            x_train=np.array(feature)
            feature=getdata('./feature_data/ZhangTest/ZhangTestseqgram2',x_test)
            x_test=np.array(feature)
        if 'ACC' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/'+str(G)+'ZhangTrainacc',x_train)
            x_train=np.array(feature)
            feature=getdata('./feature_data/ZhangTest/'+str(G)+'ZhangTestacc',x_test)
            x_test=np.array(feature)
        if '3gram1' in feature_extraction:
            feature=getdata('./feature_data/ZhangTrain/3ZhangTraingram1',x_train)
            x_train=np.array(feature)
            feature=getdata('./feature_data/ZhangTest/3ZhangTestgram1',x_test)
            x_test=np.array(feature)
            print(x_train.shape)
            print(x_test.shape)
            print(y_train.shape)
            print(y_test.shape)



    if standard != '':
        if standard == 'Minmax':
            std = MinMaxScaler()
        elif standard == 'Robust':
            std = RobustScaler()
        elif standard == 'Normal':
            std = Normalizer()
        elif standard == 'Standard':
            std = StandardScaler()
        x_train = std.fit_transform(x_train)
        if name == 'ZhangTest':
            x_test = std.fit_transform(x_test)
        print('standard Successful!')

    resultname = str(name)
    if name == 'ZhangTest' or name == 'ZhangTrain':
        resultname = 'Zhang'
    for i in range(len(feature_extraction)):
        if i != 0:
            resultname += '_'
            resultname += feature_extraction[i]
        else:
            resultname += feature_extraction[i]
    if feature_selection != '':
        resultname += '_'
        resultname += feature_selection

    if over_sampling != '':
        resultname += '_'
        resultname += over_sampling

    if feature_selection != '':
        path = './feature_selection/TempFile/'+resultname+'_result/'
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path) 
    print(resultname)

    if feature_selection != '':
        if name == 'ZhangTest' or name == 'ZhangTrain':
            file = open('./feature_selection/TempFile/'+resultname+'_result/ZhangOrderList.csv', 'r', newline='') 
        else:
            file = open('./feature_selection/TempFile/'+resultname+'_result/'+name+'OrderList.csv', 'r', newline='')  
        orderlist = csv.reader(file)
        for i in orderlist:
            feature_list = [int(float(x)) for x in i]
        file.close()

        xtrain = np.zeros((len(x_train), k))
        for i in range(0, k):
            index = feature_list[i]
            xtrain[:, i] = x_train[:, index]
        x_train = xtrain

        if name == 'ZhangTest':
            xtest = np.zeros((len(x_test), k))
            for i in range(0, k):
                index = feature_list[i]
                xtest[:, i] = x_test[:, index]
            x_test = xtest
    
    print('train_d:'+str(x_train.shape))
    if name == 'ZhangTest':
        print('test_d:'+str(x_test.shape))

    best_score = -1
    train_x = x_train[:,0:k]
    train_y = y_train
    KF = KFold(n_splits = 10,random_state = 7,shuffle=True)

    if name == 'ZhangTest':
        test_x = x_test[:,0:k]
        test_y = y_test
        KF = LeaveOneOut()
    


    fr= open('./result/'+resultname+'_result/same_k_'+str(k)+'.txt', 'a')
    g = math.pow(2, pg)
    c = math.pow(2, pc)
    y_pred_list = []
    y_test_list = []
    y_score_list = []
    clf = SVC(C=c, gamma=g, kernel='rbf', probability=True, random_state=7)
    for train, test in KF.split(train_x, train_y):
        clf.fit(train_x[train], train_y[train])
        y_pred = clf.predict(train_x[test])
        y_score = clf.predict_proba(train_x[test])
        print(y_pred)
        print(y_score)
        y_pred_list.extend(y_pred)
        y_test_list.extend(train_y[test])
        y_score_list.extend(y_score)
    y_score_list = np.array(y_score_list)
    confusion = confusion_matrix(y_test_list, y_pred_list)
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    Accuracy = (TP + TN) / (TP + TN + FN + FP)
    Sensitivity = TP / (TP + FN)
    Specifity = TN / (TN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    fpr_jackknife, tpr_jackknife, thresholds = roc_curve(y_test_list, y_score_list[:, 1])
    roc_auc_jackknife = auc(fpr_jackknife, tpr_jackknife)
    print("OA:" , str(Accuracy))
    print("SVM jackknifeCV AUC：", roc_auc_jackknife)
    if name == 'ZhangTest':
        fr.writelines('trainset loo:'+'\n')
    else:
        fr.writelines('trainset 10-fold:'+'\n')
    fr.writelines('k = '+str(k)+'\n')
    fr.writelines('gamma = '+str(g)+' c = '+str(c)+'\n')
    fr.writelines('OA:'+str(Accuracy)+'\n')
    fr.writelines('SN：'+str(Sensitivity)+'\n')
    fr.writelines('SP:'+str(Specifity)+'\n')
    fr.writelines('MCC:'+str(MCC)+'\n')
    fr.writelines("AUC:"+str(roc_auc_jackknife)+'\n'+'\n')
    if name == 'ZhangTest':
        clf = SVC(C=c, gamma=g, kernel='rbf', probability=True, random_state=7)
        clf.fit(train_x,train_y)
        predicted = clf.predict(test_x)
        probass_y = clf.predict_proba(test_x)[:, 1]
        m=confusion_matrix(test_y,predicted)
        print('k = '+str(k))
        print(m)
        TN = m[0, 0]
        FP = m[0, 1]
        FN = m[1, 0]
        TP = m[1, 1]
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Sensitivity = TP / (TP + FN)
        Specifity = TN / (TN + FP)
        MCC = ((TP * TN) - (FP * FN)) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        if Accuracy > best_score:
            best_score = Accuracy
        print('OA:', Accuracy)
        print('SN：', Sensitivity)
        print('SP:', Specifity)
        print('MCC:', MCC)
        print("AUC:", roc_auc_score(test_y, probass_y))
        fr.writelines('Independent test:'+'\n')
        fr.writelines('OA:'+str(Accuracy)+'\n')
        fr.writelines('SN：'+str(Sensitivity)+'\n')
        fr.writelines('SP:'+str(Specifity)+'\n')
        fr.writelines('MCC:'+str(MCC)+'\n')
        fr.writelines("AUC:"+str(roc_auc_score(test_y, probass_y))+'\n'+'\n')
    fr.close()
    if name == 'ZhangTest':
        print('Independent test best_acc:'+str(best_score))

    print('Success!')
    
k = 305
g = -5
c = 5
main('ZhangTest',['ACC'],'ANOVA','Minmax',11,k,g,c,'')