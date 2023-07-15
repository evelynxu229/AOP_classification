import csv
import math
import os
# import six
import sys
# sys.modules['sklearn.externals.six'] = six
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
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
from feature_selection.method.ANOVA import run1
from feature_selection.method.MRMD import run2
import random


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


    if over_sampling != '':
        if over_sampling == 'SMOTE':
            smo = SMOTE(sampling_strategy=1.0,random_state=42)
            x_train, y_train = smo.fit_sample(x_train, y_train)

        elif over_sampling == 'SMOTEENN':
            print("Don't use SMOTEENN!")
            return 0
        print('Over_sampling Successful!')
        print(x_train.shape)
        print(y_train.shape)

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



    if feature_selection != '':
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
    

    print('train_d:'+str(x_train.shape))

    x_train_zip = enumerate(x_train)
    x_train_zip = [[k,v] for k,v in x_train_zip]
    dataset = []
    for i in range(len(x_train_zip)):
        dataset.append([x_train_zip[i],y_train[i]])

    r=random.random
    random.seed(7)
    random.shuffle(dataset,random=r)
    dataset = np.array(dataset)

    x_train_zip = dataset[:,0]
    y_train2 = dataset[:,1]
    y_train2 = y_train2.astype('int')
    shuffle_index = [index for index,value in x_train_zip]
    x_value = [value for index,value in x_train_zip]
    x_value = np.array(x_value)

    train_x = x_train[:,0:k]
    train_y = y_train

    KF = LeaveOneOut()
    # KF = KFold(n_splits = 10)
    if pg == 0 and pc == 0:
        for i in range(3, -15-1, -2):
            for j in range(-3, 15+1, 2):
                fr= open('./result/'+resultname+'_result/same_k_'+str(k)+'.txt', 'a')
                g = math.pow(2, i)
                c = math.pow(2, j)
                clf = SVC(C=c, gamma=g, kernel='rbf', probability=True, random_state=7)

                n = 0
                result = []
                y_true = []
                y_score_list = []
                for train_index,test_index in KF.split(train_x):
                    n += 1
                    x_train_tmp = train_x[train_index]
                    y_train_tmp = train_y[train_index]
                    x_test_tmp = train_x[test_index]
                    y_test_tmp = train_y[test_index]
                    # print(test_index)
                    print(str(n) + ": " + str(np.array(x_train_tmp).shape) + " "+str(np.array(x_test_tmp).shape))
                    clf.fit(x_train_tmp,y_train_tmp)
                    res = clf.predict(x_test_tmp)
                    y_score = clf.predict_proba(x_test_tmp)
                    # print(np.array(y_score).shape)
                    result.extend(res)
                    y_true.extend(y_test_tmp)
                    y_score_list.extend(y_score)
                    # print(np.array(y_score_list).shape)
                # with open('ob3.txt','w') as f:
                #     for i in range(len(y_train)):
                #         f.writelines(str(shuffle_index[i])+" "+str(train_y[i])+" "+str(shuffle_index[i] >= 1552)+" "+str(y_true[i])+"\n")
                
                y_pred = []
                y_true2 = []
                y_score2 = []
                for n in range(len(shuffle_index)):
                    if shuffle_index[n] < 1805:
                        y_pred.append(result[n])
                        y_true2.append(y_true[n])
                        y_score2.append(y_score_list[n])
                y_score2 = np.array(y_score2)

                confusion = confusion_matrix(y_true2, y_pred)
                TP = confusion[1, 1]
                TN = confusion[0, 0]
                FP = confusion[0, 1]
                FN = confusion[1, 0]
                Accuracy = (TP + TN) / (TP + TN + FN + FP)
                Sensitivity = TP / (TP + FN)
                Specifity = TN / (TN + FP)
                MCC = ((TP * TN) - (FP * FN)) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
                fpr_jackknife, tpr_jackknife, _ = roc_curve(y_true2, y_score[:, 1])
                roc_auc_jackknife = auc(fpr_jackknife, tpr_jackknife)
                print(i,j,k,Accuracy)
                print("SVM jackknifeCV AUC：", roc_auc_jackknife)
                fr.writelines('trainset loo:'+'\n')
                fr.writelines('k = '+str(k)+'\n')
                fr.writelines('gamma = '+str(g)+' c = '+str(c)+'\n')
                fr.writelines('OA:'+str(Accuracy)+'\n')
                fr.writelines('SN：'+str(Sensitivity)+'\n')
                fr.writelines('SP:'+str(Specifity)+'\n')
                fr.writelines('MCC:'+str(MCC)+'\n')
                fr.writelines("AUC:"+str(roc_auc_jackknife)+'\n'+'\n')

    else:
        fr= open('./result/'+resultname+'_result/same_k_'+str(k)+'.txt', 'a')
        g = math.pow(2, pg)
        c = math.pow(2, pc)
        clf = SVC(C=c, gamma=g, kernel='rbf', probability=True, random_state=7)

        n = 0
        result = []
        y_true = []
        y_score_list = []
        for train_index,test_index in KF.split(train_x):
            n += 1
            x_train_tmp = train_x[train_index]
            y_train_tmp = train_y[train_index]
            x_test_tmp = train_x[test_index]
            y_test_tmp = train_y[test_index]
            # print(test_index)
            print(str(n) + ": " + str(np.array(x_train_tmp).shape) + " "+str(np.array(x_test_tmp).shape))
            clf.fit(x_train_tmp,y_train_tmp)
            res = clf.predict(x_test_tmp)
            y_score = clf.predict_proba(x_test_tmp)
            result.extend(res)
            y_true.extend(y_test_tmp)
            y_score_list.extend(y_score)
        # with open('ob3.txt','w') as f:
        #     for i in range(len(y_train)):
        #         f.writelines(str(shuffle_index[i])+" "+str(train_y[i])+" "+str(shuffle_index[i] >= 1552)+" "+str(y_true[i])+"\n")
        
        y_pred = []
        y_true2 = []
        y_score2 = []
        for n in range(len(shuffle_index)):
            if shuffle_index[n] < 1805:
                y_pred.append(result[n])
                y_true2.append(y_true[n])
                # print(n)
                y_score2.append(y_score_list[n])
        y_score2 = np.array(y_score2)
        y_score2 = np.squeeze(y_score2)

        confusion = confusion_matrix(y_true2, y_pred)  
        TP = confusion[1, 1]
        TN = confusion[0, 0]
        FP = confusion[0, 1]
        FN = confusion[1, 0]
        Accuracy = (TP + TN) / (TP + TN + FN + FP)
        Sensitivity = TP / (TP + FN)
        Specifity = TN / (TN + FP)
        MCC = ((TP * TN) - (FP * FN)) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        print(np.array(y_true2).shape)
        print(np.array(y_score2).shape)
        fpr_jackknife, tpr_jackknife, thresholds = roc_curve(y_true2, y_score2[:, 1])
        roc_auc_jackknife = auc(fpr_jackknife, tpr_jackknife)
        print(pg,pc,k,Accuracy)
        print("SVM jackknifeCV AUC：", roc_auc_jackknife)
        # fr.writelines('trainset loo:'+'\n')
        fr.writelines('trainset 10-fold:'+'\n')
        fr.writelines('k = '+str(k)+'\n')
        fr.writelines('gamma = '+str(g)+' c = '+str(c)+'\n')
        fr.writelines('OA:'+str(Accuracy)+'\n')
        fr.writelines('SN：'+str(Sensitivity)+'\n')
        fr.writelines('SP:'+str(Specifity)+'\n')
        fr.writelines('MCC:'+str(MCC)+'\n')
        fr.writelines("AUC:"+str(roc_auc_jackknife)+'\n'+'\n')
    print('Success!')


k = 305
g = 3
c = 5

main('Feng',['ACC'],'ANOVA','Minmax',11,k,g,c,'SMOTE')
