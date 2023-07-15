import csv
import math
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN 
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

def main(name,feature_extraction,feature_selection,standard,G,dimension,over_sampling):
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
        if 'ACC' in feature_extraction:
            feature=getdata('./feature_data/Feng/'+str(G)+'Fengacc',x_train)
            x_train=np.array(feature)
        if '3gram1' in feature_extraction:
            feature=getdata('./feature_data/Feng/3Fenggram1',x_train)
            x_train=np.array(feature)
            print(x_train.shape)
            print(y_train.shape)
    elif name == 'Zhang':
        x_train = []
        x_test = []
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
        if name == 'Feng':
            if over_sampling == 'SMOTE':
                smo = SMOTE(random_state=42)
                x_train, y_train = smo.fit_sample(x_train, y_train)

            elif over_sampling == 'SMOTEENN':
                smo = SMOTEENN(random_state=42)
                x_train, y_train = smo.fit_sample(x_train, y_train)
        elif name == 'Zhang':
            print("Balanced Before!")
        print('Over_sampling Successful!')
        print(x_train.shape)
        print(y_train.shape)

    if standard != '':
        if standard == 'Minmax':
            std = MinMaxScaler()
        elif standard == 'Robust':
            std = RobustScaler()
        elif standard == 'Standard':
            std = StandardScaler()
        x_train = std.fit_transform(x_train)
        if name == 'Zhang':
            x_test = std.fit_transform(x_test)
        print('standard Successful!')
		
    # resultname
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
    print('resultname:',resultname)


    if feature_selection != '':
        S1 = x_train.shape[0]
        S2 = x_train.shape[1]
        x_train_changed  = np.zeros((S1+1, S2+1))
        x_train_changed[0][0] = 0
        for i in range(1,S2+1):
            x_train_changed[0][i] = i - 1
        for i in range(1,S1+1):
            x_train_changed[i][0] = y_train[i-1]
            for j in range(1,S2+1):
                x_train_changed[i][j] = x_train[i-1][j-1]

        file = open('./feature_selection/TempFile/'+resultname+'_result/'+name+'_x_train_changed.csv', 'w', newline='') 
        content = csv.writer(file, dialect='excel')
        for i in range(len(x_train)+1):
            content.writerow(x_train_changed[i])
        file.close()

        feature_list = []
        if feature_selection == 'ANOVA':
            feature_list = run1('./feature_selection/TempFile/'+resultname+'_result/'+name+'_x_train_changed.csv')
        elif feature_selection == 'MRMD':
            path = './feature_selection/TempFile/'+resultname+'_result/'+name+'OrderList.csv'
            isExists=os.path.exists(path)
            if not isExists:
                feature_list = run2('./feature_selection/TempFile/'+resultname+'_result/'+name+'_x_train_changed.csv')
            else:
                file = open('./feature_selection/TempFile/'+resultname+'_result/'+name+'OrderList.csv', 'r') 
                orderlist = csv.reader(file)
                for i in orderlist:
                    feature_list = [int(float(x)) for x in i]
                file.close()

            for i in range(len(feature_list)):
                if feature_list[i] == '0.0.1':
                    feature_list[i] = '0.0'
            print(feature_list)  

                
        file = open('./feature_selection/TempFile/'+resultname+'_result/'+name+'OrderList.csv', 'w', newline='')  
        content = csv.writer(file, dialect='excel')  
        content.writerow(feature_list)
        file.close()

        k = dimension
        feature_list = [int(float(x)) for x in feature_list]
        xtrain = np.zeros((len(x_train), k))
        for i in range(0, k):
            index = feature_list[i]
            xtrain[:, i] = x_train[:, index]
        x_train = xtrain

        if name == 'Zhang':
            xtest = np.zeros((len(x_test), k))
            for i in range(0, k):
                index = feature_list[i]
                xtest[:, i] = x_test[:, index]
            x_test = xtest
    
    print('trainset_d:'+str(x_train.shape))

    path = './result/'+resultname+'_result/'
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    best_score_all_dimension = []
    best_score_all_dimension1 = []
    best_score_all_dimension2 = []
    train_y = y_train
    KF = KFold(n_splits = 10,random_state = 7)
    loo = LeaveOneOut()
    f1 = 5
    if name == 'Feng':
        for k in range(f1,dimension+1,5):
            train_x=x_train[:,0:k]
            
            indexi = -1
            indexj = -1
            indexk = -1
            bestscore = -1

            file = open('./result/'+resultname+'_result/'+resultname+'_result.txt', 'a', newline='')  

            for i in range(3, -15-1, -2):
                for j in range(-3, 15+1, 2):
                    g = math.pow(2, i)
                    c = math.pow(2, j)
                    clf = SVC(C=c, gamma=g, kernel='rbf', probability=True, random_state=7)
                    # score=cross_val_score(clf,train_x,train_y,cv=LeaveOneOut(),scoring='accuracy',n_jobs=-1).mean()
                    score = cross_val_score(clf,train_x,train_y,cv=KF,scoring='accuracy',n_jobs=-1).mean()
                    print(i,j,k,score)
                    
                    if score > bestscore:
                        indexi = i
                        indexj = j
                        indexk = k
                        bestscore = score
            
            print(k,bestscore)
            best_score_all_dimension.append(bestscore)
            file.writelines('std = '+str(standard)+'\n')
            file.writelines('k = '+str(k)+'\n')
            file.writelines('maxscore:'+str(bestscore)+'\n')
            file.writelines('index:'+str([indexi,indexj,indexk,bestscore])+'\n')
            file.close()


    if name == 'Zhang':
        test_y = y_test
        for k in range(f1,dimension+1,5):
            train_x = x_train[:,0:k]
            test_x = x_test[:,0:k]
            indexi = -1
            indexj = -1
            indexk = -1

            bestscore = -1
            bestscore1 = -1
            bestscore2 = -1
            file1 = open('./result/'+resultname+'_result/'+resultname+'_result_Train.txt', 'a', newline='')  
            file2 = open('./result/'+resultname+'_result/'+resultname+'_result_Test.txt', 'a', newline='')  

            for i in range(3, -15-1, -2):
                for j in range(-3, 15+1, 2):
                    g = math.pow(2, i)
                    c = math.pow(2, j)
                    clf = SVC(C=c, gamma=g, kernel='rbf', probability=True, random_state=7)
                    # train loo
                    score1 = cross_val_score(clf,train_x,train_y,cv=loo,scoring='accuracy',n_jobs=-1).mean()
                    # test
                    clf.fit(train_x,train_y)
                    predicted = clf.predict(test_x)
                    score2 = accuracy_score(test_y,predicted)
                    score = (score1 + score2)/2
                    print(i,j,k,score1,score2,score)
                    
                    if score > bestscore:
                        indexi = i
                        indexj = j
                        indexk = k
                        bestscore = score
                        bestscore1 = score1
                        bestscore2 = score2
            
            print(k,bestscore1,bestscore2,bestscore)
            best_score_all_dimension.append(bestscore)
            best_score_all_dimension1.append(bestscore1)
            best_score_all_dimension2.append(bestscore2)
            file1.writelines('std = '+str(standard)+'\n')
            file1.writelines('k = '+str(k)+'\n')
            file1.writelines('maxscore:'+str(bestscore1)+'\n')
            file1.writelines('index:'+str([indexi,indexj,indexk,bestscore1])+'\n')
            file1.close()
            file2.writelines('std = '+str(standard)+'\n')
            file2.writelines('k = '+str(k)+'\n')
            file2.writelines('maxscore:'+str(bestscore2)+'\n')
            file2.writelines('index:'+str([indexi,indexj,indexk,bestscore2])+'\n')
            file2.close()
    if name == 'Feng':
        print('Feng_best_score_all:'+str(max(best_score_all_dimension)))
    if name == 'Zhang':
        index = best_score_all_dimension.index(max(best_score_all_dimension))
        print('ZhangTrain_best_score_all:'+str(best_score_all_dimension1[index]))
        print('ZhangTest_best_score_all:'+str(best_score_all_dimension2[index]))
        print('Average_best_score_all:'+str(max(best_score_all_dimension)))



main('Zhang',['ACC'],'ANOVA','Minmax',11,350,'')
main('Feng',['ACC'],'ANOVA','Minmax',11,350,'')
