import csv

import time
import numpy as np
from numpy import linalg as LA


nusers = 943 + 1
nitems = 1682 + 1


maxi_err = 0
mini_err = 10

#Nuclear norm Minimization
def nuclear_minimization_norm(Atrain, masking, lamda): #y, m, lamda
    x = np.random.randint(5, size=(Atrain.shape))
    print("nuclear_minimization_norm for lamda = ", lamda)
    for ep in range(10):
        tmp = Atrain - np.multiply(masking, x)
        tmp = x + tmp
        fn1,fn2, fn3 = np.linalg.svd(tmp, full_matrices=False)

        y = sftmax(np.diag(fn2), lamda/2.0)


        x = np.dot(fn1, np.dot(y, fn3))
    return x.round()


# Calculation as a part of Nuclear norm Minimization.
def sftmax(q, d):
    q = np.absolute(q)
    cp = q.copy()
    q -= d
    dim = (q.shape)
    for i in range(dim[0]):
        for j in range(dim[1]):
            if q[i][j] < 0:
                cp[i][j] = -1
            elif q[i][j] > 0:
                cp[i][j] = 1
            else:
                cp[i][j] = 0
            q[i][j] = abs(q[i][j])

    for i in range(dim[0]):
        for j in range(dim[1]):
            q[i][j] -= d
            if q[i][j] < 0:
                q[i][j] = 0
    re = np.multiply(cp, q)
    return re



#Model for Training and Testing and collecting err
def model_create(testdata, traindata, masking, lamda):
    fil_mat = nuclear_minimization_norm(traindata, masking, lamda)
    err = 0.0
    global maxi_err, mini_err
    mini_err = 100
    maxi_err = 0
    for ndata in testdata:
        tmp = (ndata[2]-fil_mat[ndata[0]][ndata[1]])
        tmp = abs(tmp)
        if tmp < mini_err:
            mini_err = tmp
        if tmp > maxi_err:
            maxi_err = tmp
        err += tmp
    err /= len(testdata)
    return err


#For Loading the dataset
def Load_data():
    fold4 = 'u1.base'
    fold5th = 'u1.test'

    traindata = np.zeros((nusers,nitems))
    masking = np.zeros((nusers,nitems))
    testdata = []

    with open(fold5th, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for datarow in reader:
            testdata.append([int(datarow[0]), int(datarow[1]), float(datarow[2])])

    with open(fold4, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for datarow in reader:
            u = int(datarow[0])
            i = int(datarow[1])
            rt = float(datarow[2])
            traindata[u][i] = rt
            masking[u][i] = 1

    return testdata, traindata, masking

#For 4foldtrain and 1 fold test validation in the Load_data
def five_fold_calc():

    lamdas = [0.1, 0.4, 0.6, 0.8, 1.0]
    for lamda in lamdas:
        print("lamda = ",lamda)
        err = 0.0

        testdata,traindata, masking = Load_data()
        tmp = model_create(testdata,traindata, masking, lamda)

        tmp /= (maxi_err - mini_err)
        err += tmp
    


        print("For lamda = ",lamda,"MAE = ", err)
        print("\n");
    return err


five_fold_calc()
