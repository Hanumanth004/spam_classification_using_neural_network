import numpy as np
from random import seed
from scipy.special import expit
from random import randint
import sys
import csv
from sklearn.metrics import confusion_matrix

seed(1)
hidden_size=40
hidden_size1=30
classes=2

BATCH_SIZE=10
input_size=57

Wh=np.random.randn(hidden_size,input_size)*0.01
bh=np.random.randn(hidden_size,1)*0.01
Wh1=np.random.randn(hidden_size1,hidden_size)*0.01
bh1=np.random.randn(hidden_size1,1)*0.01
Wo=np.random.randn(classes,hidden_size1)*0.01
bo=np.random.randn(classes,1)*0.01

learning_rate = 0.05

def confusion_matrix_t(y_true, y_pred):
    print "confusion matrix:"
    print(confusion_matrix(y_true, y_pred))

def inferenceFunc(X_data):
    correctly_classified=0
    predict=[]
    actual=[row[-1] for row in X_data]
    for i in xrange(len(X_data)):
        X_tmp=X_data[i]
        label=X_tmp[-1]
        X=X_tmp[0:X_tmp.shape[0]-1]
        X=np.reshape(X, ((X_tmp.shape[0]-1),1))
        #forward propogation!
        h1=np.dot(Wh,X) + bh
        hg1=expit(h1)
        h2=np.dot(Wh1,hg1) + bh1
        hg2=expit(h2)
        h3=np.dot(Wo,hg2) + bo
        hg3=expit(h3)
        y=hg3
        predicted=np.argmax(y)
        predict.append(predicted) 
        if predicted == label:
            correctly_classified+=1
            
    print "Training Accuracy:%f" %(correctly_classified/float(len(X_data))*100.0)
    confusion_matrix_t(actual, predict)
    

def str_column_to_float(X_data, column):
    for row in X_data:
        row[column]=float(row[column].strip())

mWh, mWh1, mWo = np.zeros_like(Wh), np.zeros_like(Wh1), np.zeros_like(Wo)
mbh, mbh1, mbo  = np.zeros_like(bh), np.zeros_like(bh1), np.zeros_like(bo)

def lossFunction(X_tr):
    global Wh
    global bh
    global Wh1
    global bh1
    global Wo
    global bo

    loss=0.0
    dbo=np.zeros_like(bo)
    dbh=np.zeros_like(bh)
    dWh=np.zeros_like(Wh)
    dWo=np.zeros_like(Wo)
    dbh1=np.zeros_like(bh1)
    dWh1=np.zeros_like(Wh1)

    for i in xrange(len(X_tr)):
        X_temp=X_tr[i]
        X=X_temp[0:X_temp.shape[0]-1]
        label=X_temp[-1]
        if int(label)==0:
            T=np.array([1,0]).reshape(2,1)
        else:
            T=np.array([0,1]).reshape(2,1)
        
        X=np.reshape(X, ((X_temp.shape[0]-1),1))

        #forward propogation! 
        h1=np.dot(Wh,X) + bh
        hg1=expit(h1)
        h2=np.dot(Wh1,hg1) + bh1
        hg2=expit(h2)
        h3=np.dot(Wo,hg2) + bo
        hg3=expit(h3)
        y=hg3
        loss+=np.sum(0.5*(T-y)*(T-y))
        
        #backward propogation
        de=-(T-y)
        dhg3=hg3*(1-hg3)

        dy=dhg3*de
        dbo+=dy
        dWo+=np.dot(dy,hg2.T)
        dh12=np.dot(Wo.T, dy)

        dh12tmp=hg2*(1-hg2)*dh12
        dbh1+=dh12tmp
        dWh1+=np.dot(dh12tmp,hg1.T)
        dh11=np.dot(Wh1.T, dh12tmp)

        dbh11tmp=hg1*(1-hg1)*dh11
        dbh+=dbh11tmp
        dWh+=np.dot(dbh11tmp,X.T)

        np.clip(dWh,-5,5,dWh)
        np.clip(dWh1,-5,5,dWh1)
        np.clip(dWo,-5,5,dWo)
        np.clip(dbh,-5,5,dbh)
        np.clip(dbh1,-5,5,dbh1)
        np.clip(dbo,-5,5,dbo)

        if(i%BATCH_SIZE==0):
            #Uncomment this section for vanilla parameter update
            """ 
            #update weights
            Wh+=-learning_rate*dWh
            Wh1+=-learning_rate*dWh1
            Wo+=-learning_rate*dWo
            bh+=-learning_rate*dbh
            bh1+=-learning_rate*dbh1
            bo+=-learning_rate*dbo
            """
            #Adagrad parameter update
            for param, dparam, mem in zip([Wh, Wh1, Wo, bh, bh1, bo],
                                          [dWh,dWh1,dWo,dbh,dbh1,dbo],
                                          [mWh,mWh1,mWo,mbh,mbh1,mbo]):
                mem+=dparam*dparam
                param+=-learning_rate*dparam/np.sqrt((mem)+1e-8)


            dbo=np.zeros_like(bo)
            dbh=np.zeros_like(bh)
            dWh=np.zeros_like(Wh)
            dWh1=np.zeros_like(Wh1)
            dbh1=np.zeros_like(bh1)
            dWo=np.zeros_like(Wo)
    return loss


#I have split the training, testing and validation set and put into
#different files (please look at the split_dataset.py). Below functions
#read the files supplied in the command line arguments!
sample=[]
with open(sys.argv[1], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)


X_tr=sample
X_tr=np.array(X_tr)
X_tr=X_tr.astype(np.float)


sample=[]
with open(sys.argv[2], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)

X_val=sample
X_val=np.array(X_val)
X_val=X_val.astype(np.float)

sample=[]
with open(sys.argv[3], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)

X_te=sample
X_te=np.array(X_te)
X_te=X_te.astype(np.float)

"""
label1=[]
for i in X_tr:
    label1.append(i[-1])
print label1.count(0)
print label1.count(1)

label=[]
for i in X_val:
    label.append(i[-1])

print label.count(0)
print label.count(1)
"""

#Please look at the weight_matrix folder for the model!
choice=raw_input("Enter yes to train the network!")
if choice=="yes": 
    for ep in xrange(500):
        loss=lossFunction(X_tr)
        np.random.shuffle(X_tr)
        if(ep%50==0):
            print "epoch number:%d" %(ep)
            #print "%d %f" %(ep,loss)
            correctly_classified=0
            for i in xrange(len(X_tr)):
                X_tmp=X_tr[i]
                label=X_tmp[-1]
                X=X_tmp[0:X_tmp.shape[0]-1]
                X=np.reshape(X, ((X_tmp.shape[0]-1),1))
                #forward propogation!
                h1=np.dot(Wh,X) + bh
                hg1=expit(h1)
                h2=np.dot(Wh1,hg1) + bh1
                hg2=expit(h2)
                h3=np.dot(Wo,hg2) + bo
                hg3=expit(h3)
                y=hg3
                predicted=np.argmax(y)
                if predicted == label:
                    correctly_classified+=1
                    
            print "Training Accuracy:%f" %(correctly_classified/float(len(X_tr))*100.0)
            
            correctly_classified=0
            for i in xrange(len(X_val)):
                X_tmp=X_val[i]
                label=X_tmp[-1]
                X=X_tmp[0:X_tmp.shape[0]-1]
                X=np.reshape(X, ((X_tmp.shape[0]-1),1))
                #forward propogation!
                h1=np.dot(Wh,X) + bh
                hg1=expit(h1)
                h2=np.dot(Wh1,hg1) + bh1
                hg2=expit(h2)
                h3=np.dot(Wo,hg2) + bo
                hg3=expit(h3)
                y=hg3
                predicted=np.argmax(y)
                if predicted == label:
                    correctly_classified+=1 
            print "Validation Accuracy:%f" %(correctly_classified/float(len(X_val))*100.0)
    """
    print "Accuracy on testing and validation set"
    print "\n"
    print "Training set"
    inferenceFunc(X_tr)
    print "\n"
    print "Validation set"
    inferenceFunc(X_val)
    print "\n"
    print "Testing set"
    inferenceFunc(X_te)
    """

    option=raw_input("Enter yes to save the model!")
    if option=="yes":
        np.save("./weight_matrix/Wh", Wh) 
        np.save("./weight_matrix/bh", bh) 
        np.save("./weight_matrix/Wh1", Wh1) 
        np.save("./weight_matrix/bh1", bh1) 
        np.save("./weight_matrix/Wo", Wo) 
        np.save("./weight_matrix/bo", bo) 
else:
    print "Using the existing model"
    Wh=np.load("./weight_matrix/Wh.npy")
    bh=np.load("./weight_matrix/bh.npy")
    Wh1=np.load("./weight_matrix/Wh1.npy")
    bh1=np.load("./weight_matrix/bh1.npy")
    Wo=np.load("./weight_matrix/Wo.npy")
    bo=np.load("./weight_matrix/bo.npy")


#check the accuracy of the validation and testing set after the training is done

print "Accuracy on testing and validation set"
print "\n"
print "Training set"
inferenceFunc(X_tr)
print "\n"
print "Validation set"
inferenceFunc(X_val)
print "\n"
print "Testing set"
inferenceFunc(X_te)
