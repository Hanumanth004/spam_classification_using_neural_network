import numpy as np
from random import randint
import sys
import csv
from sklearn.decomposition import PCA


dimension_reduction=1
nf=20

def str_column_to_float(X_data, column):
    for row in X_data:
        row[column]=float(row[column].strip())

sample=[]
with open(sys.argv[1], 'rb') as f:
    reader=csv.reader(f.read().splitlines())
    for row in reader:
        sample.append(row)

for i in range(len(sample[0])):
    str_column_to_float(sample, i)

X_data=sample

print X_data

"""
label1=[]
for i in X_data:
    label1.append(i[-1])
print label1.count(0)
print label1.count(1)

"""
np.random.shuffle(X_data)

if dimension_reduction==1:
    label=[]
    dataset_tmp=[]
    temp2=[]
    for row in X_data:
        label.append(row[-1])
        dataset_tmp.append(row[:-1])
    
    #Apply dimensionality reduction
    pca = PCA(n_components=nf)
    pca.fit(dataset_tmp)
    dataset_new=pca.transform(dataset_tmp)
    temp1 = dataset_new.tolist()

    for i, row in zip(xrange(len(label)), temp1):
        temp2.append(np.append(row,label[i]))

    temp2=(np.array(temp2)).tolist()
    X_data=temp2

    with open('train_w_pca.csv', 'w') as f:
        wr=csv.writer(f)
        wr.writerows(X_data[0:3680])
    
    with open('validation_w_pca.csv', 'w') as f:
        wr=csv.writer(f)
        wr.writerows(X_data[3681:4140])

    with open('test_w_pca.csv', 'w') as f:
        wr=csv.writer(f)
        wr.writerows(X_data[4141:])
else:
    
    with open('train.csv', 'w') as f:
        wr=csv.writer(f)
        wr.writerows(X_data[0:3680])
    
    with open('validation.csv', 'w') as f:
        wr=csv.writer(f)
        wr.writerows(X_data[3681:4140])

    with open('test_w_pca.csv', 'w') as f:
        wr=csv.writer(f)
        wr.writerows(X_data[4141:])

"""
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


label1=[]
for i in X_tr:
    label1.append(i[-1])
print label1.count(0)
print label1.count(1)
"""
