import numpy as np
#Create fake dataset
def CreateClustredData(N,k):
    np.random.seed(1234)
    pointPerCluster=float (N)/k
    X=[]
    y=[]
    for i in range(k):
        incomeCentroid=np.random.uniform(2000.0,200000.00)
        ageCentroid=np.random.uniform(20.0,70.0)
        for j in range(int(pointPerCluster)):
            X.append([np.random.normal(incomeCentroid,10000.0), np.random.normal(ageCentroid,2.0)])
            y.append(i)
    X=np.array(X)
    y=np.array(y)
    return X,y

import matplotlib.pyplot as plt

from pylab import *
from sklearn.preprocessing import MinMaxScaler

(x,y)=CreateClustredData(100,5)

plt.figure(figsize=(8,6))
plt.scatter(x[:,0],x[:,1],c=y.astype(np.float64))
plt.show()


from sklearn import svm,datasets
c=1.0
svc=svm.SVC(kernel="linear",C=c).fit(x,y)

def plotprediction(clf,X,Y):
    xx, yy = np.meshgrid(np.linspace(x[:,0].min(), x[:,0].max(), 500), np.linspace(x[:,1].min(), x[:,1].max(), 500))
    


#Convert to Numpy arrays
    npx=xx.ravel()
    npy=yy.ravel()


#convert to a list of 2D ( income , age ) point
    SamplePoints=np.c_[npx,npy]

#Generate predicted label for each point

    z=clf.predict(SamplePoints)

    plt.figure(figsize=(8,6))
    z=z.reshape(xx.shape)
    plt.contourf(xx,yy,z,cmap=plt.cm.Paired,alpha=0.8)
    plt.scatter(x[:,0],x[:,1],c=y.astype(np.float64))
    plt.show()

plotprediction(svc,x,y)

    