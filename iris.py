from sklearn.datasets import load_iris

iris=load_iris()
numsample, numfeature = iris.data.shape 
print(numsample)
print(numfeature)
print(list(iris.target_names))


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(iris.data,iris.target,test_size=0.2, random_state=0)


import xgboost as xgb

train=xgb.DMatrix(x_train,y_train)
test=xgb.DMatrix(x_test,y_test)


param={
    "max_depth":4,
    "eta":0.4,
    "objective":"multi:softmax",
    "num_class":3
}
epochs=10

model=xgb.train(param,train,epochs)

predict=model.predict(test)
print(predict)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predict))

