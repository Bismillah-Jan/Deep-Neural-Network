# -*- coding: utf-8 -*-
"""
Deep Neural Network
Assignment 2A
@author: Bismillah Jan
@author-email: bismillahjan222@gmail.com
"""
import numpy as np
import scipy
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd



class linearClassifier:
    """
    Base class for linear classifier
    """
    def __init__(self,**kwargs):
        self.W = np.array([0])
        self.bias = 0
        
    def train(self,**kwargs):
        pass
    
    def score(self,x):
        """
        Return the discriminant score
        Input: x is m x d numpy array of m examples
        returns: m dimensional numpy vector of scores
        """
        return np.dot(x,self.W) + self.bias
        
    def classify(self,x):
        """
        Return the label score
        Input: x is m x d numpy array of m examples
        returns: m dimensional numpy vector of labels
        """
        return 2*(self.score(x)>=0) - 1
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
        
    def __str__(self):
        s = []
        for i,w in enumerate(self.W):
            si = "(%0.2f) x%i" % (w,i+1)
            s.append(si)
        si = "(%0.2f)" % (self.bias)
        s.append(si)
        s = ' + '.join(s) 
        s += " = 0"
        return s
            
class Perceptron(linearClassifier):        
    """
    Perceptron Implementation
    """
    def train(self,Xtr,Ytr,**kwargs):
       #the training function
        if "alpha" in kwargs:
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 1.0
        if "epochs" in kwargs:
            self.epochs = kwargs["epochs"]
        else:
            self.epochs = 100
        if "theta" in kwargs:
            self.theta = kwargs["theta"]
        else:
            self.theta = 0.0
            
        d = Xtr.shape[1]
        self.W = np.zeros(d)
        ###### put your code here ######
        #step 01
       
        for j in range(0, self.epochs):
            for i in range(Xtr.shape[0]):
                y_in=self.score(Xtr[i])
                
                if y_in>self.theta:
                    y=4.0
                elif y_in< -self.theta:
                    y=-2.0
                else:
                    y=0    
                if y!=Ytr[i]:
                    oldW=self.W
                    oldB=self.bias
                    self.W= self.W+ (self.alpha*Ytr[i]*Xtr[i])
                    self.bias=self.bias+ (self.alpha*Ytr[i])
                    
                if (self.W==oldW).all() and self.bias==oldB:
                     print ("epoch: ",j, "weight: ", self.W,oldW, self.bias)
                     break
        
            
    def classify(self,x):
        """
        Perceptron Classification Function
        """
        z = self.score(x)
        y = np.zeros(x.shape[0])
        for i,zi in enumerate(z):
            if zi < -self.theta:
                y[i] = -2.0
            elif zi > self.theta:
                y[i] = 4.0
            else:
                y[i] = 0         
        return y
                

if __name__ == '__main__':
    filename = "breast-cancer-wisconsin.data"
    inFile=open(filename, 'r')
    data=inFile.read()

    data=pd.read_csv(filename)
    data = scipy.array(data)
    X = data[:, 1:10]
    y = data[:, 10] #extract
    y[y==2]=-2
    xTrain, xTest, yTrain, yTest=train_test_split(X, y, test_size=0.2)
    P = Perceptron()
    P.train(xTrain,yTrain,alpha = 1, theta = 0.0, epochs = 1000)
    yPred = P.classify(xTest)
    print ("Predicted Labels:",yPred)
    print ("True Labels:",yTest)
    print ("Error:", np.sum(yPred!=yTest)) 
    Acc=metrics.accuracy_score(yTest, yPred)
    print ("Accuracy: ", Acc*100, "%")
    print ("Classifier function:",P)
    
    