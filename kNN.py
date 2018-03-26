import numpy as np
import random
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k) :
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=lambda x: x[1],reverse=True) 
    return sortedClassCount[0][0]

def dataGen(dimensionOfData,Quantity,valueRange):
    #First parameter is data's dimension,the second is the number of data generated.
    #The third is parameters'value range like ((1,2),(3,4))
    randomDat = np.zeros((Quantity,dimensionOfData))
    for i in range(Quantity):
        for j in range(dimensionOfData):
            randomDat[i][j]=random.uniform(valueRange[j][0],valueRange[j][1])
    return randomDat

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    lineNum = len(arrayOLines)
    returnMat = np.zeros((lineNum,3))
    index = 0
    classLableVector = []
    for line in arrayOLines:
        line = line.strip()
        listFromLine= line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLableVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLableVector

def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    m = dataSet.shape[0]
    normDataSet = dataSet-np.tile(minVal,(m,1))
    normDataSet = normDataSet/(np.tile(ranges,(m,1)))
    return normDataSet,ranges,minVal

def datingClassTest():
    horatio = 0.2
    datingDataMat,datingLables = file2matrix(r'E:\DataScience\machinelearninginaction-master\Ch02\datingTestSet.txt')
    normDat,ranges,minval = autoNorm(datingDataMat)
    m = int(datingDataMat.shape[0])
    testNum = int(m*horatio)
    testResult = []
    errno = 0
    for i in range(testNum):
        Retmp = classify0(normDat[i,:],normDat[testNum:m,:],datingLables[testNum:m],3)
        print("initial %s and test %s" %(datingLables[i],Retmp))
        if datingLables[i] != Retmp:
            errno += 1
        testResult.append(Retmp)
    return 1.0*errno/testNum
        






if __name__=='__main__':
    testDat = dataGen(2,5,((1,2),(3,4)))
    print(testDat)    