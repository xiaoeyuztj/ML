import math
def calcShannonEnt(dataSet):
    classSet = {}
    numEntries = float(len(dataSet))
    for datum in dataSet:
        featureVal=datum[-1] 
        classSet[featureVal] = classSet.get(featureVal,0) + 1
    ShannonEnt = 0.
    for fk,fv in classSet.items():
        Prob = fv/numEntries
        ShannonEnt -= Prob * math.log(Prob,2)
    return ShannonEnt        
def createDataSet():
    dataSet =[[1,1,'yes'],
              [1,1,'yes'],
              [1,0,'no'],
              [0,1,'no'],
              [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
    
def splitDataSet(dataSet,axis,value):
    returnDat = []
    for datum in dataSet:
        if datum[axis] == value:
            reduceDatum = datum[:axis]
            reduceDatum.extend(datum[axis+1:])
            returnDat.append(reduceDatum)
    return returnDat

def chooseBestFeatureToSplit(dataSet):
    BaseEnt = calcShannonEnt(dataSet)
    #############
    numFeature = len(dataSet[0])-1
    numEntries = len(dataSet)
    #############
    BestFeature = -1
    BestEnt = 0
    for i in range(numFeature):
        newEnt = 0.
        featValSet = set(datum[i] for datum in dataSet)
        for val in featValSet: 
            subDataSet = splitDataSet(dataSet,i,val)
            Prob = float(len(subDataSet))/numEntries
            newEnt += Prob*calcShannonEnt(subDataSet)
        if BaseEnt - newEnt > BestEnt:
            BestEnt = BaseEnt - newEnt
            BestFeature = i
        print(BaseEnt-newEnt,i)
    return BestFeature

def creatTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestLabel = labels[bestFeature]
    dTree = {bestLabel:{}}
    del(labels[bestFeature])
    uniqueVal = set([datum[bestFeature] for datum in dataSet])
    for val in uniqueVal:
        sublabels = labels[:]
        dTree[bestLabel][val] = creatTree(splitDataSet(dataSet,bestFeature,val),sublabels)
    return dTree 

def classifyByDecisionTree(inputTree,featLabels,testVec):
    judgeFeat = inputTree.keys()[0]
    judgeTree = inputTree[judgeFeat]
    judgeFeatIndex = featLabels.index(judgeFeat)
    for key in judgeTree.keys():
        if testVec[judgeFeatIndex] == key:
            if type(judgeTree[key]).__name__ == 'dict':
                classLabel = classifyByDecisionTree(judgeTree[key],featLabels,testVec)
            else: classLabel = judgeTree[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    

    

    
            


        

        
        