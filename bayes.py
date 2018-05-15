import numpy as np
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWordsToVec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in vocabList:
        if word in inputSet:
            returnVec[vocabList.index(word)] = 1
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    numItem = len(trainMatrix)
    numFeat = len(trainMatrix[0])
    p1 = sum(trainCategory)/numItem
    # Can return a dict with more than one choice
    # for choice in set(trainCategory):
    #     dict[choice] = dict.get(choice,0) + 1
    p0Vect = np.ones(numFeat)
    p1Vect = np.ones(numFeat)
    p0Tot = 2.
    p1Tot = 2.
    for i in range(numItem):
        if trainCategory[i] == 1:
            p1Vect += trainMatrix[i]
            p1Tot  += sum(trainMatrix[i])
        else:  
            p0Vect += trainMatrix[i]
            p0Tot  += sum(trainMatrix[i])
    return np.log(p0Vect/p0Tot),np.log(p1Vect/p1Tot),p1

def trainNB(trainMatrix,trainCategory):
    numItem = len(trainCategory)
    numWord = len(trainMatrix[0])
    p1 = sum(trainCategory)/numItem
    p0Vect = np.ones(numWord)
    p1Vect = np.ones(numWord)
    numP1 = 2.
    numP0 = 2.
    for i in range(numItem):
        if trainCategory[i] == 1:
            p1Vect += trainMatrix[i]
            numP1 += 1
        else:
            p0Vect += trainMatrix[i]
            numP0 += 1
    return np.log(p0Vect/numP0),np.log(p1Vect/numP1),p1

def nbTest(testVec,p0Vect,p1Vect,p1Class):
    p1 = sum(testVec*p1Vect)+np.log(p1Class)
    print(p1)
    p0 = sum(testVec*p0Vect)+np.log(1.0-p1Class)
    print(p0)
    if p1 > p0:
        return 1
    else:
        return 0

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def calcMostFreq(vocabList,fullText):
    wordCount={}
    for word in fullText:
        wordCount[word] = wordCount.get(word,0) + 1
    #for token in vocabList:
    #    wordCount[token] = fullText.count(token)
    sortedFreq = sorted(wordCount.iteritems(),key=lambda x:x[1],reverse=True)
    return sortedFreq[:30]

def localTokens(feed0,feed1):
    import feedparser
    docList=[];classList=[];fullText=[]
    minLen = min(len(feed0['entries']),len(feed1['entries']))
    for i in range(minLen):
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        classList.append(0)
        fullText.extend(wordList)
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        classList.append(1)
        fullText.extend(wordList)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for freqWords in top30Words:
        vocabList.remove(freqWords[0])
    trainingSet = range(2*minLen);testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,2*minLen))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainingMat=[];testingMat=[];trainingClass=[]
    for docIndex in trainingSet:
        trainingMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainingClass.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(trainingMat,trainingMat)
    errNo = 0
    for docIndex in testSet:
          testVec = bagOfWords2VecMN(vocabList,docList[docIndex])
          result = nbTest(testVec,p0V,p1V,pSpam)
          if result != classList[docIndex]:
              errNo += 1
    return p0V,p1V,vocabList

def textParse(bigString):
    import rs 
    listOfTokens = rs.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 3]





