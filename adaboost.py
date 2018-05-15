import numpy
def stumpClassify(dataMatrix,dimen,thresVal,threshIneq):
    retMat = numpy.ones(((numpy.shape(dataMatrix))[0],1)) 
    if threshIneq == 'lt':
        retMat[dataMatrix[:,dimen] <= thresVal] = -1
    else:
        retMat[dataMatrix[:,dimen] >= thresVal] = -1
    return retMat

def buildStump(dataArr,classLabels,D):
    dataMat = numpy.mat(dataArr)
    labelMat = numpy.mat(classLabels).T
    step = 10.0
    m,n = numpy.shape(dataMat)
    minErr = numpy.inf
    bestStump={}
    bestClassEst = numpy.mat(numpy.ones((m,1)))
    for i in range(n):
        minVal = min(dataMat[:,i])
        maxVal = max(dataMat[:,i])
        stepSize = float((maxVal- minVal)/step)
        for j in range(-1,int(step)+1):
            for inequal in ['lt','gt']:
                predictMat = stumpClassify(dataMat,i,(minVal+j*stepSize),inequal)
                errArr = numpy.mat(numpy.ones((m,1)))
                errArr[predictMat  == labelMat] = 0
                weightedErr = D.T * errArr
                if weightedErr < minErr:
                    bestStump['dimen'] = i
                    bestStump['thresh'] = (minVal+j*stepSize)
                    bestStump['ineq'] = inequal
                    minErr = weightedErr
                    bestClassEst = predictMat
    return bestClassEst,minErr,bestStump

def loadSimpData(): 
    dataMat = [[1.,2.1],[2,1.1],[1.3,1.],[1.,1.],[2.,1.]]
    classLabels = [1.,1.,-1.,-1.,1.]
    return dataMat,classLabels


    

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr=[]
    dataMat = numpy.mat(dataArr)
    labelMat = numpy.mat(classLabels).T
    m,n = numpy.shape(dataMat)
    D = (numpy.ones((m,1))/m)
    iter = 0
    aggClassEst = numpy.mat(numpy.zeros((m,1)))
    while iter <= numIt:
         bestClassEst,error,bestStump = buildStump(dataArr,classLabels,D)
         alpha = float(0.5*numpy.log((1-error)/max(error,1e-16)))
         bestStump['alpha'] = alpha
         weakClassArr.append(bestStump)
         mulExpMat = numpy.mat(numpy.ones((m,1)))
         mulExpMat[bestClassEst == labelMat] = -1
         expon = numpy.exp(numpy.multiply(mulExpMat,alpha))
         D = numpy.multiply(D,expon)
         D = D/sum(D)
         aggClassEst += alpha*bestClassEst
         aggErrors = numpy.multiply(numpy.sign(aggClassEst)!=labelMat,numpy.mat(numpy.ones((m,1))))
         errRate = aggErrors.sum()/m
         print(errRate)
         if errRate == 0: break
         iter += 1
    return weakClassArr



def adbsApp(appData,classArr):
    appData = numpy.mat(appData)
    m = (numpy.shape(appData))[0]
    aggErr = numpy.mat(numpy.zeros((m,1)))
    for i in range(len(classArr)):
        aggErr += classArr[i]['alpha']*stumpClassify(appData,classArr[i]['dimen'],classArr[i]['thresh'],classArr[i]['ineq'])
    retMat = numpy.sign(aggErr)
    return retMat

if __name__=="__main__":
    M,N = loadSimpData()
    #D=numpy.mat(numpy.ones((5,1))/5)
    #print(buildStump(M,N,D))
    print(adbsApp([0,0],adaBoostTrainDS(M,N,9)))