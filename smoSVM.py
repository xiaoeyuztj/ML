from numpy import *
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels)
    iter = 0;b=0
    m,n = shape(dataMatrix)
    while(iter < maxIter):
        alphas = mat(zeros(m,1))
        alphaPairsChanged = 0
        for i in range(m):
            fXi = (dataMatrix[i,:]*dataMatrix.T)*multiply(alphas,labelMat)+b## different from the book
            Ei = fXi - float(labelMat[i])
            testVal = Ei * labelMat[i]
            if (testVal > toler) and (alphas[i] > 0) or (testVal < -toler) and (alphas[i] < C):
                j = selectJrand(i,m)
                alphaJold = alphas[j].copy()
                alphaIold = alphas[i].copy()
                fXj = (dataMatrix[j,:]*dataMatrix.T)*multiply(alphas,labelMat)+b
                Ej = fXj - float(labelMat[j]) 
                if labelMat[i] != labelMat[j]:
                    L = max(0,alphaJold-alphaIold)
                    H = min(C,C+alphaJold-alphaIold)
                else:
                    L = max(0,-C+alphaJold+alphaIold) 
                    H = min(C,alphaJold+alphaIold)
                KII = float(dataMatrix[i,:]*dataMatrix[i,:].T)
                KJJ = float(dataMatrix[j,:]*dataMatrix[j,:].T)
                KIJ = float(dataMatrix[i,:]*dataMatrix[j,:].T)
                eta = -2*KIJ + KII + KJJ      
                alphas[j] = clipAlpha(alphaJold + labelMat[j]*(Ej-Ei)/eta,H,L)
                if(abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough");continue
                if(eta>=0):print("eta>=0");continue
                alphas[i] = clipAlpha(alphaIold + labelMat[i]*labelMat[j]*(alphas[j]-alphaJold),H,L)
                b1New = -Ei - labelMat[i]*KII*(alphas[i] - alphaIold) - labelMat[j] * KIJ *(alphas[j] - alphaJold) + b
                b2New = -Ej - labelMat[i]*KIJ*(alphas[i] - alphaIold) - labelMat[j] * KJJ *(alphas[j] - alphaJold) + b
                if(0<alphas[i]) and (alphas[i]<C): b = b1New
                elif(0<alphas[j]) and (alphas[j]<C): b = b2New
                else: b = (b1New + b2New) / 2.0
                alphaPairsChanged  += 1
                print("iter: %d,i,%d, pairs changed %d",iter,i,alphaPairsChanged)
        if(alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d",iter)   
    return alphas,b

def selectJrand(i,m):
    j=i
    while j==i:
        j=int(random.uniform(0.0,m))
    return j

def clipAlpha(aj,H,L):
    if(aj>H):
        aj = H
    if(aj<L):
        aj = L
    return aj
    