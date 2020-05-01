import numpy as np
import operator
import copy
import Models
import Task
import torch
import torch.nn as nn
from itertools import product, permutations, combinations
from functools import partial
from collections import Counter
from random import randint

def identityM(matrix):
    """
    Function that, given Matrix, returns its corresponding numpy.ndarray m
    """
    if isinstance(matrix, np.ndarray):
        return matrix.copy()
    else:
        return matrix.m.copy()

def correctFixedColors(inMatrix, x, fixedColors):
    """
    Given an input matrix (inMatrix), an output matrix (x) and a set of colors
    that should not change between the input and the output (fixedColors),
    this function returns a copy of x, but correcting the pixels that 
    shouldn't have changed back into the original, unchanged color.
    
    inMatrix and x are required to have the same shape.
    """
    m = x.copy()
    for i,j in np.ndindex(m.shape):
        if inMatrix[i,j] in fixedColors:
            m[i,j] = inMatrix[i,j]
    return m
                
def incorrectPixels(m1, m2):
    """
    Returns the number of incorrect pixels (0 is best)
    """
    if m1.shape != m2.shape:
        return 1000
    return np.sum(m1!=m2)

def deBackgroundizeMatrix(m, color):
    """
    Given a matrix m and a color, this function returns a matrix whose elements
    are 0 or 1, depending on whether the corresponding pixel is of the given
    color or not.
    """
    return np.uint8(m == color)

def relDicts(colors):
    """
    Given a list of colors (numbers from 0 to 9, no repetitions allowed), this
    function returns two dictionaries giving the relationships between the
    color and its index in the list.
    It's just a way to map the colors to list(range(nColors)).
    """
    rel = {}
    for i in range(len(colors)):
        rel[i] = colors[i]
    invRel = {v: k for k,v in rel.items()}
    for i in range(len(colors)):
        rel[i] = [colors[i]]
    return rel, invRel

def dummify(x, nChannels, rel=None):
    """
    Given a matrix and a relationship given by relDicts, this function returns
    a nColors x shape(x) matrix consisting only of ones and zeros. For each
    channel (corresponding to a color), each element will be 1 if in the 
    original matrix x that pixel is of the corresponding color.
    If rel is not specified, it is expected that the values of x range from
    0 to nChannels-1.
    """
    img = np.full((nChannels, x.shape[0], x.shape[1]), 0, dtype=np.uint8)
    if rel==None:
        for i in range(nChannels):
            img[i] = x==i
    else:
        for i in range(len(rel)):
            img[i] = np.isin(x,rel[i])
    return img

def dummifyColor(x, color):
    """
    Given a matrix x and a color, this function returns a 2-by-shape(x) matrix
    of ones and zeros. In one channel, the elements will be 1 if the pixel is
    of the given color. In the other channel, they will be 1 otherwise.
    """
    img = np.full((2, x.shape[0], x.shape[1]), 0, dtype=np.uint8)
    img[0] = x!=color
    img[1] = x==color
    return img

def updateBestFunction(t, f, bestScore, bestFunction):
    """
    Given a task t, a partial function f, a best score and a best function, 
    this function executes f to all the matrices in t.trainSamples. If the
    resulting score is lower than bestScore, then it returns f and the new
    best score. Otherwise, it returns bestFunction again.
    """
    fun = copy.deepcopy(f)
    score = 0
    for sample in t.trainSamples:
        pred = fun(sample.inMatrix)
        score += incorrectPixels(sample.outMatrix.m, pred)
    if score < bestScore:
        bestScore = score
        bestFunction = fun
    return bestFunction, bestScore

# %% Symmetrize

# if t.lrSymmetric or t.udSymmetric or t.d1Symmetric:
# if len(t.changingColors) == 1:
def symmetrize(matrix, axis, color=None, outColor=None, refColor=None):
    """
    Given a matrix and a color, this function tries to turn pixels of that
    given color into some other one in order to make the matrix symmetric.
    "axis" is a list or set specifying the symmetry axis (lr, ud, d1 or d2).
    """
    # Left-Right
    def LRSymmetrize(m):
        width = m.shape[1] - 1
        for i in range(m.shape[0]):
            for j in range(int(m.shape[1] / 2)):
                if m[i,j] != m[i,width-j]:
                    if color==None:
                        if m[i,j]==refColor and m[i,width-j]!=refColor:
                            m[i,width-j] = outColor
                        elif m[i,j]!=refColor and m[i,width-j]==refColor:
                            m[i,j] = outColor
                    else:
                        if m[i,j] == color:
                            m[i,j] = m[i,width-j]
                        elif m[i,width-j]==color:
                            m[i,width-j] = m[i,j]
        return m
    
    # Up-Down
    def UDSymmetrize(m):
        height = m.shape[0] - 1
        for i in range(int(m.shape[0] / 2)):
            for j in range(m.shape[1]):
                if m[i,j] != m[height-i,j]:
                    if color==None:
                        if m[i,j]==refColor and m[height-i,j]!=refColor:
                            m[height-i,j] = outColor
                        elif m[i,j]!=refColor and m[height-i,j]==refColor:
                            m[i,j] = outColor
                    else:
                        if m[i,j] == color:
                            m[i,j] = m[height-i,j]
                        elif m[height-i,j]==color:
                            m[height-i,j] = m[i,j]
        return m

    # Main diagonal
    def D1Symmetrize(m):
        for i,j in np.ndindex(m.shape):
            if m[i,j] != m[j,i]:
                if color==None:
                    if m[i,j]==refColor and m[j,i]!=refColor:
                        m[j,i] = outColor
                    elif m[i,j]!=refColor and m[j,i]==refColor:
                        m[i,j] = outColor
                else:
                    if m[i,j] == color:
                        m[i,j] = m[j,i]
                    elif m[j,i]==color:
                        m[j,i] = m[i,j]
        return m
    
    def D2Symmetrize(matrix):
        for i,j in np.ndindex(m.shape):
            if m[i,j] != m[m.shape[0]-j-1, m.shape[1]-i-1]:
                if color==None:
                    if m[i,j]==refColor and m[m.shape[0]-j-1, m.shape[1]-i-1]!=refColor:
                        m[m.shape[0]-j-1, m.shape[1]-i-1] = outColor
                    elif m[i,j]!=refColor and m[m.shape[0]-j-1, m.shape[1]-i-1]==refColor:
                        m[i,j] = outColor
                else:
                    if m[i,j] == color:
                        m[i,j] = m[m.shape[0]-j-1, m.shape[1]-i-1]
                    elif m[m.shape[0]-j-1, m.shape[1]-i-1]==color:
                        m[m.shape[0]-j-1, m.shape[1]-i-1] = m[i,j]
        return m
    
    m = matrix.m.copy()
    while True:
        prevMatrix = m.copy()
        if "lr" in axis:
            m = LRSymmetrize(m)
        if "ud" in axis:
            m = UDSymmetrize(m)
        if "d1" in axis:
            m = D1Symmetrize(m)
        if "d2" in axis:
            m = D2Symmetrize(m)
        if np.array_equal(prevMatrix, m):
            break
            
    return m

# %% Train and predict models  
"""
def trainCNNDummyCommonColors(t, commonColors, k, pad):
    nChannels = len(commonColors)+2
    model = Models.OneConvModel(nChannels, k, pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    for e in range(100): # numEpochs   
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            for c in s.colors:
                if c not in commonColors:
                    itColors = commonColors + [c]
                    rel, invRel = relDicts(itColors)
                    firstCC = True
                    for cc in s.colors:
                        if cc not in itColors:
                            if firstCC:
                                rel[nChannels-1] = [cc]
                                firstCC = False
                            else:
                                rel[nChannels-1].append(cc)
                            invRel[cc] = nChannels-1
                    x = dummify(s.inMatrix.m, nChannels, rel)
                    x = torch.tensor(x).unsqueeze(0).float()
                    y = s.outMatrix.m.copy()
                    for i,j in np.ndindex(y.shape):
                        y[i,j] = invRel[y[i,j]]
                    y = torch.tensor(y).unsqueeze(0).long()
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

@torch.no_grad()
def predictCNNDummyCommonColors(matrix, model, commonColors):
    m = matrix.m.copy()
    nChannels = len(commonColors)+2
    pred = np.zeros(m.shape)
    for c in matrix.colors:
        if c not in commonColors:
            itColors = commonColors + [c]
            rel, invRel = relDicts(itColors)
            firstCC = True
            for cc in matrix.colors:
                if cc not in itColors:
                    if firstCC:
                        rel[nChannels-1] = [cc]
                        firstCC = False
                    else:
                        rel[nChannels-1].append(cc)
            x = dummify(m, nChannels, rel)
            x = torch.tensor(x).unsqueeze(0).float()
            x = model(x).argmax(1).squeeze(0).numpy()
            for i,j in np.ndindex(m.shape):
                if m[i,j] == c:
                    pred[i,j] = rel[x[i,j]][0]
    return pred
"""

def trainCNNDummyColor(t, k, pad):
    """
    This function trains a CNN with only one convolution of filter k and with
    padding values equal to pad.
    The training samples will have two channels: the background color and any
    other color. The training loop loops through all the non-background colors
    of each sample, treating them independently.
    This is useful for tasks like number 3.
    """
    model = Models.OneConvModel(2, k, pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    for e in range(50): # numEpochs            
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            for c in s.colors:
                if c != t.backgroundColor:
                    x = dummifyColor(s.inMatrix.m, c)
                    x = torch.tensor(x).unsqueeze(0).float()
                    y = deBackgroundizeMatrix(s.outMatrix.m, c)
                    y = torch.tensor(y).unsqueeze(0).long()
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

@torch.no_grad()
def predictCNNDummyColor(matrix, model):
    """
    Predict function for a model trained using trainCNNDummyColor.
    """
    m = matrix.m.copy()
    pred = np.ones(m.shape, dtype=np.uint8) * matrix.backgroundColor
    for c in matrix.colors:
        if c != matrix.backgroundColor:
            x = dummifyColor(m, c)
            x = torch.tensor(x).unsqueeze(0).float()
            x = model(x).argmax(1).squeeze(0).numpy()
            for i,j in np.ndindex(m.shape):
                if x[i,j] != 0:
                    pred[i,j] = c
    return pred

def trainCNN(t, commonColors, nChannels, k=5, pad=0):
    """
    This function trains a CNN model with kernel k and padding value pad.
    It is required that all the training samples have the same number of colors
    (adding the colors in the input and in the output).
    It is also required that the output matrix has always the same shape as the
    input matrix.
    The colors are tried to be order in a specific way: first the colors that
    are common to every sample (commonColors), and then the others.
    """
    model = Models.OneConvModel(nChannels, k, pad)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    #losses = np.zeros(100)
    for e in range(100):
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            sColors = commonColors.copy()
            for c in s.colors:
                if c not in sColors:
                    sColors.append(c)
            rel, invRel = relDicts(sColors)
            x = dummify(s.inMatrix.m, nChannels, rel)
            x = torch.tensor(x).unsqueeze(0).float()
            y = s.outMatrix.m.copy()
            for i,j in np.ndindex(y.shape):
                y[i,j] = invRel[y[i,j]]
            y = torch.tensor(y).unsqueeze(0).long()
            y_pred = model(x)
            loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        #losses[e] = loss
    return model#, losses

@torch.no_grad()
def predictCNN(matrix, model, commonColors, nChannels):
    """
    Predict function for a model trained using trainCNN.
    """
    m = matrix.m.copy()
    pred = np.zeros(m.shape, dtype=np.uint8)
    sColors = commonColors.copy()
    for c in matrix.colors:
        if c not in sColors:
            sColors.append(c)
    rel, invRel = relDicts(sColors)
    if len(sColors) > nChannels:
        return m
    x = dummify(m, nChannels, rel)
    x = torch.tensor(x).unsqueeze(0).float()
    x = model(x).argmax(1).squeeze(0).numpy()
    for i,j in np.ndindex(m.shape):
        if x[i,j] not in rel.keys():
            pred[i,j] = x[i,j]
        else:
            pred[i,j] = rel[x[i,j]][0]
    return pred

def getBestCNN(t):
    """
    This function returns the best CNN with only one convolution, after trying
    different kernel sizes and padding values.
    There are as many channels as total colors or the minimum number of
    channels that is necessary.
    """
    kernel = [3,5,7]
    pad = [0,-1]    
    bestScore = 100000
    for k, p in product(kernel, pad):
        cc = list(range(10))
        model = trainCNN(t, commonColors=cc, nChannels=10, k=k, pad=p)
        score = sum([incorrectPixels(predictCNN(t.trainSamples[s].inMatrix, model, cc, 10), \
                                     t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        if score < bestScore:
            bestScore=score
            ret = partial(predictCNN, model=model, commonColors=cc, nChannels=10)
            if score==0:
                return ret
    return ret

def getBestSameNSampleColorsCNN(t):
    kernel = [3,5,7]
    pad = [0,-1]    
    bestScore = 100000
    for k, p in product(kernel, pad):
        cc = list(t.commonSampleColors)
        nc = t.trainSamples[0].nColors
        model = trainCNN(t, commonColors=cc, nChannels=nc, k=k, pad=p)
        score = sum([incorrectPixels(predictCNN(t.trainSamples[s].inMatrix, model, cc, nc), \
                                     t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        if score < bestScore:
            bestScore=score
            ret = partial(predictCNN, model=model, commonColors=cc, nChannels=nc)
            if score==0:
                return ret
            
    return ret

# %% CNN learning the output
    
def getNeighbourColors(m, i, j, border=0):
    """
    Given a matrix m and a position i,j, this function returns a list of the
    values of the neighbours of (i,j).
    """
    x = []
    y = m[i-1,j] if i>0 else border
    x.append(y)
    y = m[i,j-1] if j>0 else border
    x.append(y)
    y = m[i+1,j] if i<m.shape[0]-1 else border
    x.append(y)
    y = m[i,j+1] if j<m.shape[1]-1 else border
    x.append(y)
    return x

def getDNeighbourColors(m, i, j, kernel=3, border=0):
    """
    Given a matrix m and a position i,j, this function returns a list of the
    values of the diagonal neighbours of (i,j).
    """
    x = []
    y = m[i-1,j-1] if (i>0 and j>0) else border
    x.append(y)
    y = m[i+1,j-1] if (i<m.shape[0]-1 and j>0) else border
    x.append(y)
    y = m[i-1,j+1] if (i>0 and j<m.shape[1]-1) else border
    x.append(y)
    y = m[i+1,j+1] if (i<m.shape[0]-1 and j<m.shape[1]-1) else border
    x.append(y)
    
    if kernel==5:
        y = m[i-2,j-2] if (i>1 and j>1) else border
        x.append(y)
        y = m[i-1,j-2] if (i>0 and j>1) else border
        x.append(y)
        y = m[i,j-2] if j>1 else border
        x.append(y)
        y = m[i+1,j-2] if (i<m.shape[0]-1 and j>1) else border
        x.append(y)
        y = m[i+2,j-2] if (i<m.shape[0]-2 and j>1) else border
        x.append(y)
        y = m[i+2,j-1] if (i<m.shape[0]-2 and j>0) else border
        x.append(y)
        y = m[i+2,j] if i<m.shape[0]-2 else border
        x.append(y)
        y = m[i+2,j+1] if (i<m.shape[0]-2 and j<m.shape[1]-1) else border
        x.append(y)
        y = m[i+2,j+2] if (i<m.shape[0]-2 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i+1,j+2] if (i<m.shape[0]-1 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i,j+2] if j<m.shape[1]-2 else border
        x.append(y)
        y = m[i-1,j+2] if (i>0 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i-2,j+2] if (i>1 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i-2,j+1] if (i>1 and j<m.shape[1]-1) else border
        x.append(y)
        y = m[i-2,j] if i>1 else border
        x.append(y)
        y = m[i-2,j-1] if (i>1 and j>0) else border
        x.append(y)
    return x

def getAllNeighbourColors(m, i, j, kernel=3, border=0):
    return getNeighbourColors(m,i,j,border) + getDNeighbourColors(m,i,j,kernel,border)

def colorNeighbours(mIn, mOut ,i, j):
    if i>0:
        mIn[i-1,j] = mOut[i-1,j]
    if j>0:
        mIn[i,j-1] = mOut[i,j-1]
    if i<mIn.shape[0]-1:
        mIn[i+1,j] = mOut[i+1,j]
    if j<mIn.shape[1]-1:
        mIn[i,j+1] = mOut[i,j+1]
        
def colorDNeighbours(mIn, mOut, i, j):
    colorNeighbours(mIn, mOut ,i, j)
    if i>0 and j>0:
        mIn[i-1,j-1] = mOut[i-1,j-1]
    if i<mIn.shape[0]-1 and j>0:
        mIn[i+1,j-1] = mOut[i+1,j-1]
    if i>0 and j<mIn.shape[1]-1:
        mIn[i-1,j+1] = mOut[i-1,j+1]
    if i<mIn.shape[0]-1 and j<mIn.shape[1]-1:
        mIn[i+1,j+1] = mOut[i+1,j+1]
     
# if len(t.changedInColors)==1 (the background color, where everything evolves)
# 311/800 tasks satisfy this condition
# Do I need inMatrix.nColors+fixedColors to be iqual for every sample?
def evolve(t, kernel=3, border=0, includeRotations=False):
    def evolveInputMatrices(mIn, mOut, changeCIC=False):
        reference = [m.copy() for m in mIn]
        for m in range(len(mIn)):
            if changeCIC:
                for i,j in np.ndindex(mIn[m].shape):
                    if mIn[m][i,j] not in set.union(fixedColors, changedOutColors): 
                        colorDNeighbours(mIn[m], mOut[m], i, j)
                        break
            else:
                for i,j in np.ndindex(mIn[m].shape):
                    if referenceIsFixed and reference[m][i,j] in fixedColors:
                        colorDNeighbours(mIn[m], mOut[m], i, j)
                    elif reference[m][i,j] in changedOutColors:
                        colorDNeighbours(mIn[m], mOut[m], i, j)
                    
    nColors = t.trainSamples[0].nColors
    
    if not t.allEqual(t.sampleColors):
        sampleRel = []
        sampleInvRel = []
        commonColors = t.orderedColors
        for s in t.trainSamples:
            colors = commonColors.copy()
            for i,j in np.ndindex(s.inMatrix.shape):
                if s.inMatrix.m[i,j] not in colors:
                    colors.append(s.inMatrix.m[i,j])
                    if len(colors) == nColors:
                        break
            if len(colors) != nColors:
                for i,j in np.ndindex(s.outMatrix.shape):
                    if s.outMatrix.m[i,j] not in colors:
                        colors.append(s.outMatrix.m[i,j])
                        if len(colors) == nColors:
                            break
            rel, invRel = relDicts(colors)
            sampleRel.append(rel)
            sampleInvRel.append(invRel)
            
        fixedColors = set()
        for c in t.fixedColors:
            fixedColors.add(sampleInvRel[0][c])
        changedOutColors = set()
        for c in t.commonChangedOutColors:
            changedOutColors.add(sampleInvRel[0][c])
        for c in range(len(commonColors), nColors):
            changedOutColors.add(c)
    else:
        fixedColors = t.fixedColors
        changedOutColors = t.commonChangedOutColors
        
    referenceIsFixed = t.trainSamples[0].inMatrix.nColors == len(fixedColors)+1
    
    outMatrices = [s.outMatrix.m.copy() for s in t.trainSamples]
    referenceOutput = [s.inMatrix.m.copy() for s in t.trainSamples]
    
    if includeRotations:
        for i in range(1,4):
            for m in range(t.nTrain):
                outMatrices.append(np.rot90(outMatrices[m].copy(), i))
                referenceOutput.append(np.rot90(referenceOutput[m].copy(), i))
                    
    
    if not t.allEqual(t.sampleColors):
        for m in range(len(outMatrices)):
            for i,j in np.ndindex(outMatrices[m].shape):
                outMatrices[m][i,j] = sampleInvRel[m%t.nTrain][outMatrices[m][i,j]]
                referenceOutput[m][i,j] = sampleInvRel[m%t.nTrain][referenceOutput[m][i,j]]
    
    colorFromNeighboursK2 = {}
    colorFromNeighboursK3 = {}
    colorFromNeighboursK5 = {}
    for i in range(10):
        referenceInput = [m.copy() for m in referenceOutput]
        evolveInputMatrices(referenceOutput, outMatrices)
        if np.all([np.array_equal(referenceInput[m], referenceOutput[m]) for m in range(len(referenceInput))]):
            evolveInputMatrices(referenceOutput, outMatrices, True)
        for m in range(len(outMatrices)):
            for i,j in np.ndindex(referenceInput[m].shape):
                if referenceInput[m][i,j] != referenceOutput[m][i,j]:
                    neighbourColors = tuple(getNeighbourColors(referenceInput[m],i,j,border))
                    colorFromNeighboursK2[neighbourColors] = referenceOutput[m][i,j]
                    neighbourColors = tuple(getAllNeighbourColors(referenceInput[m],i,j,3,border))
                    colorFromNeighboursK3[neighbourColors] = referenceOutput[m][i,j]
                    neighbourColors = tuple(getAllNeighbourColors(referenceInput[m],i,j,5,border))
                    colorFromNeighboursK5[neighbourColors] = referenceOutput[m][i,j]
       
    colorFromNeighboursK2 = {k:v for k,v in colorFromNeighboursK2.items() if \
                             not all([x not in set.union(changedOutColors, fixedColors) for x in k])}
    colorFromNeighboursK3 = {k:v for k,v in colorFromNeighboursK3.items() if \
                             not all([x not in set.union(changedOutColors, fixedColors) for x in k])}
    colorFromNeighboursK5 = {k:v for k,v in colorFromNeighboursK5.items() if \
                             not all([x not in set.union(changedOutColors, fixedColors) for x in k])}
                
    colorfromNeighboursK3Background = {}
    for m in outMatrices:
        for i,j in np.ndindex(m.shape):
            if m[i,j] in t.commonChangedInColors:
                neighbourColors = tuple(getAllNeighbourColors(m,i,j,3,border))
                colorfromNeighboursK3Background[neighbourColors] = m[i,j]
        
    
    return [colorFromNeighboursK2, colorFromNeighboursK3,\
            colorFromNeighboursK5, colorfromNeighboursK3Background]

def applyEvolve(matrix, cfn, nColors, changedOutColors=set(), fixedColors=set(),\
                changedInColors=set(), referenceIsFixed=False, commonColors=set(),\
                kernel=None, border=0):
        
    def colorPixel(m,newM,i,j):
        if newM[i,j] not in cic and colorAroundCIC==False:
            return
        if kernel==None:
            tup3 = tuple(getAllNeighbourColors(m,i,j,3,border))
            tup5 = tuple(getAllNeighbourColors(m,i,j,5,border))
            #tup = getMostSimilarTuple(tup)
            if tup3 in cfn[1].keys():
                if tup3 in cfn[3].keys():
                    if tup5 in cfn[2].keys():
                        newM[i,j] = cfn[2][tup5]
                else:
                    newM[i,j] = cfn[1][tup3] 
            elif tup5 in cfn[2].keys():
                newM[i,j] = cfn[2][tup5]
        elif kernel==2:
            tup2 = tuple(getNeighbourColors(m,i,j,border))
            if tup2 in cfn[0].keys():
                newM[i,j] = cfn[0][tup2]
        elif kernel==3:
            tup3 = tuple(getAllNeighbourColors(m,i,j,3,border))
            if tup3 in cfn[1].keys():
                newM[i,j] = cfn[1][tup3]
        elif kernel==5:
            tup5 = tuple(getAllNeighbourColors(m,i,j,5,border))
            if tup5 in cfn[2].keys():
                newM[i,j] = cfn[2][tup5]
        
    def colorPixelsAround(m,newM,i,j):
        if i>0:
            colorPixel(m,newM,i-1,j)
        if j>0:
            colorPixel(m,newM,i,j-1)
        if i<m.shape[0]-1:
            colorPixel(m,newM,i+1,j)
        if j<m.shape[1]-1:
            colorPixel(m,newM,i,j+1)
        if i>0 and j>0:
            colorPixel(m,newM,i-1,j-1)
        if i<m.shape[0]-1 and j>0:
            colorPixel(m,newM,i+1,j-1)
        if i>0 and j<m.shape[1]-1:
            colorPixel(m,newM,i-1,j+1)
        if i<m.shape[0]-1 and j<m.shape[1]-1:
            colorPixel(m,newM,i+1,j+1)
    
    m = matrix.m.copy()
    
    if len(commonColors) > 0:
        colors = list(commonColors.copy())
        for i,j in np.ndindex(m.shape):
            if m[i,j] not in colors:
                colors.append(m[i,j])
        rel, invRel = relDicts(colors)
        
        for i,j in np.ndindex(m.shape):
            m[i,j] = invRel[m[i,j]]
            
        fc = set()
        for c in fixedColors:
            fc.add(invRel[c])        
        coc = set()
        for c in changedOutColors:
            coc.add(invRel[c])
        for c in range(len(commonColors), nColors):
            coc.add(c)
        cic = set()
        for c in changedInColors:
            cic.add(invRel[c])
    else:
        fc = fixedColors
        coc = changedOutColors
        cic = changedInColors
    
    it = 0
    colorAroundCIC=False
    while True:
        it += 1
        newM = m.copy()
        #seen = np.zeros(m.shape, dtype=np.bool)
        if colorAroundCIC:
            for i,j in np.ndindex(m.shape):
                colorPixelsAround(m,newM,i,j)
            colorAroundCIC=False
            m=newM.copy()
            continue
        for i,j in np.ndindex(m.shape):
            if referenceIsFixed and m[i,j] in fixedColors:
                colorPixelsAround(m,newM,i,j)
            elif m[i,j] in coc:
                colorPixelsAround(m,newM,i,j)
        if np.array_equal(newM,m):
            if it==1:
                colorAroundCIC=True
            else:
                break
        m = newM.copy()
        
    if len(commonColors) > 0:
        for i,j in np.ndindex(m.shape):
            if m[i,j] in rel.keys():
                m[i,j] = rel[m[i,j]][0]
            else:
                m[i,j] = rel[0][0] # Patch for bug in task 22
        
    return m
    
def getBestEvolve(t):
    nColors = t.trainSamples[0].nColors
    fc = t.fixedColors
    cic = t.commonChangedInColors
    coc = t.commonChangedOutColors
    refIsFixed = t.trainSamples[0].inMatrix.nColors == len(fc)+1
    
    bestScore = 1000
    bestFunction = None
    
    cfn = evolve(t)
    if t.allEqual(t.sampleColors):
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction

    else:
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction    
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    cfn = evolve(t, includeRotations=True)
    if t.allEqual(t.sampleColors):
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction    
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction

    else:
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction    
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction

# %% Linear Models

# If input always has the same shape and output always has the same shape
# And there is always the same number of colors in each sample    
def trainLinearModel(t, commonColors, nChannels):
    """
    This function trains a linear model.
    It is required that all the training samples have the same number of colors
    (adding the colors in the input and in the output).
    It is also required that all the input matrices have the same shape, and
    all the output matrices have the same shape.
    The colors are tried to be order in a specific way: first the colors that
    are common to every sample (commonColors), and then the others.
    """
    model = Models.LinearModel(t.inShape, t.outShape, nChannels)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for e in range(100):
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            sColors = commonColors.copy()
            for c in s.colors:
                if c not in sColors:
                    sColors.append(c)
            rel, invRel = relDicts(sColors)
            x = dummify(s.inMatrix.m, nChannels, rel)
            x = torch.tensor(x).unsqueeze(0).float()
            y = s.outMatrix.m.copy()
            for i,j in np.ndindex(y.shape):
                y[i,j] = invRel[y[i,j]]
            y = torch.tensor(y).unsqueeze(0).view(1,-1).long()
            y_pred = model(x)
            loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

@torch.no_grad()
def predictLinearModel(matrix, model, commonColors, nChannels, outShape):
    """
    Predict function for a model trained using trainLinearModel.
    """
    m = matrix.m.copy()
    pred = np.zeros(outShape, dtype=np.uint8)
    sColors = commonColors.copy()
    for c in matrix.colors:
        if c not in sColors:
            sColors.append(c)
    rel, invRel = relDicts(sColors)
    if len(sColors) > nChannels:
        return
    x = dummify(m, nChannels, rel)
    x = torch.tensor(x).unsqueeze(0).float()
    x = model(x).argmax(1).squeeze(0).view(outShape).numpy()
    for i,j in np.ndindex(outShape):
        if x[i,j] not in rel.keys():
            pred[i,j] = x[i,j]
        else:
            pred[i,j] = rel[x[i,j]][0]
    return pred

def trainLinearDummyModel(t):
    """
    This function trains a linear model.
    The training samples will have two channels: the background color and any
    other color. The training loop loops through all the non-background colors
    of each sample, treating them independently.
    """
    model = Models.LinearModelDummy(t.inShape, t.outShape)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for e in range(100):
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            for c in s.colors:
                if c != t.backgroundColor:
                    x = dummifyColor(s.inMatrix.m, c)
                    x = torch.tensor(x).unsqueeze(0).float()
                    y = deBackgroundizeMatrix(s.outMatrix.m, c)
                    y = torch.tensor(y).unsqueeze(0).long()
                    y = y.view(1, -1)
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model
    
@torch.no_grad()
def predictLinearDummyModel(matrix, model, outShape, backgroundColor):
    """
    Predict function for a model trained using trainLinearDummyModel.
    """
    m = matrix.m.copy()
    pred = np.zeros(outShape, dtype=np.uint8)
    for c in matrix.colors:
        if c != backgroundColor:
            x = dummifyColor(m, c)
            x = torch.tensor(x).unsqueeze(0).float()
            x = model(x).argmax(1).squeeze().view(outShape).numpy()
            for i,j in np.ndindex(outShape):
                if x[i,j] != 0:
                    pred[i,j] = c
    return pred

def trainLinearModelShapeColor(t):
    """
    For trainLinearModelShapeColor we need to have the same shapes in the input
    and in the output, and in the exact same positions. The training loop loops
    through all the shapes of the task, and its aim is to predict the final
    color of each shape.
    The features of the linear model are:
        - One feature per color in the task. Value of 1 if the shape has that
        color, 0 otherwise.
        - Several features representing the number of pixels of the shape.
        Only one of these features can be equal to 1, the rest will be equal
        to 0.
        - 5 features to encode the number of holes of the shape (0,1,2,3 or 4)
        - Feature encoding whether the shape is a square or not.
        - Feature encoding whether the shape is a rectangle or not.
        - Feature encoding whether the shape touches the border or not.
    """
    inColors = set.union(*t.changedInColors+t.changedOutColors) - t.unchangedColors
    colors = list(inColors) + list(set.union(*t.changedInColors+t.changedOutColors) - inColors)
    rel, invRel = relDicts(list(colors))
    shapePixelNumbers = t.shapePixelNumbers
    _,nPixelsRel = relDicts(shapePixelNumbers)
    # inFeatures: [colors that change], [number of pixels]+1, [number of holes] (0-4),
    # isSquare, isRectangle, isBorder
    nInFeatures = len(inColors) + len(shapePixelNumbers) + 1 + 5 + 3
    model = Models.SimpleLinearModel(nInFeatures, len(colors))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    num_epochs = 80
    trainShapes = []
    for s in t.trainSamples:
        for shapeI in range(s.inMatrix.nShapes):
            trainShapes.append((s.inMatrix.shapes[shapeI],\
                                s.outMatrix.shapes[shapeI].color))
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0
            for s,label in trainShapes:
                inFeatures = torch.zeros(nInFeatures)
                if s.color in inColors:
                    inFeatures[invRel[s.color]] = 1
                    inFeatures[len(inColors)+nPixelsRel[s.nPixels]] = 1
                    inFeatures[len(inColors)+len(shapePixelNumbers)+1+min(s.nHoles, 4)] = 1
                    inFeatures[nInFeatures-1] = int(s.isSquare)
                    inFeatures[nInFeatures-2] = int(s.isRectangle)
                    inFeatures[nInFeatures-1] = s.isBorder
                    #inFeatures[nInFeatures-4] = s.nHoles
                    #inFeatures[t.nColors+5] = s.position[0].item()
                    #inFeatures[t.nColors+6] = s.position[1].item()
                    y = torch.tensor(invRel[label]).unsqueeze(0).long()
                    x = inFeatures.unsqueeze(0).float()
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
            if loss == 0:
                continue
            loss.backward()
            optimizer.step()
            for p in model.parameters():
                p.data.clamp_(min=0.05, max=1)
    return model

@torch.no_grad()
def predictLinearModelShapeColor(matrix, model, colors, unchangedColors, shapePixelNumbers):
    """
    Predict function for a model trained using trainLinearModelShapeColor.
    """
    inColors = colors - unchangedColors
    colors = list(inColors) + list(colors - inColors)
    rel, invRel = relDicts(list(colors))
    _,nPixelsRel = relDicts(shapePixelNumbers)
    nInFeatures = len(inColors) + len(shapePixelNumbers) + 1 + 5 + 3
    pred = matrix.m.copy()
    for shape in matrix.shapes:
        if shape.color in inColors:
            inFeatures = torch.zeros(nInFeatures)
            inFeatures[invRel[shape.color]] = 1
            if shape.nPixels not in nPixelsRel.keys():
                inFeatures[len(inColors)+len(shapePixelNumbers)] = 1
            else:
                inFeatures[len(inColors)+nPixelsRel[shape.nPixels]] = 1
            inFeatures[len(inColors)+len(shapePixelNumbers)+1+min(shape.nHoles, 4)] = 1
            inFeatures[nInFeatures-1] = int(shape.isSquare)
            inFeatures[nInFeatures-2] = int(shape.isRectangle)
            inFeatures[nInFeatures-3] = shape.isBorder
            #inFeatures[nInFeatures-4] = shape.nHoles
            #inFeatures[nColors+5] = shape.position[0].item()
            #inFeatures[nColors+6] = shape.position[1].item()
            x = inFeatures.unsqueeze(0).float()
            y = model(x).squeeze().argmax().item()
            pred = changeColorShapes(pred, [shape], rel[y][0])
    return pred

# %% LSTM
def prepare_sequence(seq, to_ix):
    """
    Utility function for LSTM.
    """
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def trainLSTM(t, inColors, colors, inRel, outRel, reverse, order):
    """
    This function tries to train a model that colors shapes according to a
    sequence.
    """
    EMBEDDING_DIM = 10
    HIDDEN_DIM = 10
    model = Models.LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(inColors), len(colors))
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 150
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            inShapes = [shape for shape in s.inMatrix.shapes if shape.color in inColors]
            inSeq = sorted(inShapes, key=lambda x: (x.position[order[0]], x.position[order[1]]), reverse=reverse)
            inSeq = [shape.color for shape in inSeq]
            outShapes = [shape for shape in s.outMatrix.shapes if shape.color in colors]
            targetSeq = sorted(outShapes, key=lambda x: (x.position[order[0]], x.position[order[1]]), reverse=reverse)
            targetSeq = [shape.color for shape in targetSeq]
            inSeq = prepare_sequence(inSeq, inRel)
            targetSeq = prepare_sequence(targetSeq, outRel)
            tag_scores = model(inSeq)
            loss += loss_function(tag_scores, targetSeq)
        loss.backward()
        optimizer.step()
    return model
    
@torch.no_grad()
def predictLSTM(matrix, model, inColors, colors, inRel, rel, reverse, order):
    """
    Predict function for a model trained using trainLSTM.
    """
    m = matrix.m.copy()
    inShapes = [shape for shape in matrix.shapes if shape.color in inColors]
    if len(inShapes)==0:
        return m
    sortedShapes = sorted(inShapes, key=lambda x: (x.position[order[0]], x.position[order[1]]), reverse=reverse)
    inSeq = [shape.color for shape in sortedShapes]
    inSeq = prepare_sequence(inSeq, inRel)
    pred = model(inSeq).argmax(1).numpy()
    for shapeI in range(len(sortedShapes)):
        m = changeColorShapes(m, [sortedShapes[shapeI]], rel[pred[shapeI]][0])
    return m
             
def getBestLSTM(t):
    """
    This function tries to find out which one is the best-fitting LSTM model
    for the task t. The LSTM models try to change the color of shapes to fit
    sequences. Examples are tasks 175, 331, 459 or 594.
    4 LSTM models are trained, considering models that order shapes by X
    coordinage, models that order them by Y coordinate, and considering both
    directions of the sequence (normal and reverse).
    """
    colors = set.union(*t.changedInColors+t.changedOutColors)
    inColors = colors - t.unchangedColors
    if len(inColors) == 0:
        return partial(identityM)
    _,inRel = relDicts(list(inColors))
    colors = list(inColors) + list(colors - inColors)
    rel, outRel = relDicts(colors)
    
    for s in t.trainSamples:
        inShapes = [shape for shape in s.inMatrix.shapes if shape.color in inColors]
        outShapes = [shape for shape in s.outMatrix.shapes if shape.color in colors]
        if len(inShapes) != len(outShapes) or len(inShapes) == 0:
            return partial(identityM)
            
    reverse = [True, False]
    order = [(0,1), (1,0)]    
    bestScore = 1000
    for r, o in product(reverse, order):        
        model = trainLSTM(t, inColors=inColors, colors=colors, inRel=inRel,\
                          outRel=outRel, reverse=r, order=o)
        
        score = 0
        for s in t.trainSamples:
            m = predictLSTM(s.inMatrix, model, inColors, colors, inRel, rel, r, o)
            score += incorrectPixels(m, s.outMatrix.m)
        if score < bestScore:
            bestScore=score
            ret = partial(predictLSTM, model=model, inColors=inColors,\
                          colors=colors, inRel=inRel, rel=rel, reverse=r, order=o) 
            if bestScore==0:
                return ret
    return ret

# %% Other utility functions

def insertShape(matrix, shape):
    """
    Given a matrix (numpy.ndarray) and a Task.Shape, this function returns the
    same matrix but with the shape inserted.
    """
    m = matrix.copy()
    shapeM = shape.m.copy()
    for i,j in np.ndindex(shape.shape):
        if shapeM[i,j] != 255:
            if shape.position[0]+i<matrix.shape[0] and shape.position[1]+j<matrix.shape[1]:
                m[tuple(map(operator.add, (i,j), shape.position))] = shapeM[i,j]
    return m

def deleteShape(matrix, shape, backgroundColor):
    """
    Given a matrix (numpy.ndarray) and a Task.Shape, this function substitutes
    the shape by the background color of the matrix.
    """
    m = matrix.copy()
    for c in shape.pixels:
        m[tuple(map(operator.add, c, shape.position))] = backgroundColor
    return m

def colorMap(matrix, cMap):
    """
    cMap is a dict of color changes. Each input color can map to one and only
    one output color. Only valid if t.sameIOShapes.
    """
    m = matrix.m.copy()
    for i,j in np.ndindex(m.shape):
        if m[i,j] in cMap.keys(): # Otherwise, it means m[i,j] unchanged
            m[i,j] = cMap[matrix.m[i,j]]
    return m

def changeColorShapes(matrix, shapes, color):
    """
    Given a matrix (numpy.ndarray), a list of Task.Shapes (they are expected to
    be present in the matrix) and a color, this function returns the same
    matrix, but with the shapes of the list having the given color.
    """
    if len(shapes) == 0:
        return matrix
    m = matrix.copy()
    if color not in list(range(10)):
        return m
    for s in shapes:
        for c in s.pixels:
            m[tuple(map(operator.add, c, s.position))] = color
    return m

def changeShapes(m, inColor, outColor, bigOrSmall=None, isBorder=None):
    """
    Given a Task.Matrix, this function changes the Task.Shapes of the matrix
    that have color inColor to having the color outColor, if they satisfy the
    given conditions bigOrSmall (is the shape the smallest/biggest one?) and
    isBorder.
    """
    return changeColorShapes(m.m.copy(), m.getShapes(inColor, bigOrSmall, isBorder), outColor)

# %% Things with features
def isFixedShape(shape, fixedShapeFeatures):
    return shape.hasFeatures(fixedShapeFeatures)

def hasFeatures(candidate, reference):
    if all([i==False for i in reference]):
        return False
    for i in range(len(reference)):
        if reference[i] and not candidate[i]:
            return False
    return True

# %% Change Colors with features

def getClosestFixedShapeColor(shape, fixedShapes):
    def getDistance(x1, x2):
        x, y = sorted((x1, x2))
        if x[0] <= x[1] < y[0] and all( y[0] <= y[1] for y in (x1,x2)):
            return y[0] - x[1]
        return 0
    
    color = 0
    minDistance = 1000
    for fs in fixedShapes:
        xDist = getDistance([fs.position[0],fs.position[0]+fs.shape[0]-1], \
                            [shape.position[0],shape.position[0]+shape.shape[0]-1])
        yDist = getDistance([fs.position[1],fs.position[1]+fs.shape[1]-1], \
                            [shape.position[1],shape.position[1]+shape.shape[1]-1])
        
        if xDist+yDist < minDistance:
            minDistance = xDist+yDist
            color = fs.color
    return color

def getShapeFeaturesForColorChange(t, fixedShapeFeatures=None, fixedColors=None,\
                                   predict=False):  
    shapeFeatures = []
    
    if predict:
        matrices = [t]
    else:
        matrices = [s.inMatrix for s in t.trainSamples]
                  
    for m in range(len(matrices)):
        # Smallest and biggest shapes:
        biggestShape = 0
        smallestShape = 1000
        for shape in matrices[m].shapes:
            if shape.color not in fixedColors:
                if shape.nPixels>biggestShape:
                    biggestShape=shape.nPixels
                if shape.nPixels<smallestShape:
                    smallestShape=shape.nPixels
                    
        # Closest fixed shape
        if predict:
            fixedShapes = []
            for shape in matrices[0].shapes:
                if shape.hasFeatures(fixedShapeFeatures):
                    fixedShapes.append(shape)
        else:
            fixedShapes = t.trainSamples[m].fixedShapes
        
        for shape in matrices[m].shapes:
            shFeatures = []
            for c in range(10):
                shFeatures.append(shape.color==c)
            shFeatures.append(shape.isBorder)
            shFeatures.append(not shape.isBorder)
            shFeatures.append(shape.lrSymmetric)
            shFeatures.append(shape.udSymmetric)
            shFeatures.append(shape.d1Symmetric)
            shFeatures.append(shape.d2Symmetric)
            shFeatures.append(shape.isSquare)
            shFeatures.append(shape.isRectangle)
            for nPix in range(1,30):
                shFeatures.append(shape.nPixels==nPix)
            for nPix in range(1,6):
                shFeatures.append(shape.nPixels>nPix)
            for nPix in range(2,7):
                shFeatures.append(shape.nPixels<nPix)
            shFeatures.append((shape.nPixels%2)==0)
            shFeatures.append((shape.nPixels%2)==1)
            for h in range(5):
                shFeatures.append(shape.nHoles==h)
            shFeatures.append(shape.nPixels==biggestShape)
            shFeatures.append(shape.nPixels==smallestShape)
            #shFeatures.append(self.isUniqueShape(sh))
            #shFeatures.append(not self.isUniqueShape(sh))
            closestFixedShapeColor = getClosestFixedShapeColor(shape, fixedShapes)
            for c in range(10):
                shFeatures.append(closestFixedShapeColor==c)
                
            shapeFeatures.append(shFeatures)
            
    return shapeFeatures

def getColorChangesWithFeatures(t):
    shapeFeatures = getShapeFeaturesForColorChange(t, fixedColors=t.fixedColors)
    colorChangesWithFeatures = {}
    nFeatures = len(shapeFeatures[0])
    trueList = []
    falseList = []
    for i in range(nFeatures):
        trueList.append(True)
        falseList.append(False)
    for c in t.totalOutColors:
        colorChangesWithFeatures[c] = trueList
    changeCounter = Counter() # How many shapes change to color c?
    # First, initialise the colorChangesWithFeatures. For every outColor, we
    # detect which features are True for all the shapes that change to that color.
    shapeCounter = 0            
    for sample in t.trainSamples:
        for shape in sample.inMatrix.shapes:
            shapeChanges = False
            for i,j in np.ndindex(shape.shape):
                if shape.m[i,j]!=255:
                    if sample.outMatrix.m[shape.position[0]+i,shape.position[1]+j]!=shape.m[i,j]:
                        color = sample.outMatrix.m[shape.position[0]+i,shape.position[1]+j]
                        shapeChanges=True
                        break
            if shapeChanges:
                changeCounter[color] += 1
                colorChangesWithFeatures[color] = \
                [colorChangesWithFeatures[color][x] and shapeFeatures[shapeCounter][x]\
                 for x in range(nFeatures)]
            shapeCounter += 1
    # Now, there might be more True values than necessary in a certain entry
    # of colorChangesWithFeatures. Therefore, we try to determine the minimum
    # number of necessary True features.
    for c in t.totalOutColors:
        if colorChangesWithFeatures[c] == trueList:
            continue
        trueIndices = [i for i, x in enumerate(colorChangesWithFeatures[c]) if x]
        # First, check if only one feature is enough
        goodIndices = []
        for index in trueIndices:
            trueCount = 0
            featureList = falseList.copy()
            featureList[index] = True
            for sf in shapeFeatures:
                if hasFeatures(sf, featureList):
                    trueCount += 1
            # If the true count matches the number of changed shapes, we're done!
            if trueCount == changeCounter[c]:
                goodIndices.append(index)
        if len(goodIndices) > 0:
            featureList = falseList.copy()
            for index in goodIndices:
                featureList[index] = True
            colorChangesWithFeatures[c] = featureList
        # If we're not done, then check with combinations of 2 features
        else:
            for i,j in combinations(trueIndices, 2):
                trueCount = 0
                featureList = falseList.copy()
                featureList[i] = True
                featureList[j] = True
                for sf in shapeFeatures:
                    if hasFeatures(sf, featureList):
                        trueCount += 1
                # If the true count matches the number of changed shapes, we're done!
                if trueCount == changeCounter[c]:
                    colorChangesWithFeatures[c] = featureList
                    break   
                    
    return colorChangesWithFeatures

def changeShapesWithFeatures(matrix, ccwf, fixedColors, fixedShapeFeatures):
    """
    ccwp stands for 'color change with properties'. It's a dictionary. Its keys
    are integers encoding the color of the output shape, and its values are the
    properties that the input shape has to satisfy in order to execute the
    color change.
    """
    featureList = getShapeFeaturesForColorChange(matrix, fixedColors=fixedColors,\
                                                 fixedShapeFeatures=fixedShapeFeatures,\
                                                 predict=True)
    m = matrix.m.copy()
    sortedCcwf = {k: v for k, v in sorted(ccwf.items(), key=lambda item: sum(item[1]))}
    for color in sortedCcwf.keys():
        for sh in range(len(matrix.shapes)):
            if (matrix.shapes[sh].color in fixedColors) or \
            (matrix.shapes[sh].hasFeatures(fixedShapeFeatures)):
                continue
            if hasFeatures(featureList[sh], ccwf[color]):
                m = changeColorShapes(m, [matrix.shapes[sh]], color)
                #break
    return m


# %% Change pixels with features

def zoltanRecolor(t):
    """
    if t.sameIOShapes
    """
    Input = [s.inMatrix.m for s in t.trainSamples]
    Output = [s.outMatrix.m for s in t.trainSamples]
        
    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1,8):
            for Q2 in range(1,8):
                if Q1+Q2 == t:
                    Pairs.append((Q1,Q2))
                    
    for Q1, Q2 in Pairs:
        for v in range(4):
            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}
            
            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v ==2:
                            p1 = i%Q1
                        else:
                            p1 = (n-1-i)%Q1
                        if v == 0 or v ==3:
                            p2 = j%Q2
                        else :
                            p2 = (k-1-j)%Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)
                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            if possible:
                
                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v ==2:
                                p1 = i%Q1
                            else:
                                p1 = (n-1-i)%Q1
                            if v == 0 or v ==3:
                                p2 = j%Q2
                            else :
                                p2 = (k-1-j)%Q2
                           
                            color1 = x[i][j]
                            rule = (p1,p2,color1)
                            
                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False 
                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v
        
    if Best_Dict == -1:
        return [-1]#meaning that we didn't find a rule that works for the traning cases
    else:
        return [Best_Dict, Best_v, Best_Q1, Best_Q2]
    
def executeZoltanRecolor(matrix, Best_Dict, Best_v, Best_Q1, Best_Q2):
    m = np.zeros(matrix.shape, dtype = np.uint8)
    for i,j in np.ndindex(matrix.shape):
        if Best_v == 0 or Best_v ==2:
            p1 = i%Best_Q1
        else:
            p1 = (matrix.shape[0]-1-i)%Best_Q1
        if Best_v == 0 or Best_v ==3:
            p2 = j%Best_Q2
        else :
            p2 = (matrix.shape[1]-1-j)%Best_Q2
       
        color1 = matrix.m[i,j]
        rule = (p1, p2, color1)
        if (p1, p2, color1) in Best_Dict:
            m[i][j] = 0 + Best_Dict[rule]
        else:
            m[i][j] = 0 + color1
 
    return m
    


def doRulesWithReference(m, reference, rules):
    for i,j in np.ndindex(m.shape):
        y = (m[i,j], reference[i,j])
        if y in rules.keys():
            m[i,j] = rules[y]
    return m    

def doPixelMod2Row(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%2 == 0:
            reference[i,:] = onesRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3Row(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    twosRow = np.full(m.shape[1], 2, dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%3 == 0:
            reference[i,:] = onesRow.copy()
        elif i%3 == 1:
            reference[i,:] = twosRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2RowReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%2 == 0:
            reference[m.shape[0]-i-1,:] = onesRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3RowReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    twosRow = np.full(m.shape[1], 2, dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%3 == 0:
            reference[m.shape[0]-i-1,:] = onesRow.copy()
        elif i%3 == 1:
            reference[m.shape[0]-i-1,:] = twosRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2Col(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%2 == 0:
            reference[:,j] = onesCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3Col(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    twosCol = np.full(m.shape[0], 2, dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%3 == 0:
            reference[:,j] = onesCol.copy()
        elif j%3 == 1:
            reference[:,j] = twosCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2ColReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%2 == 0:
            reference[:,m.shape[1]-j-1] = onesCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3ColReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    twosCol = np.full(m.shape[0], 2, dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%3 == 0:
            reference[:,m.shape[1]-j-1] = onesCol.copy()
        elif j%3 == 1:
            reference[:,m.shape[1]-j-1] = twosCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2Alternate(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        reference[i,j] = (i+j)%2
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3Alternate(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        reference[i,j] = (i+j)%3
    m = doRulesWithReference(m, reference, rules)
    return m

def getPixelChangeCriteria(t):
    # Row
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%2 == 0:
                reference[i,:] = onesRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2Row, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        twosRow = np.full(sample.inMatrix.shape[1], 2, dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%3 == 0:
                reference[i,:] = onesRow.copy()
            elif i%3 == 1:
                reference[i,:] = twosRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3Row, rules=x)
    
    # Row Reverse
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%2 == 0:
                reference[sample.inMatrix.shape[0]-i-1,:] = onesRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2RowReverse, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        twosRow = np.full(sample.inMatrix.shape[1], 2, dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%3 == 0:
                reference[sample.inMatrix.shape[0]-i-1,:] = onesRow.copy()
            elif i%3 == 1:
                reference[sample.inMatrix.shape[0]-i-1,:] = twosRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3RowReverse, rules=x)

    # Col
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%2 == 0:
                reference[:,j] = onesCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2Col, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        twosCol = np.full(sample.inMatrix.shape[0], 2, dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%3 == 0:
                reference[:,j] = onesCol.copy()
            elif j%3 == 1:
                reference[:,j] = twosCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3Col, rules=x)
    
    # Col Reverse
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%2 == 0:
                reference[:,sample.inMatrix.shape[1]-j-1] = onesCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2ColReverse, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        twosCol = np.full(sample.inMatrix.shape[0], 2, dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%3 == 0:
                reference[:,sample.inMatrix.shape[1]-j-1] = onesCol.copy()
            elif j%3 == 1:
                reference[:,sample.inMatrix.shape[1]-j-1] = twosCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3ColReverse, rules=x)
    
    # Alternate
    # Mod2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        for i,j in np.ndindex(sample.inMatrix.shape):
            reference[i,j] = (i+j)%2
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2Alternate, rules=x)
    # Mod3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        for i,j in np.ndindex(sample.inMatrix.shape):
            reference[i,j] = (i+j)%3
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3Alternate, rules=x)
    
    return 0


def getPixelFeatures(m, i, j):
    """
    m is the matrix, and (i,j) is the position of the pixel.
    """
    pixelFeatures = []
    for color in range(10):
        pixelFeatures.append(m[i,j]==color)
    for x in range(2):
        pixelFeatures.append((i%2)==x)
        pixelFeatures.append((j%2)==x)
        for y in range(2):
            pixelFeatures.append((i%2)==x and (j%2)==y)
        for y in range(3):
            pixelFeatures.append((i%2)==x and (j%3)==y)
    for x in range(3):
        pixelFeatures.append((i%3)==x)
        pixelFeatures.append((j%3)==x) 
        for y in range(3):
            pixelFeatures.append((i%3)==x and (j%3)==y)
    pixelFeatures.append(((i+j)%2)==0)
    pixelFeatures.append(((i+j)%2)==1)
    
    return pixelFeatures

def getPixelChangesWithFeatures(t):
    """
    Only to be used if t.sameIOShapes.
    """
    pixelChangesWithFeatures = {}
    nFeatures = 41 # Is there a way to not hard code this?
    trueList = []
    falseList = []
    for i in range(nFeatures):
        trueList.append(True)
        falseList.append(False)
    for c in t.totalOutColors:
        pixelChangesWithFeatures[c] = trueList
        
    trueCounts = [0]*nFeatures
    trueCountsMatrix = [trueCounts]*nFeatures
    
    for sample in t.trainSamples:
        for i,j in np.ndindex(sample.inMatrix.shape):
            color = sample.outMatrix.m[i,j]
            pixelFeatures = getPixelFeatures(sample.inMatrix.m, i, j)
            pixelChangesWithFeatures[color] = \
            [pixelChangesWithFeatures[color][x] and pixelFeatures[x] for x in range(nFeatures)]
                        
            trueCounts = [trueCounts[x]+pixelFeatures[x] for x in range(nFeatures)]
            for x in range(nFeatures):
                trueCountsMatrix[x] = [trueCountsMatrix[x][y]+pixelFeatures[x]*pixelFeatures[y] for y in range(nFeatures)]
            
    # Now, there might be more True values than necessary in a certain entry
    # of colorChangesWithFeatures. Therefore, we try to determine the minimum
    # number of necessary True features.
    for c in t.totalOutColors:
        if pixelChangesWithFeatures[c] == trueList:
            continue
        trueIndices = [i for i, x in enumerate(pixelChangesWithFeatures[c]) if x]
        # First, check if only one feature is enough
        goodIndices = []
        for index in trueIndices:
            trueCount = 0
            featureList = falseList.copy()
            featureList[index] = True
            for sample in t.trainSamples:
                for i,j in np.ndindex(sample.inMatrix.shape):
                    if sample.outMatrix.m[i,j]==c and hasFeatures(getPixelFeatures(sample.inMatrix.m,i,j), featureList):
                        trueCount += 1
            if trueCount == trueCounts[index]:
                goodIndices.append(index)
        if len(goodIndices) > 0:
            featureList = falseList.copy()
            for index in goodIndices:
                featureList[index] = True
            pixelChangesWithFeatures[c] = featureList
            
        # If we're not done, then check with combinations of 2 features
        else:
            for x,y in combinations(trueIndices, 2):
                trueCount = 0
                featureList = falseList.copy()
                featureList[x] = True
                featureList[y] = True
                for sample in t.trainSamples:
                    for i,j in np.ndindex(sample.inMatrix.shape):
                        if sample.outMatrix.m[i,j]==c and hasFeatures(getPixelFeatures(sample.inMatrix.m,i,j), featureList):
                            trueCount += 1
                if trueCount == trueCountsMatrix[x][y]:
                    pixelChangesWithFeatures[c] = featureList
                    break
            
    return pixelChangesWithFeatures

def changePixelsWithFeatures(matrix, pcwf):
    m = matrix.m.copy()
    sortedPcwf = {k: v for k, v in sorted(pcwf.items(), key=lambda item: sum(item[1][10:-1]), reverse=True)}
    for i,j in np.ndindex(m.shape):
        pixelFeatures = getPixelFeatures(matrix.m, i, j)
        for color in sortedPcwf.keys():
            if hasFeatures(pixelFeatures, pcwf[color]):
                m[i,j] = color
                break
    return m
    

# %% Surround Shape

# TODO
def surroundShape(matrix, shape, color, nSteps = False, untilColor = False):
    def addPixel(i,j):
        if matrix[tuple(map(operator.add, c, shape.position))] == untilColor:
            return False
        else:
            matrix[tuple(map(operator.add, c, shape.position))] = color
            pixels.add((i, j))

    x = matrix.copy()
    pixels = shape.pixels.copy()
    while True:
        y = x.copy()
        for c in shape.pixels:
            addPixel(c[0]+1, c[1])
            addPixel(c[0]-1, c[1])
            addPixel(c[0]+1, c[1]+1)
            addPixel(c[0]+1, c[1]-1)
            addPixel(c[0]-1, c[1]+1)
            addPixel(c[0]-1, c[1]-1)
            addPixel(c[0], c[1]+1)
            addPixel(c[0], c[1]-1)
        x = y.copy()
    return x
    
# TODO
def surroundAllShapes(m, shape, shapeColor, surroundColor, nSteps = False, untilColor = False):
    x = m.m.copy()
    shapesToSurround = [s for s in m.shapes if s.color == shapeColor]
    for s in shapesToSurround:
        x = surroundShape(x, s, outColor, untilColor)
    return x

# TODO
def extendColorAcross(matrix, color, direction, until, untilBorder = True):
    m = matrix.copy()
    
    if direction == "v" or direction == "u":
        for j in range(m.shape[1]):
            colorCells = False
            for i in reversed(range(m.shape[0])):
                if not colorCells:
                    if matrix[i,j] == color:
                        colorCells = True
                else:
                    if matrix[i,j] == until:
                        colorCells = False
                    else:
                        m[i,j] = color
            if colorCells and not untilBorder:
                for i in range(m.shape[0]):
                    if matrix[i,j] == color:
                        break
                    m[i,j] = matrix[i,j]
                
    if direction == "v" or direction == "d":
        for j in range(m.shape[1]):
            colorCells = False
            for i in range(m.shape[0]):
                if not colorCells:
                    if matrix[i,j] == color:
                        colorCells = True
                else:
                    if matrix[i,j] == until:
                        colorCells = False
                    else:
                        m[i,j] = color
            if colorCells and not untilBorder:
                for i in reversed(range(m.shape[0])):
                    if matrix[i,j] == color:
                        break
                    m[i,j] = matrix[i,j]
                    
    #if direction == "h" or direction == "l":
    #if direction == "h" or direction == "r":
    return m

# %% Color longest line
def colorLongestLines(matrix, cic, coc, direction):
    """
    cic stands for "changedInColor"
    coc stands for "changedOutColor"
    direction can be one of 4 strings: 'v', 'h', 'hv', 'd' (vertical,
    horizontal, diagonal)
    It is assumed t.sameIOShapes
    """    
    m = matrix.m.copy()
    
    longest=0
    positions = set()
    if direction=='h':
        for i in range(m.shape[0]):
            count = 0
            for j in range(m.shape[1]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add((i,j))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add((i,m.shape[1]-1))
        for pos in positions:
            for j in range(pos[1]-longest, pos[1]):
                m[pos[0],j] = coc
        return m                
        
    elif direction=='v':
        for j in range(m.shape[1]):
            count = 0
            for i in range(m.shape[0]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add((i,j))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add((m.shape[0]-1,j))
        for pos in positions:
            for i in range(pos[0]-longest, pos[0]):
                m[i,pos[1]] = coc
        return m 
                        
    elif direction=='hv':
        longestH = 0
        longestV = 0
        positionsH = set()
        positionsV = set()
        for i in range(m.shape[0]):
            count = 0
            for j in range(m.shape[1]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longestH:
                        if count > longestH:
                            positionsH = set()
                        longestH = count
                        positionsH.add((i,j))
                    count = 0
            if count >= longestH:
                if count > longestH:
                    positionsH = set()
                longestH = count
                positionsH.add((i,m.shape[1]-1))
        for j in range(m.shape[1]):
            count = 0
            for i in range(m.shape[0]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longestV:
                        if count > longestV:
                            positionsV = set()
                        longestV = count
                        positionsV.add((i,j))
                    count = 0
            if count >= longestV:
                if count > longestV:
                    positionsV = set()
                longestV = count
                positionsV.add((m.shape[0]-1,j))
        for pos in positionsH:
            for j in range(pos[1]-longestH, pos[1]):
                m[pos[0],j] = coc
        for pos in positionsV:
            for i in range(pos[0]-longestV, pos[0]):
                m[i,pos[1]] = coc
        return m
    
    elif direction=='d':
        # Direction of main diagonal
        for i in reversed(range(m.shape[0])):
            count = 0
            jLimit = min(m.shape[1], m.shape[0]-i)
            for j in range(jLimit):
                if m[i+j,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((i+j-1,j-1), 'main'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((i+jLimit-1,jLimit-1), 'main'))
        for j in range(1, m.shape[1]):
            count = 0
            iLimit = min(m.shape[0], m.shape[1]-j)
            for i in range(iLimit):
                if m[i,j+i]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((i-1,j+i-1), 'main'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((iLimit-1,j+iLimit-1), 'main'))
                
        # Direction of counterdiagonal
        for i in range(m.shape[0]):
            count = 0
            jLimit = min(m.shape[1], i+1)
            for j in range(jLimit):
                if m[i-j,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((i-j+1, j-1), 'counter'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((i-jLimit+1,jLimit-1), 'counter'))
        for j in range(m.shape[1]):
            count = 0
            iLimit = min(m.shape[0], m.shape[1]-j)
            for i in range(iLimit):
                if m[m.shape[0]-i-1,j+i]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((m.shape[0]-i,j+i-1), 'counter'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((m.shape[0]-iLimit,j+iLimit-1), 'counter'))
        
        # Draw the lines
        for pos in positions:
            if pos[1]=='main':
                for x in range(longest):
                    m[pos[0][0]-x, pos[0][1]-x] = coc
            else:
                for x in range(longest):
                    m[pos[0][0]+x, pos[0][1]-x] = coc
        return m
    return m
    
# %% Move shapes    

def moveShape(matrix, shape, background, direction, until = -1, nSteps = 100):
    """
    'direction' can be l, r, u, d, ul, ur, dl, dr
    (left, right, up, down, horizontal, vertical, diagonal1, diagonal2)
    'until' can be a color or -1, which will be interpreted as border
    If 'until'==-2, then move until the shape encounters anything
    """
    m = matrix.copy()
    m = changeColorShapes(m, [shape], background)
    s = copy.deepcopy(shape)
    step = 0
    while True and step != nSteps:
        step += 1
        for c in s.pixels:
            pos = (s.position[0]+c[0], s.position[1]+c[1])
            if direction == "l":
                newPos = (pos[0], pos[1]-1)
            if direction == "r":
                newPos = (pos[0], pos[1]+1)
            if direction == "u":
                newPos = (pos[0]-1, pos[1])
            if direction == "d":
                newPos = (pos[0]+1, pos[1])
            if direction == "ul":
                newPos = (pos[0]-1, pos[1]-1)
            if direction == "ur":
                newPos = (pos[0]-1, pos[1]+1)
            if direction == "dl":
                newPos = (pos[0]+1, pos[1]-1)
            if direction == "dr":
                newPos = (pos[0]+1, pos[1]+1)
                
            if newPos[0] not in range(m.shape[0]) or \
            newPos[1] not in range(m.shape[1]):
                if until != -1 and until != -2:
                    return matrix.copy()
                else:
                    return insertShape(m, s)
            if until == -2 and m[newPos] != background:
                return insertShape(m, s)
            if m[newPos] == until:
                return insertShape(m, s)
            
        if direction == "l":
            s.position = (s.position[0], s.position[1]-1)
        if direction == "r":
            s.position = (s.position[0], s.position[1]+1)
        if direction == "u":
            s.position = (s.position[0]-1, s.position[1])
        if direction == "d":
            s.position = (s.position[0]+1, s.position[1])
        if direction == "ul":
            s.position = (s.position[0]-1, s.position[1]-1)
        if direction == "ur":
            s.position = (s.position[0]-1, s.position[1]+1)
        if direction == "dl":
            s.position = (s.position[0]+1, s.position[1]-1)
        if direction == "dr":
            s.position = (s.position[0]+1, s.position[1]+1)
      
    return insertShape(m, s) 
    
def moveAllShapes(matrix, background, direction, until, nSteps=100, color=None):
    """
    direction can be l, r, u, d, ul, ur, dl, dr, h, v, d1, d2, all, any
    """
    if color==None or color=="multiColor":
        shapesToMove = matrix.multicolorShapes
    elif color=="diagonalMultiColor":
        shapesToMove=matrix.multicolorDShapes
    elif color=="singleColor":
        shapesToMove = [s for s in matrix.shapes if s.color!=background]
    elif color=="diagonalSingleColor":
        shapesToMove = [s for s in matrix.dShapes if s.color!=background]
    else:
        shapesToMove = [s for s in matrix.shapes if s.color in color]
    if direction == 'l':
        shapesToMove.sort(key=lambda x: x.position[1])
    if direction == 'r':
        shapesToMove.sort(key=lambda x: x.position[1]+x.shape[1], reverse=True)
    if direction == 'u':
        shapesToMove.sort(key=lambda x: x.position[0])  
    if direction == 'd':
        shapesToMove.sort(key=lambda x: x.position[0]+x.shape[0], reverse=True)
    m = matrix.m.copy()
    for s in shapesToMove:
        newMatrix = m.copy()
        if direction == "any":
            for d in ['l', 'r', 'u', 'd', 'ul', 'ur', 'dl', 'dr']:
                newMatrix = moveShape(m, s, background, d, until)
                if not np.all(newMatrix == m):
                    return newMatrix
                    break
        else:
            m = moveShape(m, s, background, direction, until, nSteps)
    return m
    
def moveShapeToClosest(matrix, shape, background, until=None, diagonals=False, restore=True):
    """
    Given a matrix (numpy.ndarray) and a Task.Shape, this function moves the
    given shape until the closest shape with the color given by "until".
    """
    m = matrix.copy()
    s = copy.deepcopy(shape)
    m = deleteShape(m, shape, background)
    if until==None:
        if hasattr(shape, "color"):
            until=shape.color
        else:
            return matrix
    if until not in m:
        return matrix
    nSteps = 0
    while True:
        for c in s.pixels:
            pixelPos = tuple(map(operator.add, c, s.position))
            if nSteps <= pixelPos[0] and m[pixelPos[0]-nSteps, pixelPos[1]] == until:
                while nSteps>=0 and m[pixelPos[0]-nSteps, pixelPos[1]]!=background:
                    nSteps-=1
                s.position = (s.position[0]-nSteps, s.position[1])
                return insertShape(m, s)
            if pixelPos[0]+nSteps < m.shape[0] and m[pixelPos[0]+nSteps, pixelPos[1]] == until:
                while nSteps>=0 and m[pixelPos[0]+nSteps, pixelPos[1]]!=background:
                    nSteps-=1
                s.position = (s.position[0]+nSteps, s.position[1])
                return insertShape(m, s)
            if nSteps <= pixelPos[1] and m[pixelPos[0], pixelPos[1]-nSteps] == until:
                while nSteps>=0 and m[pixelPos[0], pixelPos[1]-nSteps]!=background:
                    nSteps-=1
                s.position = (s.position[0], s.position[1]-nSteps)
                return insertShape(m, s)
            if pixelPos[1]+nSteps < m.shape[1] and m[pixelPos[0], pixelPos[1]+nSteps] == until:
                while nSteps>=0 and m[pixelPos[0], pixelPos[1]+nSteps]!=background:
                    nSteps-=1
                s.position = (s.position[0], s.position[1]+nSteps)
                return insertShape(m, s)
            if diagonals:
                if nSteps <= pixelPos[0] and nSteps <= pixelPos[1] and \
                m[pixelPos[0]-nSteps, pixelPos[1]-nSteps] == until:
                    s.position = (s.position[0]-nSteps+1, s.position[1]-nSteps+1)
                    return insertShape(m, s)
                if nSteps <= pixelPos[0] and pixelPos[1]+nSteps < m.shape[1] and \
                m[pixelPos[0]-nSteps, pixelPos[1]+nSteps] == until:
                    s.position = (s.position[0]-nSteps+1, s.position[1]+nSteps-1)
                    return insertShape(m, s)
                if pixelPos[0]+nSteps < m.shape[0] and nSteps <= pixelPos[1] and \
                m[pixelPos[0]+nSteps, pixelPos[1]-nSteps] == until:
                    s.position = (s.position[0]+nSteps-1, s.position[1]-nSteps+1)
                    return insertShape(m, s)
                if pixelPos[0]+nSteps < m.shape[0] and pixelPos[1]+nSteps < m.shape[1] and \
                m[pixelPos[0]+nSteps, pixelPos[1]+nSteps] == until:
                    s.position = (s.position[0]+nSteps-1, s.position[1]+nSteps-1)
                    return insertShape(m, s)
        nSteps += 1
        if nSteps > m.shape[0] and nSteps > m.shape[1]:
            if restore:
                return matrix
            else:
                return m
        
def moveAllShapesToClosest(matrix, background, colorsToMove=None, until=None, \
                           diagonals=False, restore=True, fixedShapeFeatures=None):
    """
    This function moves all the shapes with color "colorsToMove" until the
    closest shape with color "until".
    """
    m = matrix.m.copy()
    fixedShapes = []
    if until == None:
        colorsToMove = []
        for shape in matrix.shapes:
            if hasFeatures(shape.boolFeatures, fixedShapeFeatures):
                fixedShapes.append(shape)
                colorsToMove.append(shape.color)
    elif colorsToMove==None:
        colorsToMove = matrix.colors - set([background, until])
    else:
        colorsToMove = [colorsToMove]
    for ctm in colorsToMove:
        for shape in matrix.shapes:
            if shape not in fixedShapes:
                if shape.color == ctm:
                    m = moveShapeToClosest(m, shape, background, until, diagonals, restore)
    return m

def getBestMoveShapes(t):
    """
    This functions tries to find, for a given task t, the best way to move
    shapes.
    """
    directions = ['l', 'r', 'u', 'd', 'ul', 'ur', 'dl', 'dr', 'any']
    bestScore = 1000
    bestFunction = partial(identityM)
        
    # Move all shapes in a specific direction, until a non-background thing is touched
    for d in directions:
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="singleColor")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="diagonalSingleColor")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="multiColor")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="diagonalMultiColor")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    colorsToChange = list(t.colors - t.fixedColors - set({t.backgroundColor}))
    ctc = [[c] for c in colorsToChange] + [colorsToChange] # Also all colors
    for c in ctc:
        for d in directions:
            moveUntil = colorsToChange + [-1] + [-2] #Border, any
            for u in moveUntil:
                f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                direction=d, until=u)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
    
    if t.backgroundColor != -1 and hasattr(t, 'fixedColors'):
        colorsToMove = set(range(10)) - set([t.backgroundColor]) - t.fixedColors
        for ctm in colorsToMove:
            for uc in t.unchangedColors:
                f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                                 background=t.backgroundColor, until=uc)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
                f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                                 background=t.backgroundColor, until=uc, restore=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
                f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                            background=t.backgroundColor, until=uc, diagonals=True)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
                f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                            background=t.backgroundColor, until=uc, diagonals=True, restore=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
    if all([len(sample.fixedShapes)>0 for sample in t.trainSamples]):
        f = partial(moveAllShapesToClosest, background=t.backgroundColor,\
                    fixedShapeFeatures = t.fixedShapeFeatures)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
        f = partial(moveAllShapesToClosest, background=t.backgroundColor,\
                    fixedShapeFeatures = t.fixedShapeFeatures, restore=False)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)  
        if bestScore==0:
            return bestFunction    
        
    return bestFunction

# %% Complete rectangles
def completeRectangles(matrix, sourceColor, newColor):
    """
    It is assumed that the background is clear.
    """
    m = matrix.m.copy()
    for s in matrix.multicolorDShapes:
        if hasattr(s, 'color') and s.color==sourceColor:
            newShape = copy.deepcopy(s)
            newShape.m[newShape.m==255] = newColor
            m = insertShape(m, newShape)
    return m

# %% Delete shapes
# Like that this only solves task 96. It's clearly not general enough.
def deletePixels(matrix, diagonals=False):
    """
    Given a matrix, this functions deletes all the pixels. This means that all
    the dShapes consisting of only 1 pixel are converted to the color that
    surrounds most of that pixel.
    """
    m = matrix.m.copy()
    if diagonals:
        shapes = matrix.dShapes
    else:
        shapes = matrix.shapes
    for s in shapes:
        if s.nPixels==1:
            surrColors = Counter()
            if s.position[0]>0:
                surrColors[m[s.position[0]-1, s.position[1]]] += 1
            if s.position[1]<m.shape[1]-1:
                surrColors[m[s.position[0], s.position[1]+1]] += 1
            if s.position[0]<m.shape[0]-1:
                surrColors[m[s.position[0]+1, s.position[1]]] += 1
            if s.position[1]>0:
                surrColors[m[s.position[0], s.position[1]-1]] += 1
            if len(set(surrColors.values()))==1:
                if s.position[0]>0 and s.position[1]>0:
                    surrColors[m[s.position[0]-1, s.position[1]-1]] += 1
                if s.position[0]>0 and s.position[1]<m.shape[1]-1:
                    surrColors[m[s.position[0]-1, s.position[1]+1]] += 1
                if s.position[0]<m.shape[0]-1 and s.position[1]<m.shape[1]-1:
                    surrColors[m[s.position[0]+1, s.position[1]+1]] += 1
                if s.position[0]<m.shape[0]-1 and s.position[1]>0:
                    surrColors[m[s.position[0]+1, s.position[1]-1]] += 1
            
            m[s.position[0],s.position[1]] = max(surrColors.items(), key=operator.itemgetter(1))[0]
    return m

# %% Connect Pixels

def connectPixels(matrix, pixelColor=None, connColor=None, fixedColors=set(),\
                  allowedChanges={}, lineExclusive=False):
    """
    Given a matrix, this function connects all the pixels that have the same
    color. This means that, for example, all the pixels between two red pixels
    will also be red.
    If "pixelColor" is specified, then only the pixels with the specified color
    will be connected.
    Ìf "connColor" is specified, the color used to connect the pixels will be
    the one given by this parameter.
    If there are any colors in the "fixedColors" set, then it is made sure that
    they remain unchanged.
    "allowedChanges" is a dictionary determining which color changes are
    allowed. It's exclusive with the options unchangedColors and connColor.
    """
    m = matrix.copy()
    # Row
    for i in range(m.shape[0]):
        lowLimit = 0
        while lowLimit < m.shape[1] and matrix[i, lowLimit] != pixelColor:
            lowLimit += 1
        lowLimit += 1
        upLimit = m.shape[1]-1
        while upLimit > lowLimit and matrix[i, upLimit] != pixelColor:
            upLimit -= 1
        if upLimit > lowLimit:
            if lineExclusive:
                for j in range(lowLimit, upLimit):
                    if matrix[i,j] == pixelColor:
                        lowLimit = upLimit
                        break
            for j in range(lowLimit, upLimit):
                if connColor != None:
                    if matrix[i,j] != pixelColor and matrix[i,j] not in fixedColors:
                        m[i,j] = connColor
                else:
                    if matrix[i,j] in allowedChanges.keys():
                        m[i,j] = allowedChanges[matrix[i,j]]
       
    # Column             
    for j in range(m.shape[1]):
        lowLimit = 0
        while lowLimit < m.shape[0] and matrix[lowLimit, j] != pixelColor:
            lowLimit += 1
        lowLimit += 1
        upLimit = m.shape[0]-1
        while upLimit > lowLimit and matrix[upLimit, j] != pixelColor:
            upLimit -= 1
        if upLimit > lowLimit:
            if lineExclusive:
                for i in range(lowLimit, upLimit):
                    if matrix[i,j] == pixelColor:
                        lowLimit = upLimit
                        break
            for i in range(lowLimit, upLimit):
                if connColor != None:
                    if matrix[i,j] != pixelColor and matrix[i,j] not in fixedColors:
                        m[i,j] = connColor
                else:
                    if matrix[i,j] in allowedChanges.keys():
                        m[i,j] = allowedChanges[matrix[i,j]]
 
    return m

def connectAnyPixels(matrix, pixelColor=None, connColor=None, fixedColors=set(),\
                     allowedChanges={}, lineExclusive=False):
    m = matrix.m.copy()
    if pixelColor==None:
        if connColor==None:
            for c in matrix.colors - set([matrix.backgroundColor]):
                m = connectPixels(m, c, c, lineExclusive=lineExclusive)
            return m
        else:
            for c in matrix.colors - set([matrix.backgroundColor]):
                m = connectPixels(m, c, connColor, lineExclusive=lineExclusive)
            return m
    else:
        if len(allowedChanges)>0:
            m = connectPixels(m, pixelColor, allowedChanges=allowedChanges,\
                              lineExclusive=lineExclusive)
        else:
            m = connectPixels(m, pixelColor, connColor, fixedColors, lineExclusive=lineExclusive)
    return m

def rotate(matrix, angle):
    """
    Angle can be 90, 180, 270
    """
    assert angle in [90, 180, 270], "Invalid rotation angle"
    if isinstance(matrix, np.ndarray):
        m = matrix.copy()
    else:
        m = matrix.m.copy()
    return np.rot90(m, int(angle/90))    
    
def mirror(matrix, axis):
    """
    Axis can be lr, up, d1, d2
    """
    if isinstance(matrix, np.ndarray):
        m = matrix.copy()
    else:
        m = matrix.m.copy()
    assert axis in ["lr", "ud", "d1", "d2"], "Invalid mirror axis"
    if axis == "lr":
        return np.fliplr(m)
    if axis == "ud":
        return np.flipud(m)
    if axis == "d1":
        return m.T
    if axis == "d2":
        return m[::-1,::-1].T

"""
def flipShape(matrix, shape, axis, background):
    # Axis can be lr, ud
    m = matrix.copy()
    smallM = np.ones((shape.shape[0]+1, shape.shape[1]+1), dtype=np.uint8) * background
    for c in shape.pixels:
        smallM[c] = shape.color
    if axis == "lr":
        smallM = np.fliplr(smallM)
    if axis == "ud":
        smallM = np.flipud(smallM)
    for i,j in np.ndindex(smallM.shape):
        m[shape.position[0]+i, shape.position[1]+j] = smallM[i,j]
    return m

def flipAllShapes(matrix, axis, color, background):
    m = matrix.m.copy()
    shapesToMirror = [s for s in matrix.shapes if s.color in color]
    for s in shapesToMirror:
        m = flipShape(m, s, axis, background)
    return m
"""

def mapPixels(matrix, pixelMap, outShape):
    """
    Given a Task.Matrix as input, this function maps each pixel of that matrix
    to an outputMatrix, given by outShape.
    The dictionary pixelMap determines which pixel in the input matrix maps
    to each pixel in the output matrix.
    """
    inMatrix = matrix.m.copy()
    m = np.zeros(outShape, dtype=np.uint8)
    for i,j in np.ndindex(outShape):
        m[i,j] = inMatrix[pixelMap[i,j]]
    return m

def switchColors(matrix, color1=None, color2=None):
    """
    This function switches the color1 and the color2 in the matrix.
    If color1 and color2 are not specified, then the matrix is expected to only
    have 2 colors, and they will be switched.
    """
    if type(matrix) == np.ndarray:
        m = matrix.copy()
    else:
        m = matrix.m.copy()
    if color1==None or color2==None:
        color1 = m[0,0]
        for i,j in np.ndindex(m.shape):
            if m[i,j]!=color1:
                color2 = m[i,j]
                break
    for i,j in np.ndindex(m.shape):
        if m[i,j]==color1:
            m[i,j] = color2
        else:
            m[i,j] = color1        
    return m

# %% Rotation things

# TODO (task 26)
def makeShapeRotationInvariant(matrix, color):
    m = matrix.m.copy()
    
    return m

# %% Follow row/col patterns
def identifyColor(m, pixelPos, c2c, rowStep=None, colStep=None):
    """
    Utility function for followPattern.
    """
    if colStep!=None and rowStep!=None:
        i = 0
        while i+pixelPos[0] < m.shape[0]:
            j = 0
            while j+pixelPos[1] < m.shape[1]:
                if m[pixelPos[0]+i, pixelPos[1]+j] != c2c:
                    return m[pixelPos[0]+i, pixelPos[1]+j]
                j += colStep
            i += rowStep
        return c2c
    
def identifyColStep(m, c2c):
    """
    Utility function for followPattern.
    """
    colStep = 1
    while colStep < int(m.shape[1]/2)+1:
        isGood = True
        for j in range(colStep):
            for i in range(m.shape[0]):
                block = 0
                colors = set()
                while j+block < m.shape[1]:
                    colors.add(m[i,j+block])
                    block += colStep
                if c2c in colors:
                    if len(colors) > 2:
                        isGood = False
                        break
                else:
                    if len(colors) > 1:
                        isGood = False
                        break
            if not isGood:
                break  
        if isGood:
            return colStep 
        colStep+=1 
    return m.shape[1]

def identifyRowStep(m, c2c):
    """
    Utility function for followPattern.
    """
    rowStep = 1
    while rowStep < int(m.shape[0]/2)+1:
        isGood = True
        for i in range(rowStep):
            for j in range(m.shape[1]):
                block = 0
                colors = set()
                while i+block < m.shape[0]:
                    colors.add(m[i+block,j])
                    block += rowStep
                if c2c in colors:
                    if len(colors) > 2:
                        isGood = False
                        break
                else:
                    if len(colors) > 1:
                        isGood = False
                        break
            if not isGood:
                break  
        if isGood:
            return rowStep 
        rowStep+=1  
    return m.shape[0]            

def followPattern(matrix, rc, colorToChange=None, rowStep=None, colStep=None):
    """
    Given a Task.Matrix, this function turns it into a matrix that follows a
    pattern. This will be made row-wise, column-wise or both, depending on the
    parameter "rc". "rc" can be "row", "column" or "both".
    'colorToChange' is the number corresponding to the only color that changes,
    if any.
    'rowStep' and 'colStep' are only to be given if the rowStep/colStep is the
    same for every train sample.
    """  
    m = matrix.m.copy()
            
    if colorToChange!=None:
        if rc=="col":
            rowStep=m.shape[0]
            if colStep==None:
                colStep=identifyColStep(m, colorToChange)
        if rc=="row":
            colStep=m.shape[1]
            if rowStep==None:
                rowStep=identifyRowStep(m, colorToChange)
        if rc=="both":
            if colStep==None and rowStep==None:
                colStep=identifyColStep(m, colorToChange)
                rowStep=identifyRowStep(m, colorToChange) 
            elif rowStep==None:
                rowStep=m.shape[0]
            elif colStep==None:
                colStep=m.shape[1]                       
        for i,j in np.ndindex((rowStep, colStep)):
            color = identifyColor(m, (i,j), colorToChange, rowStep, colStep)
            k = 0
            while i+k < m.shape[0]:
                l = 0
                while j+l < m.shape[1]:
                    m[i+k, j+l] = color
                    l += colStep
                k += rowStep
            
    return m

# %% Fill the blank
def fillTheBlankParameters(t):
    matrices = []
    for s in t.trainSamples:
        m = s.inMatrix.m.copy()
        blank = s.blankToFill
        m[blank.position[0]:blank.position[0]+blank.shape[0],\
          blank.position[1]:blank.position[1]+blank.shape[1]] = s.outMatrix.m.copy()
        matrices.append(Task.Matrix(m))
        
    x = []
    x.append(all([m.lrSymmetric for m in matrices]))
    x.append(all([m.udSymmetric for m in matrices]))
    x.append(all([m.d1Symmetric for m in matrices]))
    x.append(all([m.d2Symmetric for m in matrices]))
    return x

def fillTheBlank(matrix, params):
    m = matrix.m.copy()
    if len(matrix.blanks) == 0:
        return m
    blank = matrix.blanks[0]
    color = blank.color
    pred = np.zeros(blank.shape, dtype=np.uint8)
    
    # lr
    if params[0]:
        for i,j in np.ndindex(blank.shape):
            if m[blank.position[0]+i, m.shape[1]-1-(blank.position[1]+j)] != color:
                pred[i,j] = m[blank.position[0]+i, m.shape[1]-1-(blank.position[1]+j)]
    # ud
    if params[1]:
        for i,j in np.ndindex(blank.shape):
            if m[m.shape[0]-1-(blank.position[0]+i), blank.position[1]+j] != color:
                pred[i,j] = m[m.shape[0]-1-(blank.position[0]+i), blank.position[1]+j]
    # d1
    if params[2] and m.shape[0]==m.shape[1]:
        for i,j in np.ndindex(blank.shape):
            if m[blank.position[1]+j, blank.position[0]+i] != color:
                pred[i,j] = m[blank.position[1]+j, blank.position[0]+i]
    # d2 (persymmetric matrix)
    if params[3] and m.shape[0]==m.shape[1]:
        for i,j in np.ndindex(blank.shape):
            if m[m.shape[1]-1-(blank.position[1]+j), m.shape[0]-1-(blank.position[0]+i)] != color:
                pred[i,j] = m[m.shape[1]-1-(blank.position[1]+j), m.shape[0]-1-(blank.position[0]+i)]
    
    return pred
    
# %% Operations with more than one matrix

# All the matrices need to have the same shape

def pixelwiseAnd(matrices, falseColor, targetColor=None, trueColor=None):
    """
    This function returns the result of executing the pixelwise "and" operation
    in a list of matrices.
    
    Parameters
    ----------
    matrices: list
        A list of numpy.ndarrays of the same shape
    falseColor: int
        The color of the pixel in the output matrix if the "and" operation is
        false.
    targetColor: int
        The color to be targeted by the "and" operation. For example, if
        targetColor is red, then the "and" operation will be true for a pixel
        if all that pixel is red in all of the input matrices.
        If targetColor is None, then the "and" operation will return true if
        the pixel has the same color in all the matrices, and false otherwise.
    trueColor: int
        The color of the pixel in the output matrix if the "and" operation is
        true.
        If trueColor is none, the output color if the "and" operation is true
        will be the color of the evaluated pixel.
    """
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            if all([x[i,j] == matrices[0][i,j] for x in matrices]):
                if trueColor == None:
                    m[i,j] = matrices[0][i,j]
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
        else:
            if all([x[i,j] == targetColor for x in matrices]):
                if trueColor == None:
                    m[i,j] = matrices[0][i,j]
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
    return m

"""
def pixelwiseOr(matrices, falseColor, targetColor=None, trueColor=None, \
                trueValues=None):
    See pixelwiseAnd.
    trueValues is a list with as many elements as matrices.
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            isFalse = True
            for x in matrices:
                if x[i,j] != falseColor:
                    isFalse = False
                    if trueColor == None:
                        m[i,j] = x[i,j]
                    else:
                        m[i,j] = trueColor
                    break
            if isFalse:
                m[i,j] = falseColor
        else:
            if any([x[i,j] == targetColor for x in matrices]):
                if trueColor == None:
                    m[i,j] = targetColor
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
    return m
"""

def pixelwiseOr(matrices, falseColor, targetColor=None, trueColor=None, \
                trueValues=None):
    """
    See pixelwiseAnd.
    trueValues is a list with as many elements as matrices.
    """
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            trueCount = 0
            index = 0
            for x in matrices:
                if x[i,j] != falseColor:
                    trueCount += 1
                    trueIndex = index
                index += 1
            if trueCount==0:
                m[i,j] = falseColor
            else:
                if trueColor!=None:
                    m[i,j] = trueColor
                elif trueValues!=None:
                    if trueCount==1:
                        m[i,j] = trueValues[trueIndex]
                    else:
                        m[i,j] = matrices[trueIndex][i,j]
                else:
                    m[i,j] = matrices[trueIndex][i,j]
        else:
            if any([x[i,j] == targetColor for x in matrices]):
                if trueColor == None:
                    m[i,j] = targetColor
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
    return m

def pixelwiseXor(m1, m2, falseColor, targetColor=None, trueColor=None, \
                 firstTrue=None, secondTrue=None):
    """
    See pixelwiseAnd. The difference is that the Xor operation only makes sense
    with two input matrices.
    """
    m = np.zeros(m1.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            if (m1[i,j] == falseColor) != (m2[i,j] == falseColor):
                if trueColor == None:
                    if firstTrue == None:
                        if m1[i,j] != falseColor:
                            m[i,j] = m1[i,j]
                        else:
                            m[i,j] = m2[i,j]
                    else:
                        if m1[i,j] != falseColor:
                            m[i,j] = firstTrue
                        else:
                            m[i,j] = secondTrue
                else:
                    m[i,j] = trueColor     
            else:
                m[i,j] = falseColor
        else:
            if (m1[i,j] == targetColor) != (m2[i,j] == targetColor):
                if trueColor == None:
                    if firstTrue == None:
                        if m1[i,j] != falseColor:
                            m[i,j] = m1[i,j]
                        else:
                            m[i,j] = m2[i,j]
                    else:
                        if m1[i,j] != falseColor:
                            m[i,j] = firstTrue
                        else:
                            m[i,j] = secondTrue
                else:
                    m[i,j] = trueColor     
            else:
                m[i,j] = falseColor
    return m

# %% Downsize and Minimize
    
def getDownsizeFactors(matrix):
    """
    Still unused
    """
    xDivisors = set()
    for x in range(1, matrix.shape[0]):
        if (matrix.shape[0]%x)==0:
            xDivisors.add(x)
    yDivisors = set()
    for y in range(1, matrix.shape[1]):
        if (matrix.shape[1]%y)==0:
            yDivisors.add(y)
    
    downsizeFactors = set()
    for x,y in product(xDivisors, yDivisors):
        downsizeFactors.add((x,y))
 
    return downsizeFactors

def downsize(matrix, newShape, falseColor=None):
    """
    Given a matrix and a shape, this function returns a new matrix with the
    given shape. The elements of the return matrix are given by the colors of 
    each of the submatrices. Each submatrix is only allowed to have the
    background color and at most another one (that will define the output
    color of the corresponding pixel).
    """
    if falseColor==None:
        falseColor = matrix.backgroundColor
    if (matrix.shape[0]%newShape[0])!=0 or (matrix.shape[1]%newShape[1])!=0:
        return matrix.m.copy()
    xBlock = int(matrix.shape[0]/newShape[0])
    yBlock = int(matrix.shape[1]/newShape[1])
    m = np.full(newShape, matrix.backgroundColor, dtype=np.uint8)
    for i,j in np.ndindex(newShape[0], newShape[1]):
        color = -1
        for x,y in np.ndindex(xBlock, yBlock):
            if matrix.m[i*xBlock+x, j*yBlock+y] not in [matrix.backgroundColor, color]:
                if color==-1:
                    color = matrix.m[i*xBlock+x, j*yBlock+y]
                else:
                    return matrix.m.copy()
        if color==-1:
            m[i,j] = falseColor
        else:
            m[i,j] = color
    return m

def minimize(matrix):
    """
    Given a matrix, this function returns the matrix resulting from the
    following operations:
        If two consecutive rows are equal, delete one of them
        If two consecutive columns are equal, delete one of them
    """
    m = matrix.m.copy()
    x = 1
    for i in range(1, matrix.shape[0]):
        if np.array_equal(m[x,:],m[x-1,:]):
            m = np.delete(m, (x), axis=0)
        else:
            x+=1
    x = 1
    for i in range(1, matrix.shape[1]):
        if np.array_equal(m[:,x],m[:,x-1]):
            m = np.delete(m, (x), axis=1)
        else:
            x+=1
    return m
            
        

# %% Operations to extend matrices
    
def getFactor(matrix, factor):
    """
    Given a Task.Task.inShapeFactor (that can be a string), this function
    returns its corresponding tuple for the given matrix.
    """
    if factor == "squared":
        f = (matrix.shape[0], matrix.shape[1])
    elif factor == "xSquared":
        f = (matrix.shape[0], 1)
    elif factor == "ySquared":
        f = (1, matrix.shape[1])
    elif factor == "nColors":
        f = (matrix.nColors, matrix.nColors)
    elif factor == "nColors-1":
        f = (matrix.nColors-1, matrix.nColors-1)
    else:
        f = factor
    return f

def multiplyPixels(matrix, factor):
    """
    Factor is a 2-dimensional tuple.
    The output matrix has shape matrix.shape*factor. Each pixel of the input
    matrix is expanded by factor.
    """
    factor = getFactor(matrix, factor)
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i,j in np.ndindex(matrix.m.shape):
        for k,l in np.ndindex(factor):
            m[i*factor[0]+k, j*factor[1]+l] = matrix.m[i,j]
    return m

def multiplyMatrix(matrix, factor):
    """
    Copy the matrix "matrix" into every submatrix of the output, which has
    shape matrix.shape * factor.
    """
    factor = getFactor(matrix, factor)
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i,j in np.ndindex(factor):
        m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = matrix.m.copy()
    return m

def matrixTopLeft(matrix, factor, background=0):
    """
    Copy the matrix into the top left corner of the multiplied matrix
    """
    factor = getFactor(matrix, factor)
    m = np.full(tuple(s * f for s, f in zip(matrix.shape, factor)), background, dtype=np.uint8)
    m[0:matrix.shape[0], 0:matrix.shape[1]] = matrix.m.copy()
    return m
    
def matrixBotRight(matrix, factor, background=0):
    """
    Copy the matrix into the bottom right corner of the multiplied matrix
    """
    factor = getFactor(matrix, factor)
    m = np.full(tuple(s * f for s, f in zip(matrix.shape, factor)), background, dtype=np.uint8)
    m[(factor[0]-1)*matrix.shape[0]:factor[0]*matrix.shape[0], \
      (factor[1]-1)*matrix.shape[1]:factor[1]*matrix.shape[1]]
    return m

def getBestMosaic(t):
    """
    Given a task t, this function tries to find the best way to generate a
    mosaic, given that the output shape is always bigger than the input shape
    with a shape factor that makes sense.
    A mosaic is a matrix that takes an input matrix as reference, and then
    copies it many times. The copies can include rotations or mirrorings.
    """
    factor = t.inShapeFactor
    ops = []
    ops.append(partial(identityM))
    ops.append(partial(mirror, axis="lr"))
    ops.append(partial(mirror, axis="ud"))
    ops.append(partial(rotate, angle=180))
    if t.inMatricesSquared:
        ops.append(partial(mirror, axis="d1"))
        ops.append(partial(mirror, axis="d2"))
        ops.append(partial(rotate, angle=90))
        ops.append(partial(rotate, angle=270))
    bestOps = []
    for i in range(factor[0]):
        bestOps.append([])
        for j in range(factor[1]):
            bestScore = 1000
            bestOp = partial(identityM)
            for op in ops:
                score = 0
                for s in t.trainSamples:
                    inM = s.inMatrix.m.copy()
                    outM = s.outMatrix.m[i*inM.shape[0]:(i+1)*inM.shape[0], j*inM.shape[1]:(j+1)*inM.shape[1]]
                    score += incorrectPixels(op(inM),outM)
                if score < bestScore:
                    bestScore = score
                    bestOp = op
                    if score==0:
                        break
            bestOps[i].append(bestOp)
    return bestOps

def generateMosaic(matrix, ops, factor):
    """
    Generates a mosaic from the given matrix using the operations given in the
    list ops. The output matrix has shape matrix.shape*factor.
    """
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i in range(factor[0]):
        for j in range(factor[1]):
            m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = \
            ops[i][j](matrix)
    return m

# Only if the factor is squared
def getBestMultiplyMatrix(t):  
    def getFullMatrix(matrix, color):
        return np.full(matrix.shape, color, dtype=np.uint8)
    # Possible operations on the matrix
    ops = []
    ops.append(partial(identityM))
    ops.append(partial(mirror, axis="lr"))
    ops.append(partial(mirror, axis="ud"))
    ops.append(partial(rotate, angle=180))
    if t.inMatricesSquared:
        ops.append(partial(mirror, axis="d1"))
        ops.append(partial(mirror, axis="d2"))
        ops.append(partial(rotate, angle=90))
        ops.append(partial(rotate, angle=270))
    if all([n==2 for n in t.nInColors]):
        ops.append(partial(switchColors))
    # TODO You can to better than that, come on
    falseColor = 0
    
    # Conditions
    def trueCondition(matrix, pixel):
        return True
    def maxColor(matrix, pixel):
        return pixel==max(matrix.colorCount, key=matrix.colorCount.get)
    def minColor(matrix,pixel):
        return pixel==min(matrix.colorCount, key=matrix.colorCount.get)
    def isColor(matrix, pixel, color):
        return pixel==color
    def nonZero(matrix, pixel):
        return pixel!=0
    def zero(matrix, pixel):
        return pixel==0
    conditions = []
    conditions.append(partial(trueCondition))
    conditions.append(partial(maxColor))
    conditions.append(partial(minColor))
    conditions.append(partial(nonZero))
    conditions.append(partial(zero))
    for c in t.colors:
        conditions.append(partial(isColor, color=c))

    bestScore = 1000
    for op, cond in product(ops, conditions):
        score = 0
        for s in t.trainSamples:
            factor = getFactor(s.inMatrix, t.inShapeFactor)
            for i,j in np.ndindex(factor):
                inM = s.inMatrix.m.copy()
                outM = s.outMatrix.m[i*inM.shape[0]:(i+1)*inM.shape[0], j*inM.shape[1]:(j+1)*inM.shape[1]]
                if cond(s.inMatrix, inM[i,j]):
                    score += incorrectPixels(op(inM),outM)
                else:
                    score += incorrectPixels(getFullMatrix(inM, falseColor), outM)
        if score < bestScore:
            bestScore = score
            opCond = (op, cond)
            if score==0:
                return opCond
    return opCond

def doBestMultiplyMatrix(matrix, opCond):
    factor = matrix.shape
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i,j in np.ndindex(factor):
        if opCond[1](matrix, matrix.m[i,j]):
            m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = \
            opCond[0](matrix)
    return m

# %% Multiply pixels

def multiplyPixelsAndAnd(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseAnd with these two
    matrices as inputs
    """
    factor = getFactor(matrix, factor)
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseAnd([m, newM], falseColor)
    return multipliedM

def multiplyPixelsAndOr(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseOr with these two
    matrices as inputs
    """
    factor = getFactor(matrix, factor)
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseOr([m, newM], falseColor)
    return multipliedM

def multiplyPixelsAndXor(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseXor with these two
    matrices as inputs
    """
    factor = getFactor(matrix, factor)
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseXor(m, newM, falseColor)
    return multipliedM

# %% Operations considering all submatrices of task with outShapeFactor
    
def getSubmatrices(m, factor):
    """
    Given a matrix m and a factor, this function returns a list of all the
    submatrices with shape determined by the factor.
    """
    matrices = []
    nRows = int(m.shape[0] / factor[0])
    nCols = int(m.shape[1] / factor[1])
    for i,j in np.ndindex(factor):
        matrices.append(m[i*nRows:(i+1)*nRows, j*nCols:(j+1)*nCols])
    return matrices

def outputIsSubmatrix(t, isGrid=False):
    """
    Given a task t that has outShapeFactor, this function returns true if any
    of the submatrices is equal to the output matrix for every sample.
    """
    for sample in t.trainSamples:
        if isGrid:
            matrices = [c[0].m for c in sample.inMatrix.grid.cellList]
        else:
            matrices = getSubmatrices(sample.inMatrix.m, sample.outShapeFactor)
        anyIsSubmatrix = False
        for m in matrices:
            if np.array_equal(m, sample.outMatrix.m):
                anyIsSubmatrix = True
                break
        if not anyIsSubmatrix:
            return False
    return True

def selectSubmatrixWithMaxColor(matrix, color, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with most appearances
    of the color given. If the matrix is not a grid, an outShapeFactor must be
    specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    maxCount = 0
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Task.Matrix(mat)
        if color in m.colors:
            if m.colorCount[color]>maxCount:
                bestMatrix = mat.copy()
                maxCount = m.colorCount[color]
                matricesWithProperty = 1
            if m.colorCount[color]==maxCount:
                matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
    
def selectSubmatrixWithMinColor(matrix, color, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with least appearances
    of the color given. If the matrix is not a grid, an outShapeFactor must be
    specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    minCount = 1000
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Task.Matrix(mat)
        if color in m.colors:
            if m.colorCount[color]<minCount:
                bestMatrix = mat.copy()
                minCount = m.colorCount[color]
                matricesWithProperty = 1
            elif m.colorCount[color]==minCount:
                matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
    
def selectSubmatrixWithMostColors(matrix, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with the most number of
    colors. If the matrix is not a grid, an outShapeFactor must be specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    maxNColors = 0
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Task.Matrix(mat)
        if len(m.colorCount)>maxNColors:
            bestMatrix = mat.copy()
            maxNColors = len(m.colorCount)
            matricesWithProperty = 1
        elif len(m.colorCount)==maxNColors:
            matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
    
def selectSubmatrixWithLeastColors(matrix, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with the least number
    of colors. If the matrix is not a grid, an outShapeFactor must be
    specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    minNColors = 1000
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Task.Matrix(mat)
        if len(m.colorCount)<minNColors:
            bestMatrix = mat.copy()
            minNColors = len(m.colorCount)
            matricesWithProperty = 1
        elif len(m.colorCount)==minNColors:
            matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
        
def getBestSubmatrixPosition(t, outShapeFactor=None, isGrid=False):
    """
    Given a task t, and assuming that all the input matrices have the same
    shape and all the ouptut matrices have the same shape too, this function
    tries to check whether the output matrix is just the submatrix in a given
    position. If that's the case, it returns the position. Otherwise, it
    returns 0.
    """
    iteration = 0
    possiblePositions = []
    for sample in t.trainSamples:
        if isGrid:
            matrices = [c[0].m for c in sample.inMatrix.grid.cellList]
        else:
            matrices = getSubmatrices(sample.inMatrix.m, outShapeFactor)
            
        possiblePositions.append(set())
        for m in range(len(matrices)):
            if np.array_equal(matrices[m], sample.outMatrix.m):
                possiblePositions[iteration].add(m)
        
        iteration += 1
    positions = set.intersection(*possiblePositions)
    if len(positions)==1:
        return next(iter(positions))
    else:
        return 0
                
def selectSubmatrixInPosition(matrix, position, outShapeFactor=None, isGrid=False):
    """
    Given a matrix and a position, this function returns the submatrix that
    appears in the given position (submatrices are either defined by
    outShapeFactor or by the shape of the grid cells).
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    return matrices[position].copy()

def maxColorFromCell(matrix):
    """
    Only to be called if matrix.isGrid.
    Given a matrix with a grid, this function returns a matrix with the same
    shape as the grid. Every pixel of the matrix will be colored with the 
    color that appears the most in the corresponding cell of the grid.
    """
    m = np.zeros(matrix.grid.shape, dtype=np.uint8)
    for i,j  in np.ndindex(matrix.grid.shape):
        color = max(matrix.grid.cells[i][j][0].colorCount.items(), key=operator.itemgetter(1))[0]
        m[i,j] = color
    return m
        
def pixelwiseAndInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseAnd(matrices, falseColor, targetColor, trueColor)

def pixelwiseOrInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None, \
                             trueValues=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseOr(matrices, falseColor, targetColor, trueColor, trueValues)

def pixelwiseXorInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None, \
                              firstTrue=None, secondTrue=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseXor(matrices[0], matrices[1], falseColor, targetColor, trueColor, firstTrue, secondTrue)

# %% Operations considering all submatrices of a grid

def pixelwiseAndInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None):
    matrices = [c[0].m for c in matrix.grid.cellList]
    return pixelwiseAnd(matrices, falseColor, targetColor, trueColor)

def pixelwiseOrInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None, \
                                 trueValues=None):
    matrices = [c[0].m for c in matrix.grid.cellList]
    return pixelwiseOr(matrices, falseColor, targetColor, trueColor, trueValues)

def pixelwiseXorInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None, \
                                  firstTrue=None, secondTrue=None):
    m1 = matrix.grid.cellList[0][0].m.copy()
    m2 = matrix.grid.cellList[1][0].m.copy()
    return pixelwiseXor(m1, m2, falseColor, targetColor, trueColor, firstTrue, secondTrue)

# %% Stuff added by Roderic
#replicate shape
def getBestReplicateShapes(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    for attributes in [set(['MoCl'])]:
        for cc in [cc[0] for cc in t.colorChanges]:
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=False, multicolor=True, anchorType='all', anchorColor=cc,\
                                    mirror=None, rotate=0, allCombs=True, scale=1, deleteOriginal=False), bestScore, bestFunction)
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=False, multicolor=True, anchorType='all', anchorColor=cc,\
                                    mirror=None, rotate=0, allCombs=True, scale=1, deleteOriginal=True), bestScore, bestFunction)
            if bestScore == 0:
                return bestFunction
            for mirror in [None, 'lr', 'ud']:
                for rotate in range(0, 4):
                    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=False, multicolor=True, anchorType='all', anchorColor=cc,\
                                    mirror=mirror, rotate=rotate, allCombs=False, scale=1, deleteOriginal=False), bestScore, bestFunction)
                    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=False, multicolor=True, anchorType='all', anchorColor=cc,\
                                    mirror=mirror, rotate=rotate, allCombs=False, scale=1, deleteOriginal=True), bestScore, bestFunction)
                    if bestScore == 0:      
                        return bestFunction
                    
    for attributes in [set(['UnCo'])]:
        for cc in [cc[0] for cc in t.colorChanges]:
                bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                                mirror=None, rotate=0, allCombs=True, scale=1, deleteOriginal=True), bestScore, bestFunction)
                bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                                mirror=None, rotate=0, allCombs=True, scale=1, deleteOriginal=False), bestScore, bestFunction)
                """
                bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=False, multicolor=True, anchorType='all', anchorColor=cc,\
                                mirror=mirror, rotate=rotate, allCombs=True, scale=1, deleteOriginal=True), bestScore, bestFunction)
                bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=True, anchorType='all', anchorColor=cc,\
                                mirror=mirror, rotate=rotate, allCombs=True, scale=1, deleteOriginal=True), bestScore, bestFunction)
                bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=False, multicolor=True, anchorType='all', anchorColor=cc,\
                                mirror=mirror, rotate=rotate, allCombs=False, scale=1, deleteOriginal=True), bestScore, bestFunction)
                bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=True, anchorType='all', anchorColor=cc,\
                                mirror=mirror, rotate=rotate, allCombs=False, scale=1, deleteOriginal=True), bestScore, bestFunction)
                """
    return bestFunction

def replicateShapes(matrix, attributes, diagonal=False, multicolor=True, anchorType=None, anchorColor=0,\
                    mirror=None, rotate=0, allCombs=False, scale=1, deleteOriginal=False):
    m = matrix.m.copy()
    score = -1
    repShape = 0
    #first find the shape or shapes to replicate
    if diagonal:
        if multicolor:
            shList = matrix.multicolorDShapes
        else:
            shList = matrix.dShapes
    else:
        if multicolor:
            shList = matrix.multicolorShapes
        else:
            shList = matrix.shapes
            
    attrList = matrix.getShapeAttributes(matrix.backgroundColor, not multicolor, diagonal)
    for shi in range(len(shList)):
        if len(attrList[shi].intersection(attributes)) > score:
            repShape = shList[shi]
            score = len(attrList[shi].intersection(attributes))
    if repShape == 0:
        return m
    repList = [repShape] 
    
    if allCombs:
        repList = []
        for r in range(0,4):
            mr = np.rot90(repShape.m, r)
            newRep = copy.deepcopy(repShape)
            newRep.m = mr
            newRep.shape = mr.shape
            repList.append(newRep)
        for r in range(0,4):
            mr = np.rot90(repShape.m[::-1,::], r)
            newRep = copy.deepcopy(repShape)
            newRep.m = mr
            newRep.shape = mr.shape
            repList.append(newRep)
           
    elif mirror == 'lr':
        shList[0].m = shList[0].m[::,::-1]
    elif mirror == 'ud':
        shList[0].m = shList[0].m[::,::-1]
    elif rotate > 0:
        shList[0].m = np.rot90(shList[0].m,rotate)
        shList[0].shape = shList[0].m.shape
    
    if scale > 1:
        newRepList=[]
        for sc in range(4,1,-1):    
            for repShape in repList:
                newRep = copy.deepcopy(repShape)
                newRep.m = np.repeat(np.repeat(repShape.m, sc, axis=1), sc, axis=0)
                newRep.shape = (repShape.shape[0]*sc, repShape.shape[1]*sc)
                newRepList.append(newRep)
        repList = newRepList
    
    #then find places to replicate
    if anchorType == 'all':    
        for repSh in repList:
            for i in range(matrix.shape[0] - repSh.shape[0]+1):
                for j in range(matrix.shape[1] - repSh.shape[1]+1):
                    if np.all(np.logical_or(m[i:i+repSh.shape[0],j:j+repSh.shape[1]]==anchorColor,repSh.m==255)):
                        newInsert = copy.deepcopy(repSh)
                        newInsert.position = (i, j)
                        m = insertShape(m, newInsert)
       
    """            
    elif anchorType == 'subshape':
        for sh1 in repList:
            for sh2 in shList:
                if sh2.isSubshape(sh1,True,True) and len(sh1.pixels) < len(sh2.pixels):
                    for i in range(sh2.shape[0]-sh2.shape[0]):
                        if np.all(np.logical_or(sh2.m == 255, sh1.m == s2.m)):
                            newInsert = 
                            insertShape()
                    
    elif anchorType == 'pixel':
        continue
    """
    
    if deleteOriginal:
        m = deleteShape(m, repShape, matrix.backgroundColor)              
    return(m)
        
#overlapSubmatrices 
def overlapSubmatrices(matrix, colorHierarchy, shapeFactor=None):
    """
    This function returns the result of overlapping all submatrices of a given
    shape factor pixelswise with a given color hierarchy. Includes option to overlap
    all grid cells.     
    """
    if shapeFactor == None:
       submat = [t[0].m for t in matrix.grid.cellList]

    else:
        matrix = matrix.m
        sF = tuple(sin // sfact for sin, sfact in zip(matrix.shape, shapeFactor))
        submat = [matrix[sF[0]*i:sF[0]*(i+1),sF[1]*j:sF[1]*(j+1)] for i,j in np.ndindex(shapeFactor)]

    m = np.zeros(submat[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        m[i,j] = colorHierarchy[max([colorHierarchy.index(x[i,j]) for x in submat])]
    return m

#Cropshape
def getCropAttributes(t, diagonal, multicolor, sameColor=True):
    if diagonal and not multicolor:
        if t.nCommonInOutDShapes == 0:
            return set()
        attrs = set.intersection(*[s.inMatrix.getShapeAttributes(backgroundColor=0,\
                    singleColor=True, diagonals=True)[s.inMatrix.dShapes.index(s.commonDShapes[0][0])] for s in t.trainSamples])
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=0, singleColor=True, diagonals=True)
            for shi in range(len(s.inMatrix.dShapes)):
                if s.inMatrix.dShapes[shi] == s.commonDShapes[0][0]:
                    continue
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])
                    
    if not diagonal and not multicolor:
        if t.nCommonInOutShapes == 0:
            return set()
        attrs = set.intersection(*[s.inMatrix.getShapeAttributes(backgroundColor=0,\
                    singleColor=True, diagonals=False)[s.inMatrix.shapes.index(s.commonShapes[0][0])] for s in t.trainSamples])
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=0, singleColor=True, diagonals=False)
            for shi in range(len(s.inMatrix.shapes)):
                if s.inMatrix.shapes[shi] == s.commonShapes[0][0]:
                    continue
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi]) 
                        
    if not diagonal and multicolor:
        if not t.outIsInMulticolorShapeSize:
            return set()                               
        attrs = set()
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=0, singleColor=False, diagonals=False)
            crop = False
            for shi in range(len(s.inMatrix.multicolorShapes)):
                if s.inMatrix.multicolorShapes[shi].shape == s.outMatrix.shape and\
                np.all(np.logical_or(s.inMatrix.multicolorShapes[shi].m == s.outMatrix.m, s.inMatrix.multicolorShapes[shi].m==255)):
                    crop = True
                    if len(attrs) == 0:
                        attrs = shAttrs[shi]
                    attrs = attrs.intersection(shAttrs[shi])
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])    
        if not crop:
                return set()
    if diagonal and multicolor:
        if not t.outIsInMulticolorDShapeSize:
            return set()                               
        attrs = set()
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=0, singleColor=False, diagonals=True)
            crop = False
            for shi in range(len(s.inMatrix.multicolorDShapes)):
                if s.inMatrix.multicolorDShapes[shi].shape == s.outMatrix.shape and\
                np.all(np.logical_or(s.inMatrix.multicolorDShapes[shi].m == s.outMatrix.m, s.inMatrix.multicolorDShapes[shi].m==255)):
                    crop = True
                    if len(attrs) == 0:
                        attrs = shAttrs[shi]
                    attrs = attrs.intersection(shAttrs[shi])
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])    
        if not crop:
                return set()
    return(attrs - nonAttrs)
        
def getBestCropShape(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,True, False),\
                                                           backgroundColor=0, singleColor=True, diagonals=True), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,False, False),\
                                                           backgroundColor=0, singleColor=True, diagonals=False), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,True, True),\
                                                           backgroundColor=0, singleColor=False, diagonals=True), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,False, True),\
                                                           backgroundColor=0, singleColor=False, diagonals=False), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    for attr in ['LaSh', 'MoCo', 'MoCl', 'UnSh', 'UnSi']:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=0, singleColor=True, diagonals=True), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=0, singleColor=True, diagonals=False), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=0, singleColor=False, diagonals=True), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=0, singleColor=True, diagonals=False), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction
    
def cropShape(matrix, attributes, backgroundColor=0, singleColor=True, diagonals=True):
    """
    This function crops the shape out of a matrix with the maximum score according to attributes
    """
    if singleColor: 
        if diagonals:   
            shapeList = [sh for sh in matrix.dShapes]
        else:   
            shapeList = [sh for sh in matrix.shapes]
    else:
        if diagonals: 
            shapeList = [sh for sh in matrix.multicolorDShapes]
        else:
            shapeList = [sh for sh in matrix.multicolorShapes]
    bestShapes = []
    score = 0
    attrList = matrix.getShapeAttributes(backgroundColor, singleColor, diagonals)
    for i in range(len(shapeList)):
        shscore = len(attributes.intersection(attrList[i]))
        if shscore > score:
            score = shscore
            bestShapes = [i]
        elif shscore == score:
            bestShapes += [i]
    if len(bestShapes) == 0:
        return matrix
    bestShape = shapeList[bestShapes[0]].m
    bestShape[bestShape==255]=backgroundColor
    return bestShape
#   return matrix.m[bestShape.position[0]:bestShape.position[0]+bestShape.shape[0], bestShape.position[1]:bestShape.position[1]+bestShape.shape[1]]
    
#Crop a shape using a reference shape or set of shapes
def cropShapeReference(matrix, referenceShape, diagonal=True):
    """
    This part is not finished
    """
    if diagonal:
        shList = matrix.dShapes
    else:
        shList = matrix.shapes
 
    if len(referenceShape) != 1:
        return matrix.m

    else:
        foundRef = False
        for sh in shList:
            if sh == referenceShape[0]:
                refShape = sh
                foundRef = True
                break
        if not foundRef:
            return matrix.m

        #shape enclosed by references
        for sh in shList:
            if sh == refShape:
                continue
            if all(sh.position[i] >= refShape.position[i] for i in [0,1]) and all(sh.position[i]+sh.shape[i] <= refShape.position[i]+refShape.shape[i] for i in [0,1]):
                        sh=sh.m
                        sh[sh==255]=matrix.backgroundColor
                        return sh

        #otherwise return closest to reference
        bestShape = referenceShape[0]
        dist = 1000
        refPos = refShape.position
        refPixels = [(p[0]+refPos[0], p[1]+refPos[1]) for p in refShape.pixels]
        for sh in shList:
            if sh == refShape or sh.color == matrix.backgroundColor:
                continue
            #dist = abs((sh.position[0]-refPos[0])+(sh.position[1]-refPos[1])))
            for p2 in [(p[0]+sh.position[0], p[1]+sh.position[1]) for p in sh.pixels]:
                if min(abs((p[0]-p2[0]))+abs((p[1]-p2[1])) for p in refPixels) < dist:
                    bestShape = sh
                    dist = min(abs((p[0]-p2[0])+(p[1]-p2[1])) for p in refPixels)
        bestShape=bestShape.m
        bestShape[bestShape==255]=matrix.backgroundColor
        return bestShape         

def cropAllBackground(matrix):
    m = matrix.m.copy()
    bC = matrix.backgroundColor
    x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
    while x1 <= x2 and np.all(m[x1,:] == bC):
        x1 += 1
    while x2 >= x1 and np.all(m[x2,:] == bC):
        x2 -= 1
    while y1 <= y2 and np.all(m[:,y1] == bC):
        y1 += 1
    while y2 >= y1 and np.all(m[:,y2] == bC):
        y2 -= 1
    return(m[x1:x2+1,y1:y2+1])

def cropOnlyMulticolorShape(matrix, diagonals=False):
    """
    This function is supposed to be called if there is one and only one 
    multicolor shape in all the input samples. This function just returns it.
    """
    if diagonals:
        m = matrix.multicolorDShapes[0].m.copy()
        m[m==255] = matrix.multicolorDShapes[0].background
    else:
        m = matrix.multicolorShapes[0].m.copy()
        m[m==255] = matrix.multicolorShapes[0].background
    return m

def cropFullFrame(matrix, includeBorder=True, bigOrSmall = None):
    m = matrix.m.copy()
    if bigOrSmall == None and len(matrix.fullFrames) != 1:
        return m
    if bigOrSmall == "small":
        frame = matrix.fullFrames[-1]
    else:
        frame = matrix.fullFrames[0]
    if includeBorder:
        return m[frame.position[0]:frame.position[0]+frame.shape[0], \
                 frame.position[1]:frame.position[1]+frame.shape[1]]
    else:
        return m[frame.position[0]+1:frame.position[0]+frame.shape[0]-1, \
                 frame.position[1]+1:frame.position[1]+frame.shape[1]-1]
        

# %% Main function: getPossibleOperations
def getPossibleOperations(t, c):
    """
    Given a Task.Task t and a Candidate c, this function returns a list of all
    the possible operations that make sense applying to the input matrices of
    c.
    The elements of the list to be returned are partial functions, whose input
    is a Task.Matrix and whose output is a numpy.ndarray (2-dim matrix).
    """ 
    candTask = c.t
    x = [] # List to be returned
    
    ###########################################################################
    # Fill the blanks
    if t.fillTheBlank:
        params = fillTheBlankParameters(t)
        x.append(partial(fillTheBlank, params=params))
        
    # switchColors
    if all([n==2 for n in candTask.nInColors]):
        x.append(partial(switchColors))
        
    # downsize
    if candTask.sameOutShape:
        outShape = candTask.outShape
        x.append(partial(downsize, newShape=outShape))
        if t.backgroundColor!=-1:
            x.append(partial(downsize, newShape=outShape, falseColor=t.backgroundColor))
        
        
    # minimize
    x.append(partial(minimize))
        
    
    ###########################################################################
    # sameIOShapes
    if candTask.sameIOShapes:
        
        #######################################################################
        
        # ColorMap
        ncc = len(candTask.colorChanges)
        if len(set([cc[0] for cc in candTask.colorChanges])) == ncc and ncc != 0:
            x.append(partial(colorMap, cMap=dict(candTask.colorChanges)))
            
        # Symmetrize
        if all([len(x)==1 for x in candTask.changedInColors]):
            color = next(iter(candTask.changedInColors[0]))
            axis = []
            if candTask.lrSymmetric:
                axis.append("lr")
            if candTask.udSymmetric:
                axis.append("ud")
            if candTask.d1Symmetric:
                axis.append("d1")
            if candTask.d2Symmetric:
                axis.append("d2")
            x.append(partial(symmetrize, axis=axis, color=color))
            if candTask.totalOutColors==1:
                for fc in candTask.fixedColors:
                    x.append(partial(symmetrize, axis=axis, refColor=fc,\
                                     outColor=next(iter(candTask.totalOutColors))))
    
        # Complete rectangles
        if candTask.backgroundColor!=-1 and len(candTask.fixedColors)==1 and \
        len(candTask.colorChanges)==1:
            sc = next(iter(candTask.fixedColors))
            nc = next(iter(candTask.colorChanges))[1]
            x.append(partial(completeRectangles, sourceColor=sc, newColor=nc))
            
        # Change pixels with features
        #pcwf = getPixelChangesWithFeatures(candTask)
        #x.append(partial(changePixelsWithFeatures, pcwf=pcwf))
        
        #######################################################################
        # For LinearShapeModel we need to have the same shapes in the input
        # and in the output, and in the exact same positions.
        # This model predicts the color of the shape in the output.
        
        if candTask.onlyShapeColorChanges:
            ccwf = getColorChangesWithFeatures(candTask)
            fsf = candTask.fixedShapeFeatures
            x.append(partial(changeShapesWithFeatures, ccwf=ccwf, fixedColors=candTask.fixedColors,\
                             fixedShapeFeatures=fsf))
                
            if all(["getBestLSTM" not in str(op.func) for op in c.ops]):        
                x.append(getBestLSTM(candTask))
            
            # Other deterministic functions that change the color of shapes.
            for cc in candTask.commonColorChanges:
                for border, bs in product([True, False, None], ["big", "small", None]):
                    x.append(partial(changeShapes, inColor=cc[0], outColor=cc[1],\
                                     bigOrSmall=bs, isBorder=border))
            
            return x
        
        #######################################################################
        # Complete row/col patterns
        colStep=None
        rowStep=None
        if candTask.followsRowPattern:
            if candTask.allEqual(candTask.rowPatterns):
                rowStep=candTask.rowPatterns[0]
        if candTask.followsColPattern:
            if candTask.allEqual(candTask.colPatterns):
                colStep=candTask.colPatterns[0]
        if candTask.allEqual(candTask.changedInColors) and len(candTask.changedInColors[0])==1:
            c2c = next(iter(candTask.changedInColors[0]))
        else:
            c2c=None
                
        if candTask.followsRowPattern and candTask.followsColPattern:
            x.append(partial(followPattern, rc="both", colorToChange=c2c,\
                             rowStep=rowStep, colStep=colStep))
        elif candTask.followsRowPattern:
            x.append(partial(followPattern, rc="row", colorToChange=c2c,\
                             rowStep=rowStep, colStep=colStep))
        elif candTask.followsColPattern:
            x.append(partial(followPattern, rc="col", colorToChange=c2c,\
                             rowStep=rowStep, colStep=colStep))

        #######################################################################
        # CNNs
        
        #x.append(getBestCNN(candTask))
        if candTask.sameNSampleColors and all(["predictCNN" not in str(op.func) for op in c.ops]):
            x.append(getBestSameNSampleColorsCNN(candTask))

        """
        if t.backgroundColor != -1:
            model = trainCNNDummyColor(candTask, 5, -1)
            x.append(partial(predictCNNDummyColor, model=model))
            model = trainCNNDummyColor(candTask, 3, 0)
            x.append(partial(predictCNNDummyColor, model=model))
            #model = trainOneConvModelDummyColor(candTask, 7, -1)
            #x.append(partial(predictConvModelDummyColor, model=model))
        """
            
        #cc = list(t.commonSampleColors)
        #model = trainCNNDummyCommonColors(t, cc, 3, -1)
        #x.append(partial(predictCNNDummyCommonColors, model=model,\
        #                commonColors=cc))
        
        #######################################################################
        # Transformations if the color count is always the same:
        # Rotations, Mirroring, Move Shapes, Mirror Shapes, ...
        if candTask.sameColorCount:
            for axis in ["lr", "ud"]:
                x.append(partial(mirror, axis = axis))
            # You can only mirror d1/d2 or rotate if the matrix is squared.
            if candTask.inMatricesSquared:
                for axis in ["d1", "d2"]:
                    x.append(partial(mirror, axis = axis))
                for angle in [90, 180, 270]:
                    x.append(partial(rotate, angle = angle))
                
                                                         
            # Mirror shapes
            """
            for c in ctc:
                for d in ["lr", "ud"]:
                    x.append(partial(flipAllShapes, axis=d, color=c, \
                                     background=t.backgroundColor))
            """
            
            x.append(getBestMoveShapes(candTask))
                    
        #######################################################################
        # Other sameIOShapes functions
        # Move shapes
        #x.append(getBestMoveShapes(candTask))
        
        zp = zoltanRecolor(candTask)
        if len(zp)!=1:
            x.append(partial(executeZoltanRecolor, Best_Dict=zp[0], Best_v=zp[1], Best_Q1=zp[2], Best_Q2=zp[3]))
        
        fun = getPixelChangeCriteria(candTask)
        if fun != 0:
            x.append(fun)
        
        # Color longest lines
        if len(candTask.colorChanges)==1:
            change = next(iter(candTask.colorChanges))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='h'))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='v'))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='hv'))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='d'))

        # Connect Pixels
        x.append(partial(connectAnyPixels))
        if all([len(x)==1 for x in candTask.changedInColors]):
            x.append(partial(connectAnyPixels, connColor=next(iter(candTask.changedOutColors[0]))))

        fc = candTask.fixedColors
        #if hasattr(t, "fixedColors"):
        #    tfc = candTask.fixedColors
        #else:
        #    tfc = set()
        for pc in candTask.colors - candTask.commonChangedInColors:
            for cc in candTask.commonChangedOutColors:
                x.append(partial(connectAnyPixels, pixelColor=pc, \
                                 connColor=cc, fixedColors=fc))
        for pc in candTask.colors - candTask.commonChangedInColors:
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges)))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges),\
                             lineExclusive=True))
                
        for cc in candTask.commonColorChanges:
            for cc in candTask.commonColorChanges:
                for border, bs in product([True, False, None], ["big", "small", None]):
                    x.append(partial(changeShapes, inColor=cc[0], outColor=cc[1],\
                                     bigOrSmall=bs, isBorder=border))
        
        #Replicate:
        #if candTask.nCommonInOutMulticolorShapes > 0:
        """
        x.append(getBestReplicateShapes(candTask))
        """
        
        # TODO
        """
        if all([len(s.inMatrix.multicolorShapes)==1 for s in candTask.trainSamples+candTask.testSamples]) and\
        all([len(s.outMatrix.multicolorShapes)==1 for s in candTask.testSamples]):
            if all([s.outMatrix.multicolorShapes[0].isRotationInvariant() for s in candTask.trainSamples]):
                for color in candTask.commonChangedOutColors:
                    x.append(makeShapeRotationInvariant, color=color)
        """
                    
    ###########################################################################
    # Cases in which the input has always the same shape, and the output too
    if candTask.sameInShape and candTask.sameOutShape and \
    all(candTask.trainSamples[0].inMatrix.shape == s.inMatrix.shape for s in candTask.testSamples):
        """
        if candTask.backgroundColor != -1:
            model = trainLinearDummyModel(candTask)
            x.append(partial(predictLinearDummyModel, model=model, \
                             outShape=candTask.outShape,\
                             backgroundColor=candTask.backgroundColor))
        
        if candTask.sameNSampleColors:
            cc = list(candTask.commonSampleColors)
            nc = candTask.trainSamples[0].nColors
            model = trainLinearModel(candTask, cc, nc)
            x.append(partial(predictLinearModel, model=model, commonColors=cc,\
                             nChannels=nc, outShape=candTask.outShape))
        """
            
        if candTask.sameNSampleColors:
            #Cases where output colors are a subset of input colors
            if candTask.sameNInColors and all(s.inHasOutColors for s in candTask.trainSamples):
                if hasattr(candTask, 'outShapeFactor') or (hasattr(candTask,\
                              'gridCellIsOutputShape') and candTask.gridCellIsOutputShape):
                    ch = dict(sum([Counter(s.outMatrix.colorCount) for s in candTask.trainSamples],Counter()))
                    ch = sorted(ch, key=ch.get)
                    if candTask.backgroundColor in ch:
                        ch.remove(candTask.backgroundColor)
                    ch = list(set([0,1,2,3,4,5,6,7,8,9]).difference(set(ch))) + ch
                    if hasattr(candTask, 'outShapeFactor'):
                        x.append(partial(overlapSubmatrices, colorHierarchy=ch, shapeFactor=candTask.outShapeFactor))
                    else:
                        x.append(partial(overlapSubmatrices, colorHierarchy=ch))
        
        pixelMap = Models.pixelCorrespondence(candTask)
        if len(pixelMap) != 0:
            x.append(partial(mapPixels, pixelMap=pixelMap, outShape=candTask.outShape))
    
    ###########################################################################
    # Evolve
    #if candTask.sameIOShapes and all([len(x)==1 for x in candTask.changedInColors]) and\
    #len(candTask.commonChangedInColors)==1 and candTask.sameNSampleColors:
    #    x.append(getBestEvolve(candTask))
        
    ###########################################################################
    # Other cases
    
    if hasattr(candTask, 'inShapeFactor'):
        x.append(partial(multiplyPixels, factor=candTask.inShapeFactor))
        x.append(partial(multiplyMatrix, factor=candTask.inShapeFactor))
        
        for c in candTask.commonSampleColors:
            x.append(partial(multiplyPixelsAndAnd, factor=candTask.inShapeFactor,\
                             falseColor=c))
            x.append(partial(multiplyPixelsAndOr, factor=candTask.inShapeFactor,\
                             falseColor=c))
            x.append(partial(multiplyPixelsAndXor, factor=candTask.inShapeFactor,\
                             falseColor=c))
            
        if type(candTask.inShapeFactor)==tuple:
            ops = getBestMosaic(candTask)
            x.append(partial(generateMosaic, ops=ops, factor=candTask.inShapeFactor))
            
        if all([s.inMatrix.shape[0]**2 == s.outMatrix.shape[0] and \
                s.inMatrix.shape[1]**2 == s.outMatrix.shape[1] for s in candTask.trainSamples]):
            opCond = getBestMultiplyMatrix(candTask)
            x.append(partial(doBestMultiplyMatrix, opCond=opCond))
            
            
    if hasattr(candTask, 'outShapeFactor'):
        if outputIsSubmatrix(candTask):
            for color in range(10):
                x.append(partial(selectSubmatrixWithMaxColor, color=color, outShapeFactor=candTask.outShapeFactor))
                x.append(partial(selectSubmatrixWithMinColor, color=color, outShapeFactor=candTask.outShapeFactor))
            x.append(partial(selectSubmatrixWithMostColors, outShapeFactor=candTask.outShapeFactor))
            x.append(partial(selectSubmatrixWithLeastColors, outShapeFactor=candTask.outShapeFactor))
            position = getBestSubmatrixPosition(candTask, outShapeFactor=candTask.outShapeFactor)
            x.append(partial(selectSubmatrixInPosition, position=position, outShapeFactor=candTask.outShapeFactor))
        
        # Pixelwise And
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseAndInSubmatrices, factor=candTask.outShapeFactor,\
                             falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseAndInSubmatrices, \
                                     factor=candTask.outShapeFactor, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
        
        # Pixelwise Or
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseOrInSubmatrices, factor=candTask.outShapeFactor,\
                             falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseOrInSubmatrices, \
                                     factor=candTask.outShapeFactor, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
        if candTask.backgroundColor!=-1:
            colors = candTask.commonOutColors - candTask.commonInColors
            if candTask.outShapeFactor[0]*candTask.outShapeFactor[1]==len(colors):
                for c in permutations(colors, len(colors)):
                    x.append(partial(pixelwiseOrInSubmatrices, factor=candTask.outShapeFactor,\
                                     falseColor=candTask.backgroundColor,\
                                     trueValues = c))
        
        # Pixelwise Xor
        if candTask.outShapeFactor in [(2,1), (1,2)]:
            for c in candTask.commonOutColors:
                x.append(partial(pixelwiseXorInSubmatrices, factor=candTask.outShapeFactor,\
                                 falseColor=c))
            if len(candTask.commonOutColors - candTask.commonInColors)==2 and\
            candTask.backgroundColor!=-1:
                colors = candTask.commonOutColors - candTask.commonInColors
                for c in permutations(colors, 2):
                    x.append(partial(pixelwiseXorInSubmatrices, falseColor=candTask.backgroundColor,\
                                     firstTrue=c[0], secondTrue=c[1]))
            if len(candTask.totalOutColors) == 2:
                for target in candTask.totalInColors:
                    for c in permutations(candTask.totalOutColors, 2):
                        x.append(partial(pixelwiseXorInSubmatrices, \
                                         factor=candTask.outShapeFactor, falseColor=c[0],\
                                         targetColor=target, trueColor=c[1]))
    
    if candTask.inputIsGrid:
        if all([s.inMatrix.grid.shape==s.outMatrix.shape for s in candTask.trainSamples]):
            x.append(partial(maxColorFromCell))
    
    if hasattr(candTask, 'gridCellIsOutputShape') and candTask.gridCellIsOutputShape:
        if outputIsSubmatrix(candTask, isGrid=True):
            for color in range(10):
                x.append(partial(selectSubmatrixWithMaxColor, color=color, isGrid=True))
                x.append(partial(selectSubmatrixWithMinColor, color=color, isGrid=True))
            x.append(partial(selectSubmatrixWithMostColors, isGrid=True))
            x.append(partial(selectSubmatrixWithLeastColors, isGrid=True))
            position = getBestSubmatrixPosition(candTask, isGrid=True)
            x.append(partial(selectSubmatrixInPosition, position=position, isGrid=True))
        
        # Pixelwise And
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseAndInGridSubmatrices, falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseAndInGridSubmatrices, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
                        
        # Pixelwise Or
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseOrInGridSubmatrices, falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseOrInGridSubmatrices, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
        if candTask.backgroundColor!=-1:
            colors = candTask.commonOutColors - candTask.commonInColors
            if candTask.trainSamples[0].inMatrix.grid.nCells==len(colors):
                for c in permutations(colors, len(colors)):
                    x.append(partial(pixelwiseOrInGridSubmatrices,\
                                     falseColor=candTask.backgroundColor,\
                                     trueValues = c))
        
        # Pixelwise Xor
        if all([s.inMatrix.grid.nCells == 2 for s in candTask.trainSamples]) \
        and all([s.inMatrix.grid.nCells == 2 for s in candTask.testSamples]):
            for c in candTask.commonOutColors:
                x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=c))
            if len(candTask.commonOutColors - candTask.commonInColors)==2 and\
            candTask.backgroundColor!=-1:
                colors = candTask.commonOutColors - candTask.commonInColors
                for c in permutations(colors, 2):
                    x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=candTask.backgroundColor,\
                                     firstTrue=c[0], secondTrue=c[1]))
            if len(candTask.totalOutColors) == 2:
                for target in candTask.totalInColors:
                    for c in permutations(candTask.totalOutColors, 2):
                        x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=c[0],\
                                         targetColor=target, trueColor=c[1]))
                      
    # Cropshape
    if candTask.outSmallerThanIn:
        bestCrop = getBestCropShape(candTask)
        if 'attributes' in bestCrop.keywords.keys():
            for attr in bestCrop.keywords['attributes']:
                newCrop = bestCrop
                newCrop.keywords['attributes'] = set([attr])
                x.append(newCrop)
                
        #this next part is artificial
        if len(candTask.commonInDShapes) > 0:
                x.append(partial(cropShapeReference, referenceShape=candTask.commonInDShapes, diagonal=True))
        if len(candTask.commonInShapes) > 0:
                x.append(partial(cropShapeReference, referenceShape=candTask.commonInShapes, diagonal=False))
        for attrs in [set(['LaSh'])]:
                x.append(partial(cropShape, attributes=attrs, backgroundColor=0, singleColor=True, diagonals=True)) 
    
    x.append(partial(cropAllBackground))
    """
    if all([len(s.inMatrix.multicolorShapes)==1 for s in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropOnlyMulticolorShape, diagonals=False))
    if all([len(s.inMatrix.multicolorDShapes)==1 for s in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropOnlyMulticolorShape, diagonals=True))
    """
    if all([len(sample.inMatrix.fullFrames)==1 for sample in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropFullFrame))
        x.append(partial(cropFullFrame, includeBorder=True))
    if all([len(sample.inMatrix.fullFrames)>1 for sample in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropFullFrame, bigOrSmall="big"))
        x.append(partial(cropFullFrame, bigOrSmall="small"))
        x.append(partial(cropFullFrame, bigOrSmall="big", includeBorder=True))
        x.append(partial(cropFullFrame, bigOrSmall="small", includeBorder=True))
    
        
    return x
