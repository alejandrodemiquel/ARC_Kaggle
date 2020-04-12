import numpy as np
import operator
import copy
import Models
import Task
import torch
import torch.nn as nn
from itertools import product, permutations
from functools import partial
from collections import Counter

def identityM(matrix):
    """
    Function that, given Matrix, returns its corresponding numpy.ndarray m
    """
    return matrix.m

def correctUnchangedColors(inMatrix, x, unchangedColors):
    """
    Given an input matrix (inMatrix), an output matrix (x) and a set of colors
    that should not change between the input and the output (unchangedColors),
    this function returns a copy of x, but correcting the pixels that 
    shouldn't have changed back into the original, unchanged color.
    
    inMatrix and x are required to have the same shape.
    """
    m = x.copy()
    for i,j in np.ndindex(m.shape):
        if inMatrix[i,j] in unchangedColors:
            m[i,j] = inMatrix[i,j]
    return m
                
def incorrectPixels(m1, m2):
    """
    Returns the number of incorrect pixels (0 is best)
    """
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

def dummify(x, nChannels, rel):
    """
    Given a matrix and a relationship given by relDicts, this function returns
    a nColors x shape(x) matrix consisting only of ones and zeros. For each
    channel (corresponding to a color), each element will be 1 if in the 
    original matrix x that pixel is of the corresponding color.
    """
    img = np.full((nChannels, x.shape[0], x.shape[1]), 0, dtype=np.uint8)
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

# %% Symmetrize

# 7/800 solved
# if t.lrSymmetric or t.udSymmetric or t.d1Symmetric:
# if len(t.changingColors) == 1:
def Symmetrize(task, color):
    """
    Given a task and a color, this function tries to turn pixels of that color
    into some other one in order to make the matrices symmetric.
    """
    # Left-Right
    def LRSymmetrize(m, color):
        width = m.shape[1] - 1
        for i in range(m.shape[0]):
            for j in range(int(m.shape[1] / 2)):
                if m[i,j] != m[i,width-j]:
                    if m[i,j] == color:
                        m[i,j] = m[i,width-j]
                    else:
                        m[i,width-j] = m[i,j]
        return m
    
    # Up-Down
    def UDSymmetrize(m, color):
        height = m.shape[0] - 1
        for i in range(int(m.shape[0] / 2)):
            for j in range(m.shape[1]):
                if m[i,j] != m[height-i,j]:
                    if m[i,j] == color:
                        m[i,j] = m[height-i,j]
                    else:
                        m[height-i,j] = m[i,j]
        return m

    # Main diagonal
    def D1Symmetrize(m, color):
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i,j] != m[j,i]:
                    if m[i,j] == color:
                        m[i,j] = m[j,i]
                    else:
                        m[j,i] = m[i,j]
        return m
    
    # TODO
    #def D2Symmetrize(matrix, color):
    
    ret = []
    for s in task.testSamples:
        newMatrix = s.inMatrix.m
        while True:
            prevMatrix = newMatrix
            if task.lrSymmetric:
                newMatrix = LRSymmetrize(newMatrix, color)
            if task.udSymmetric:
                newMatrix = UDSymmetrize(newMatrix, color)
            if task.d1Symmetric:
                newMatrix = D1Symmetrize(newMatrix, color)
            if np.all(newMatrix == prevMatrix):
                break
        ret.append(newMatrix)
    return ret

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
    return ret

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
    return ret

# %% Other utility functions

def insertShape(matrix, shape):
    """
    Given a matrix (numpy.ndarray) and a Task.Shape, this function returns the
    same matrix but with the shape inserted.
    """
    m = matrix.copy()
    for c in shape.pixels:
        if c[0] < matrix.shape[0] and c[1] < matrix.shape[1]:
            m[tuple(map(operator.add, c, shape.position))] = shape.color
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
    rangeV = range(m.shape[1])
    rangeH = range(m.shape[0])
    
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
    
def moveAllShapes(matrix, color, background, direction, until, nSteps):
    """
    direction can be l, r, u, d, ul, ur, dl, dr, h, v, d1, d2, all, any
    """
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
    
def moveShapeToClosest(matrix, shape, background, until):
    """
    Given a matrix (numpy.ndarray) and a Task.Shape, this function moves the
    given shape until the colosest shape with the color given by "until".
    """
    m = matrix.copy()
    s = copy.deepcopy(shape)
    m = deleteShape(m, shape, background)
    if until not in m:
        return matrix
    nSteps = 0
    while True:
        for c in s.pixels:
            pixelPos = tuple(map(operator.add, c, s.position))
            if nSteps <= pixelPos[0] and m[pixelPos[0]-nSteps, pixelPos[1]] == until:
                s.position = (s.position[0]-nSteps+1, s.position[1])
                return insertShape(m, s)
            if pixelPos[0]+nSteps < m.shape[0] and m[pixelPos[0]+nSteps, pixelPos[1]] == until:
                s.position = (s.position[0]+nSteps-1, s.position[1])
                return insertShape(m, s)
            if nSteps <= pixelPos[1] and m[pixelPos[0], pixelPos[1]-nSteps] == until:
                s.position = (s.position[0], s.position[1]-nSteps+1)
                return insertShape(m, s)
            if pixelPos[1]+nSteps < m.shape[1] and m[pixelPos[0], pixelPos[1]+nSteps] == until:
                s.position = (s.position[0], s.position[1]+nSteps-1)
                return insertShape(m, s)
        nSteps += 1
        if nSteps > m.shape[0] and nSteps > m.shape[1]:
            return matrix
        
def moveAllShapesToClosest(matrix, colorToMove, background, until):
    """
    This function moves all the shapes with color "colorToMove" until the
    closest shape with color "until".
    """
    m = matrix.m.copy()
    for s in matrix.shapes:
        if s.color == colorToMove:
            m = moveShapeToClosest(m, s, background, until)
    return m

def connectPixels(matrix, pixelColor=None, connColor=None, unchangedColors=set()):
    """
    Given a matrix, this function connects all the pixels that have the same
    color. This means that, for example, all the pixels between two red pixels
    will also be red.
    If "pixelColor" is specified, then only the pixels with the specified color
    will be connected.
    Ìf "connColor" is specified, the color used to connect the pixels will be
    the one given by this parameter.
    If there are any colors in the "unchangedColors" set, then they it is made
    sure that they remain unchanged.
    """
    m = matrix.copy()
    # Row
    for i in range(m.shape[0]):
        lowLimit = 0
        while lowLimit < m.shape[1] and m[i, lowLimit] != pixelColor:
            lowLimit += 1
        upLimit = m.shape[1]-1
        while upLimit > lowLimit and m[i, upLimit] != pixelColor:
            upLimit -= 1
        if upLimit > lowLimit:
            for j in range(lowLimit, upLimit):
                if m[i,j] != pixelColor and m[i,j] not in unchangedColors:
                    m[i,j] = connColor
       
    # Column             
    for j in range(m.shape[1]):
        lowLimit = 0
        while lowLimit < m.shape[0] and m[lowLimit, j] != pixelColor:
            lowLimit += 1
        upLimit = m.shape[0]-1
        while upLimit > lowLimit and m[upLimit, j] != pixelColor:
            upLimit -= 1
        if upLimit > lowLimit:
            for i in range(lowLimit, upLimit):
                if m[i,j] != pixelColor and m[i,j] not in unchangedColors:
                    m[i,j] = connColor
 
    return m

def connectAnyPixels(matrix, pixelColor=None, connColor=None, unchangedColors=set()):
    m = matrix.m.copy()
    if pixelColor==None and connColor==None:
        for c in matrix.colors - set([matrix.backgroundColor]):
            m = connectPixels(m, c, c)
        return m
    else:
        m = connectPixels(m, pixelColor, connColor, unchangedColors)
    return m

def rotate(matrix, angle):
    """
    Angle can be 90, 180, 270
    """
    assert angle in [90, 180, 270], "Invalid rotation angle"
    m = matrix.m.copy()
    return np.rot90(m, int(angle/90))    
    
def mirror(matrix, axis):
    """
    Axis can be lr, up, d1, d2
    """
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

def pixelwiseOr(matrices, falseColor, targetColor=None, trueColor=None):
    """
    See pixelwiseAnd.
    """
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

def pixelwiseXor(m1, m2, falseColor, targetColor=None, trueColor=None):
    """
    See pixelwiseAnd. The difference is that the Xor operation only makes sense
    with two input matrices.
    """
    m = np.zeros(m1.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            if (m1[i,j] == falseColor) != (m2[i,j] == falseColor):
                if trueColor == None:
                    if m1[i,j] != falseColor:
                        m[i,j] = m1[i,j]
                    else:
                        m[i,j] = m2[i,j]
                else:
                    m[i,j] = trueColor     
            else:
                m[i,j] = falseColor
        else:
            if (m1[i,j] == targetColor) != (m2[i,j] == targetColor):
                if trueColor == None:
                    if m1[i,j] != targetColor:
                        m[i,j] = m1[i,j]
                    else:
                        m[i,j] = m2[i,j]
                else:
                    m[i,j] = trueColor     
            else:
                m[i,j] = falseColor
    return m

# %% Operations to extend matrices

def multiplyPixels(matrix, factor):
    """
    Factor is a 2-dimensional tuple.
    The output matrix has shape matrix.shape*factor. Each pixel of the input
    matrix is expanded by factor.
    """
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
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i,j in np.ndindex(factor):
        m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = matrix.m
    return m

def multiplyPixelsAndAnd(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseAnd with these two
    matrices as inputs
    """
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
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseXor(m, newM, falseColor)
    return multipliedM

# %% Operations considering all submatrices of task with outShapeFactor
    
def getSubmatrices(m, factor):
    matrices = []
    nRows = int(m.shape[0] / factor[0])
    nCols = int(m.shape[1] / factor[1])
    for i,j in np.ndindex(factor):
        matrices.append(m[i*nRows:(i+1)*nRows, j*nCols:(j+1)*nCols])
    return matrices
    
def pixelwiseAndInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseAnd(matrices, falseColor, targetColor, trueColor)

def pixelwiseOrInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseOr(matrices, falseColor, targetColor, trueColor)

def pixelwiseXorInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseXor(matrices[0], matrices[1], falseColor, targetColor, trueColor)

# %% Operations considering all submatrices of a grid

def pixelwiseAndInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None):
    matrices = [c[0].m for c in matrix.grid.cellList]
    return pixelwiseAnd(matrices, falseColor, targetColor, trueColor)

def pixelwiseOrInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None):
    matrices = [c[0].m for c in matrix.grid.cellList]
    return pixelwiseOr(matrices, falseColor, targetColor, trueColor)

def pixelwiseXorInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None):
    m1 = matrix.grid.cellList[0][0].m.copy()
    m2 = matrix.grid.cellList[1][0].m.copy()
    return pixelwiseXor(m1, m2, falseColor, targetColor, trueColor)

# %% Stuff added by Roderic

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

def crop_shape(matrix, attributes, diagonal):
    """
    This function crops the shape out of a matrix with the maximum score according to attributes
    """
    if diagonal:
        shapes = [sh for sh in matrix.dShapes]
    else:
        shapes = [sh for sh in matrix.shapes]
    bestShape = []
    score = 0
    for sh in shapes:
        shscore = len(attributes.intersection(attribute_list(sh, matrix, diagonal)))
        if shscore > score:
            score = shscore
            bestShape = [sh]
        elif shscore == score:
            bestShape += [sh]
    return bestShape[0]
    
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
    directions = ['l', 'r', 'u', 'd', 'ul', 'ur', 'dl', 'dr', 'any']
    
    ###########################################################################
    # Fill the blanks
    if t.fillTheBlank:
        params = fillTheBlankParameters(t)
        x.append(partial(fillTheBlank, params=params))
    
    ###########################################################################
    # sameIOShapes
    if candTask.sameIOShapes:
        
        if all([n==2 for n in candTask.nInColors]):
            x.append(partial(switchColors))
        #######################################################################
        # ColorMap
        ncc = len(candTask.colorChanges)
        if len(set([cc[0] for cc in candTask.colorChanges])) == ncc and ncc != 0:
            x.append(partial(colorMap, cMap=dict(candTask.colorChanges)))
            
        #######################################################################
        # For LinearShapeModel we need to have the same shapes in the input
        # and in the output, and in the exact same positions.
        # This model predicts the color of the shape in the output.
        
        if candTask.onlyShapeColorChanges:
            """
            if all(["predictLinearModelShapeColor" not in str(op.func) for op in c.ops]):
                model = trainLinearModelShapeColor(candTask)
                x.append(partial(predictLinearModelShapeColor, model=model,\
                                 colors=set.union(*candTask.changedInColors+candTask.changedOutColors), \
                                 unchangedColors=candTask.unchangedColors, \
                                 shapePixelNumbers=candTask.shapePixelNumbers))
            """
                
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
        
        x.append(getBestCNN(candTask))
        if candTask.sameNSampleColors:
            x.append(getBestSameNSampleColorsCNN(candTask))

        if t.backgroundColor != -1:
            model = trainCNNDummyColor(candTask, 5, -1)
            x.append(partial(predictCNNDummyColor, model=model))
            model = trainCNNDummyColor(candTask, 3, 0)
            x.append(partial(predictCNNDummyColor, model=model))
            #model = trainOneConvModelDummyColor(candTask, 7, -1)
            #x.append(partial(predictConvModelDummyColor, model=model))
            
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
                
            # Move shapes
            colorsToChange = list(candTask.colors - candTask.unchangedColors -\
                                  set({candTask.backgroundColor}))
            ctc = [[c] for c in colorsToChange] + [colorsToChange] # Also all colors
            for c in ctc:
                for d in directions:
                    moveUntil = colorsToChange + [-1] + [-2] #Border, any
                    #for u in t.colors - set(c) | set({-1}):
                    for u in moveUntil:
                        x.append(partial(moveAllShapes, color=c,\
                                         background=candTask.backgroundColor,\
                                         direction=d, until=u, nSteps=100))
                    #for ns in [1, 2, 3, 4]:
                    #    x.append(partial(moveAllShapes, color=c,\
                                         #background=t.backgroundColor,\
                                         #direction=d, until=-1, nSteps=ns))
            
            if candTask.backgroundColor != -1 and hasattr(candTask, 'unchangedColors'):
                colorsToMove = set(range(10)) - set([candTask.backgroundColor]) -\
                candTask.unchangedColors
                for ctm in colorsToMove:
                    for uc in candTask.unchangedColors:
                        x.append(partial(moveAllShapesToClosest, colorToMove=ctm,\
                                                        background=candTask.backgroundColor,\
                                                        until=uc))
                                                         
            # Mirror shapes
            """
            for c in ctc:
                for d in ["lr", "ud"]:
                    x.append(partial(flipAllShapes, axis=d, color=c, \
                                     background=t.backgroundColor))
            """
                    
        #######################################################################
        # Other sameIOShapes functions
        x.append(partial(connectAnyPixels))
        if hasattr(candTask, "unchangedColors"):
            uc = candTask.unchangedColors
        else:
            uc = set()
        if hasattr(t, "unchangedColors"):
            tuc = candTask.unchangedColors
        else:
            tuc = set()
        for pc in candTask.colors - candTask.commonChangedInColors:
            for cc in candTask.colors - tuc:
                x.append(partial(connectAnyPixels, pixelColor=pc, \
                                 connColor=cc, unchangedColors=uc))
        for cc in candTask.commonColorChanges:
            for cc in candTask.commonColorChanges:
                for border, bs in product([True, False, None], ["big", "small", None]):
                    x.append(partial(changeShapes, inColor=cc[0], outColor=cc[1],\
                                     bigOrSmall=bs, isBorder=border))
                
    ###########################################################################
    # Cases in which the input has always the same shape, and the output too
    if candTask.sameInShape and candTask.sameOutShape and \
    all(candTask.trainSamples[0].inMatrix.shape == s.inMatrix.shape for s in candTask.testSamples):
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
            
            #Cases where output colors are a subset of input colors
            if candTask.sameNInColors and all(s.inHasOutColors for s in candTask.trainSamples):
                if hasattr(candTask, 'outShapeFactor') or (hasattr(candTask,\
                              'gridCellIsOutputShape') and candTask.gridCellIsOutputShape):
                    ch = dict(sum([Counter(s.outMatrix.colorCount) for s in candTask.trainSamples],Counter()))
                    ch = sorted(ch, key=ch.get)
                    if t.backgroundColor in ch:
                        ch.remove(t.backgroundColor)
                    ch = [t.backgroundColor] + ch
                    if hasattr(candTask, 'outShapeFactor'):
                        x.append(partial(overlapSubmatrices, colorHierarchy=ch, shapeFactor=candTask.outShapeFactor))
                    else:
                        x.append(partial(overlapSubmatrices, colorHierarchy=ch))
        
        pixelMap = Models.pixelCorrespondence(candTask)
        if len(pixelMap) != 0:
            x.append(partial(mapPixels, pixelMap=pixelMap, outShape=candTask.outShape))
                
    ###########################################################################
    # Other cases
    
    if hasattr(candTask, 'inShapeFactor'):
        x.append(partial(multiplyPixels, factor=candTask.inShapeFactor))
        x.append(partial(multiplyMatrix, factor=candTask.inShapeFactor))
        
        for c in candTask.commonSampleColors:
            x.append(partial(multiplyPixelsAndAnd, factor=candTask.inShapeFactor,\
                             falseColor=c))
            
    if hasattr(candTask, 'outShapeFactor'):
        # TODO Select a submatrix following certain criteria
        
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
        
        # Pixelwise Xor
        if candTask.outShapeFactor in [(2,1), (1,2)]:
            for c in candTask.commonOutColors:
                x.append(partial(pixelwiseXorInSubmatrices, factor=candTask.outShapeFactor,\
                                 falseColor=c))
            if len(candTask.totalOutColors) == 2:
                for target in candTask.totalInColors:
                    for c in permutations(candTask.totalOutColors, 2):
                        x.append(partial(pixelwiseXorInSubmatrices, \
                                         factor=candTask.outShapeFactor, falseColor=c[0],\
                                         targetColor=target, trueColor=c[1]))
    
    if hasattr(candTask, 'gridCellIsOutputShape') and candTask.gridCellIsOutputShape:
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
        
        # Pixelwise Xor
        if all([s.inMatrix.grid.nCells == 2 for s in candTask.trainSamples]) \
        and all([s.inMatrix.grid.nCells == 2 for s in candTask.testSamples]):
            for c in candTask.commonOutColors:
                x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=c))
            if len(candTask.totalOutColors) == 2:
                for target in candTask.totalInColors:
                    for c in permutations(candTask.totalOutColors, 2):
                        x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=c[0],\
                                         targetColor=target, trueColor=c[1]))
            
    return x