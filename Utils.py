import numpy as np
import operator
import copy
import Models
import torch
import torch.nn as nn
from itertools import product, permutations
from functools import partial

def correctUnchangedColors(inMatrix, x, unchangedColors):
    m = x.copy()
    for i,j in np.ndindex(m.shape):
        if inMatrix[i,j] in unchangedColors:
            m[i,j] = inMatrix[i,j]
    return m
                
def correctCells(m1, m2):
    """
    Returns the number of incorrect cells (0 is best)
    """
    return np.sum(m1!=m2)

def deBackgroundize(t):
    inMatrix = []
    outMatrix = []
    bColor = t.backgroundColor
    for s in t.trainSamples:
        inMatrix.append(np.uint8(s.inMatrix.m == bColor))
        outMatrix.append(np.uint8(s.outMatrix.m == bColor))
    return inMatrix, outMatrix

def deBackgroundizeMatrix(m, color):
    return np.uint8(m == color)

def relDicts(colors):
    rel = {}
    for i in range(len(colors)):
        rel[i] = colors[i]
    invRel = {v: k for k,v in rel.items()}
    for i in range(len(colors)):
        rel[i] = [colors[i]]
    return rel, invRel

def dummify(x, nChannels, rel):
    img = np.full((nChannels, x.shape[0], x.shape[1]), 0, dtype=np.uint8)
    for i in range(len(rel)):
        img[i] = np.isin(x,rel[i])
    return img

def dummifyColor(x, color):
    img = np.full((2, x.shape[0], x.shape[1]), 0, dtype=np.uint8)
    img[0] = x!=color
    img[1] = x==color
    return img

# 7/800 solved
# if t.lrSymmetric or t.udSymmetric or t.d1Symmetric:
# if len(t.changingColors) == 1:
def Symmetrize(task, color):
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
    m = matrix.m.copy()
    pred = np.ones(m.shape) * matrix.backgroundColor
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
    commonColors are the colors to be dummified for all samples
    """
    model = Models.OneConvModel(nChannels, k, pad)
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
            y = torch.tensor(y).unsqueeze(0).long()
            y_pred = model(x)
            loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

@torch.no_grad()
def predictCNN(matrix, model, commonColors, nChannels):
    """
    Given that the number of colors in a sample is always the same (nColors),
    This function executes a convolution. The first channels correspond to the
    colors that are common to every sample.
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
    This function returns the best CNN with only one convolution.
    There are as many channels as total colors or the minimum number of
    channels that is necessary.
    """
    kernel = [3,5]
    pad = [0,-1]    
    bestScore = 1000
    for k, p in product(kernel, pad):
        """
        Forget the simple convolution for now. It's messing me task 3
        cc = list(t.colors)
        nc = len(cc)
        model = trainCNN(t, commonColors=cc, nChannels=nc, k=k, pad=p)
        score = sum([correctCells(predictCNN(t.trainSamples[s].inMatrix, model, cc, nc), \
                                  t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        if score < bestScore:
            ret = partial(predictCNN, model=model, commonColors=cc, nChannels=nc)
        """
        
        if t.sameNSampleColors:
            cc = list(t.commonSampleColors)
            nc = t.trainSamples[0].nColors
            model = trainCNN(t, commonColors=cc, nChannels=nc, k=k, pad=p)
            score = sum([correctCells(predictCNN(t.trainSamples[s].inMatrix, model, cc, nc), \
                                      t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
            if score < bestScore:
                bestScore=score
                ret = partial(predictCNN, model=model, commonColors=cc, nChannels=nc)    
    return ret

# If input always has the same shape and output always has the same shape
# And there is always the same number of colors in each sample    
def trainLinearModel(t, commonColors, nChannels):
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
    Given that the number of colors in a sample is always the same (nColors),
    This function executes a convolution. The first channels correspond to the
    colors that are common to every sample.
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

def trainLinearShapeModel(t):
    """
    For LinearShapeModel we need to have the same shapes in the input
    and in the output, and in the exact same positions.
    This model predicts the color of the shape in the output.
    """
    rel, invRel = relDicts(list(t.colors))
    nInFeatures = t.nColors + 5
    model = Models.SimpleLinearModel(nInFeatures, t.nColors)
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
                inFeatures[invRel[s.color]] = 1
                inFeatures[t.nColors] = s.nCells
                inFeatures[t.nColors+1] = int(s.isSquare)
                inFeatures[t.nColors+2] = int(s.isRectangle)
                inFeatures[t.nColors+3] = s.isBorder
                inFeatures[t.nColors+4] = s.nHoles
                #inFeatures[t.nColors+5] = s.position[0].item()
                #inFeatures[t.nColors+6] = s.position[1].item()
                y = torch.tensor(invRel[label]).unsqueeze(0).long()
                x = inFeatures.unsqueeze(0).float()
                y_pred = model(x)
                loss += criterion(y_pred, y)
            loss.backward()
            optimizer.step()
    return model

@torch.no_grad()
def predictLinearShapeModel(matrix, model, colors):
    rel, invRel = relDicts(colors) # list(t.colors)
    nColors = len(colors)
    nInFeatures = nColors + 5
    pred = matrix.m.copy()
    for shape in matrix.shapes:
        if shape.color in colors:
            inFeatures = torch.zeros(nInFeatures)
            inFeatures[invRel[shape.color]] = 1
            inFeatures[nColors] = shape.nCells
            inFeatures[nColors+1] = int(shape.isSquare)
            inFeatures[nColors+2] = int(shape.isRectangle)
            inFeatures[nColors+3] = shape.isBorder
            inFeatures[nColors+4] = shape.nHoles
            #inFeatures[nColors+5] = shape.position[0].item()
            #inFeatures[nColors+6] = shape.position[1].item()
            x = inFeatures.unsqueeze(0).float()
            y = model(x).squeeze().argmax().item()
            pred = changeColorShape(pred, shape, rel[y][0])
    return pred

# %% Other utility functions

def insertShape(matrix, shape):
    m = matrix.copy()
    for c in shape.cells:
        if c[0] < matrix.shape[0] and c[1] < matrix.shape[1]:
            m[tuple(map(operator.add, c, shape.position))] = shape.color
    return m

def deleteShape(matrix, shape, backgroundColor):
    m = matrix.copy()
    for c in shape.cells:
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

def changeColorShape(matrix, shape, color):
    """
    Returns matrix with shape changed of color
    """
    if shape == False:
        return matrix
    for c in shape.cells:
        matrix[tuple(map(operator.add, c, shape.position))] = color
    return matrix

def changeShape(m, inColor, outColor, bigOrSmall = False, isBorder = True):
    return changeColorShape(m.m.copy(), m.getShape(inColor, bigOrSmall, isBorder), outColor)

def changeShapeColorAll(m, inColor, outColor, isBorder=True, biggerThan=0, \
                        smallerThan=1000, diagonal=False):
    x = m.m.copy()
    if diagonal:
        shapesToChange = m.dShapes
    else:
        shapesToChange = m.shapes
    shapesToChange = [s for s in shapesToChange if s.isBorder == isBorder and \
                      s.color == inColor and s.nCells > biggerThan and \
                      s.nCells < smallerThan]
    for s in shapesToChange:
        x = changeColorShape(x, s, outColor)
    return x

# TODO
def surroundShape(matrix, shape, color, nSteps = False, untilColor = False):
    def addCell(i,j):
        if matrix[tuple(map(operator.add, c, shape.position))] == untilColor:
            return False
        else:
            matrix[tuple(map(operator.add, c, shape.position))] = color
            cells.add((i, j))

    x = matrix.copy()
    cells = shape.cells.copy()
    while True:
        y = x.copy()
        for c in shape.cells:
            addCell(c[0]+1, c[1])
            addCell(c[0]-1, c[1])
            addCell(c[0]+1, c[1]+1)
            addCell(c[0]+1, c[1]-1)
            addCell(c[0]-1, c[1]+1)
            addCell(c[0]-1, c[1]-1)
            addCell(c[0], c[1]+1)
            addCell(c[0], c[1]-1)
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
    m = changeColorShape(m, shape, background)
    s = copy.deepcopy(shape)
    step = 0
    while True and step != nSteps:
        step += 1
        for c in s.cells:
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
        shapesToMove.sort(key=lambda x: x.position[1]+x.yLen, reverse=True)
    if direction == 'u':
        shapesToMove.sort(key=lambda x: x.position[0])  
    if direction == 'd':
        shapesToMove.sort(key=lambda x: x.position[0]+x.xLen, reverse=True)
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
    m = matrix.copy()
    s = copy.deepcopy(shape)
    m = deleteShape(m, shape, background)
    if until not in m:
        return matrix
    nSteps = 0
    while True:
        for c in s.cells:
            cellPos = tuple(map(operator.add, c, s.position))
            if nSteps <= cellPos[0] and m[cellPos[0]-nSteps, cellPos[1]] == until:
                s.position = (s.position[0]-nSteps+1, s.position[1])
                return insertShape(m, s)
            if cellPos[0]+nSteps < m.shape[0] and m[cellPos[0]+nSteps, cellPos[1]] == until:
                s.position = (s.position[0]+nSteps-1, s.position[1])
                return insertShape(m, s)
            if nSteps <= cellPos[1] and m[cellPos[0], cellPos[1]-nSteps] == until:
                s.position = (s.position[0], s.position[1]-nSteps+1)
                return insertShape(m, s)
            if cellPos[1]+nSteps < m.shape[1] and m[cellPos[0], cellPos[1]+nSteps] == until:
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

def connectPixels(matrix, pixelColor, connColor, unchangedColors):
    m = matrix.m.copy()
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
    
def flipShape(matrix, shape, axis, background):
    """
    Axis can be lr, ud
    """
    m = matrix.copy()
    smallM = np.ones((shape.xLen+1, shape.yLen+1)) * background
    for c in shape.cells:
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

def mapPixels(matrix, pixelMap, outShape):
    inMatrix = matrix.m.copy()
    m = np.zeros(outShape)
    for i,j in np.ndindex(outShape):
        m[i,j] = inMatrix[pixelMap[i,j]]
    return m
    
# %% Operations with more than one matrix

# All the matrices need to have the same shape

def pixelwiseAnd(matrices, falseColor, targetColor=None, trueColor=None):
    """
    "matrices" is a list of matrices (numpy arrays) of the same shape.
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
    The output matrix has shape matrix.shape*factor
    """
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)))
    for i,j in np.ndindex(matrix.m.shape):
        for k,l in np.ndindex(factor):
            m[i*factor[0]+k, j*factor[1]+l] = matrix.m[i,j]
    return m

def multiplyMatrix(matrix, factor):
    """
    Copy the matrix "matrix" into every submatrix of the output, which has
    shape matrix.shape * factor.
    """
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)))
    for i,j in np.ndindex(factor):
        m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = matrix.m
    return m

# TODO Many things missing and wrong here
def multiplyPixelsAndAnd(matrix, factor, falseColor):
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseAnd([m, newM], falseColor)
    return multipliedM

# TODO multiplyPixelsAndOr
# TODO multiplyPixelsAndXor

# %% Operations considering all submatrices of a grid

def pixelwiseAndInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None):
    matrices = [c[0].m for c in matrix.grid.cells]
    return pixelwiseAnd(matrices, falseColor, targetColor, trueColor)

def pixelwiseOrInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None):
    matrices = [c[0].m for c in matrix.grid.cells]
    return pixelwiseOr(matrices, falseColor, targetColor, trueColor)

def pixelwiseXorInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None):
    m1 = matrix.grid.cells[0][0].m.copy()
    m2 = matrix.grid.cells[1][0].m.copy()
    return pixelwiseXor(m1, m2, falseColor, targetColor, trueColor)


# %% Main function: getPossibleOperations
def getPossibleOperations(t, c):
    """
    Returns a list of all possible operations to be performed on the task.
    """ 
    candTask = c.t
    x = [] # List to be returned
    directions = ['l', 'r', 'u', 'd', 'ul', 'ur', 'dl', 'dr', 'any']
    """
    ###########################################################################
    # sameIOShapes
    if candTask.sameIOShapes:
        #######################################################################
        # ColorMap
        ncc = len(candTask.colorChanges)
        if len(set([cc[0] for cc in candTask.colorChanges])) == ncc and\
        len(set([cc[1] for cc in candTask.colorChanges])) == ncc and ncc != 0:
            x.append(partial(colorMap, cMap=dict(candTask.colorChanges)))
            
        #######################################################################
        # For LinearShapeModel we need to have the same shapes in the input
        # and in the output, and in the exact same positions.
        # This model predicts the color of the shape in the output.
        isGood = True
        for s in candTask.trainSamples:
            nShapes = s.inMatrix.nShapes
            if s.outMatrix.nShapes != nShapes:
                isGood = False
                break
            for shapeI in range(nShapes):
                if not s.inMatrix.shapes[shapeI].hasSameShape(s.outMatrix.shapes[shapeI]):
                    isGood = False
                    break
            if not isGood:
                break
        if isGood:
            if all(["predictLinearShapeModel" not in str(op.func) for op in c.ops]):
                model = trainLinearShapeModel(candTask)
                x.append(partial(predictLinearShapeModel, model=model,\
                                 colors=list(candTask.colors)))
            
            # Other deterministic functions that change the color of shapes.
            for cc in candTask.commonColorChanges:
                x.append(partial(changeShapeColorAll, inColor=cc[0], outColor=cc[1],\
                                 isBorder=False))
                x.append(partial(changeShapeColorAll, inColor=cc[0], outColor=cc[1],\
                                 isBorder=True))
                for bs in ["big", "small"]:
                    x.append(partial(changeShape, inColor=cc[0], outColor=cc[1],\
                                     bigOrSmall=bs, isBorder=False))
                    x.append(partial(changeShape, inColor = cc[0], outColor = cc[1],\
                                     bigOrSmall=bs, isBorder=True))
            
            return x

        #######################################################################
        # CNNs
        
        # TODO Delete the if once issue with task 3 is solved
        if candTask.sameNSampleColors:
            x.append(getBestCNN(candTask))

        if t.backgroundColor != -1:
            model = trainCNNDummyColor(candTask, 5, -1)
            x.append(partial(predictCNNDummyColor, model=model))
            #model = trainOneConvModelDummyColor(candTask, 3, 0)
            #x.append(partial(predictConvModelDummyColor, model=model))
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
            for c in ctc:
                for d in ["lr", "ud"]:
                    x.append(partial(flipAllShapes, axis=d, color=c, \
                                     background=t.backgroundColor))
                    
        #######################################################################
        # Other sameIOShapes functions
        
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
                x.append(partial(connectPixels, pixelColor=pc, \
                                 connColor=cc, unchangedColors=uc))
        
        for cc in candTask.commonColorChanges:
            x.append(partial(changeShapeColorAll, inColor=cc[0], outColor=cc[1],\
                             isBorder=False))
            x.append(partial(changeShapeColorAll, inColor=cc[0], outColor=cc[1],\
                             isBorder=True))
            ""
            x.append(partial(changeShapeColorAll, inColor=cc[0], outColor=cc[1],\
                             biggerThan=1))
            x.append(partial(changeShapeColorAll, inColor=cc[0], outColor=cc[1],\
                             biggerThan=2))
            x.append(partial(changeShapeColorAll, inColor=cc[0], outColor=cc[1],\
                             biggerThan=3))
            x.append(partial(changeShapeColorAll, inColor=cc[0], outColor=cc[1],\
                             smallerThan=5))
            ""
            for bs in ["big", "small"]:
                x.append(partial(changeShape, inColor=cc[0], outColor=cc[1],\
                                 bigOrSmall=bs, isBorder=False))
                x.append(partial(changeShape, inColor = cc[0], outColor = cc[1],\
                                 bigOrSmall=bs, isBorder=True))
                
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
        
        pixelMap = Models.pixelCorrespondence(t)
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
    """
            
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