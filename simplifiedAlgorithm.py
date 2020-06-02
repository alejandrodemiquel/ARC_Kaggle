

# %% Setup

import numpy as np # linear algebra
import pandas as pd
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from collections import Counter
import copy
from itertools import product, permutations, combinations, combinations_with_replacement
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
#data_path = Path('data')
train_path = data_path / 'training'
eval_path = data_path / 'evaluation'
test_path = data_path / 'test'

train_tasks = { task.stem: json.load(task.open()) for task in train_path.iterdir() } 
eval_tasks = { task.stem: json.load(task.open()) for task in eval_path.iterdir() }
test_tasks = { task.stem: json.load(task.open()) for task in test_path.iterdir() }

data = test_tasks

cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def plot_pictures(pictures, labels):
    fig, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures),32))
    for i, (pict, label) in enumerate(zip(pictures, labels)):
        axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
        axs[i].set_title(label)
    plt.show()

def plot_sample(sample, predict=None):
    """
    This function plots a sample. sample is an object of the class Task.Sample.
    predict is any matrix (numpy ndarray).
    """
    if predict is None:
        plot_pictures([sample.inMatrix.m, sample.outMatrix.m], ['Input', 'Output'])
    else:
        plot_pictures([sample.inMatrix.m, sample.outMatrix.m, predict], ['Input', 'Output', 'Predict'])

def plot_task(task):
    """
    Given a task (in its original format), this function plots all of its
    matrices.
    """
    len_train = len(task['train'])
    len_test  = len(task['test'])
    len_max   = max(len_train, len_test)
    length    = {'train': len_train, 'test': len_test}
    fig, axs  = plt.subplots(len_max, 4, figsize=(15, 15*len_max//4))
    for col, mode in enumerate(['train', 'test']):
        for idx in range(length[mode]):
            axs[idx][2*col+0].axis('off')
            axs[idx][2*col+0].imshow(task[mode][idx]['input'], cmap=cmap, norm=norm)
            axs[idx][2*col+0].set_title(f"Input {mode}, {np.array(task[mode][idx]['input']).shape}")
            try:
                axs[idx][2*col+1].axis('off')
                axs[idx][2*col+1].imshow(task[mode][idx]['output'], cmap=cmap, norm=norm)
                axs[idx][2*col+1].set_title(f"Output {mode}, {np.array(task[mode][idx]['output']).shape}")
            except:
                pass
        for idx in range(length[mode], len_max):
            axs[idx][2*col+0].axis('off')
            axs[idx][2*col+1].axis('off')
    plt.tight_layout()
    plt.axis('off')
    plt.show()

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

##############################################################################
# %% CORE OBJECTS

# %% Frontiers
class Frontier:
    """
    A Frontier is defined as a straight line with a single color that crosses
    all of the matrix. For example, if the matrix has shape MxN, then a
    Frontier will have shape Mx1 or 1xN. See the function "detectFrontiers"
    for details in the implementation.
    
    ...
    
    Attributes
    ----------
    color: int
        The color of the frontier
    directrion: str
        A character ('h' or 'v') determining whether the frontier is horizontal
        or vertical
    position: tuple
        A 2-tuple of ints determining the position of the upper-left pixel of
        the frontier
    """
    def __init__(self, color, direction, position):
        """
        direction can be 'h' or 'v' (horizontal, vertical)
        color, position and are all integers
        """
        self.color = color
        self.direction = direction
        self.position = position
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False
        
def detectFrontiers(m):
    """
    m is a numpy 2-dimensional matrix.
    """
    frontiers = []
    
    # Horizontal lines
    if m.shape[0]>1:
        for i in range(m.shape[0]):
            color = m[i, 0]
            isFrontier = True
            for j in range(m.shape[1]):
                if color != m[i,j]:
                    isFrontier = False
                    break
            if isFrontier:
                frontiers.append(Frontier(color, 'h', i))
            
    # Vertical lines
    if m.shape[1]>1:
        for j in range(m.shape[1]):
            color = m[0, j]
            isFrontier = True
            for i in range(m.shape[0]):
                if color != m[i,j]:
                    isFrontier = False
                    break
            if isFrontier:
                frontiers.append(Frontier(color, 'v', j))
            
    return frontiers

# %% Grids
class Grid:
    """
    An object of the class Grid is basically a collection of frontiers that
    have all the same color.
    It is useful to check, for example, whether the cells defined by the grid
    always have the same size or not.
    
    ...
    
    Attributes
    ----------
    color: int
        The color of the grid
    m: numpy.ndarray
        The whole matrix
    frontiers: list
        A list of all the frontiers the grid is composed of
    cells: list of list of 2-tuples
        cells can be viewed as a 2-dimensional matrix of 2-tuples (Matrix, 
        position). The first element is an object of the class Matrix, and the
        second element is the position of the cell in m.
        Each element represents a cell of the grid.
    shape: tuple
        A 2-tuple of ints representing the number of cells of the grid
    nCells: int
        Number of cells of the grid
    cellList: list
        A list of all the cells
    allCellsSameShape: bool
        Determines whether all the cells of the grid have the same shape (as
        matrices).
    cellShape: tuple
        Only defined if allCellsSameShape is True. Shape of the cells.
    allCellsHaveOneColor: bool
        Determines whether the ALL of the cells of the grid are composed of
        pixels of the same color
    """
    def __init__(self, m, frontiers):
        self.color = frontiers[0].color
        self.m = m
        self.frontiers = frontiers
        hPositions = [f.position for f in frontiers if f.direction == 'h']
        hPositions.append(-1)
        hPositions.append(m.shape[0])
        hPositions.sort()
        vPositions = [f.position for f in frontiers if f.direction == 'v']
        vPositions.append(-1)
        vPositions.append(m.shape[1])
        vPositions.sort()
        # cells is a matrix (list of lists) of 2-tuples (Matrix, position)
        self.cells = []
        hShape = 0
        vShape = 0
        for h in range(len(hPositions)-1):
            if hPositions[h]+1 == hPositions[h+1]:
                continue
            self.cells.append([])
            for v in range(len(vPositions)-1):
                if vPositions[v]+1 == vPositions[v+1]:
                    continue
                if hShape == 0:
                    vShape += 1
                self.cells[hShape].append((Matrix(m[hPositions[h]+1:hPositions[h+1], \
                                                   vPositions[v]+1:vPositions[v+1]], \
                                                 detectGrid=False), \
                                          (hPositions[h]+1, vPositions[v]+1)))
            hShape += 1
            
        self.shape = (hShape, vShape) # N of h cells x N of v cells
        self.cellList = []
        for cellRow in range(len(self.cells)):
            for cellCol in range(len(self.cells[0])):
                self.cellList.append(self.cells[cellRow][cellCol])
        self.allCellsSameShape = len(set([c[0].shape for c in self.cellList])) == 1
        if self.allCellsSameShape:
            self.cellShape = self.cells[0][0][0].shape
            
        self.nCells = len(self.cellList)
            
        # Check whether each cell has one and only one color
        self.allCellsHaveOneColor = True
        for c in self.cellList:
            if c[0].nColors!=1:
                self.allCellsHaveOneColor = False
                break
        
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([f in other.frontiers for f in self.frontiers])
        else:
            return False
        
# %% Frames
"""
class Frame:
    def __init__(self, matrix):
        self.m
        self.color
        self.position
        self.shape
        self.isFull
        
def detectFrames(matrix):
    frames = []
    m = matrix.m.copy()
    for i,j in np.ndindex(m.shape):
        color = m[i,j]
        iMax = m.shape[0]
        jMax = m.shape[1]
        for k in range(i+1, m.shape[0]):
            for l in range(j+1, m.shape[1]):
                if m[k,l]==color:
                    
        
    return frames
"""

# %% Shapes and subclasses
class Shape:
    def __init__(self, m, xPos, yPos, background, isBorder):
        # pixels is a 2xn numpy array, where n is the number of pixels
        self.m = m
        self.nPixels = m.size - np.count_nonzero(m==255)
        self.background = background
        self.shape = m.shape
        self.position = (xPos, yPos)
        self.pixels = set([(i,j) for i,j in np.ndindex(m.shape) if m[i,j]!=255])
            
        # Is the shape in the border?
        self.isBorder = isBorder
        
        # Which colors does the shape have?
        self.colors = set(np.unique(m)) - set([255])
        self.nColors = len(self.colors)
        if self.nColors==1:
            self.color = next(iter(self.colors))

        self.colorCount = Counter(self.m.flatten()) + Counter({0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0})
        del self.colorCount[255]

        # Symmetries
        self.lrSymmetric = np.array_equal(self.m, np.fliplr(self.m))
        self.udSymmetric = np.array_equal(self.m, np.flipud(self.m))
        if self.m.shape[0] == self.m.shape[1]:
            self.d1Symmetric = np.array_equal(self.m, self.m.T)
            self.d2Symmetric = np.array_equal(np.fliplr(self.m), (np.fliplr(self.m)).T)
        else:
            self.d1Symmetric = False
            self.d2Symmetric = False

        self.isRectangle = 255 not in np.unique(m)
        self.isSquare = self.isRectangle and self.shape[0]==self.shape[1]
        
        if self.isRectangle and self.nColors > 1:
            self.subshapes = detectShapes(self.m, background=self.colorCount.most_common(1)[0][0],\
                                              singleColor=True, diagonals=False)
        self.nHoles = self.getNHoles()
        
        if self.nColors==1:
            self.isFullFrame = self.isFullFrame()
            self.isPartialFrame = self.isPartialFrame()
        else:
            self.isFullFrame = False
            self.isPartialFrame = False
        
        if self.nColors==1:
            self.boolFeatures = []
            for c in range(10):
                self.boolFeatures.append(self.color==c)
            self.boolFeatures.append(self.isBorder)
            self.boolFeatures.append(not self.isBorder)
            self.boolFeatures.append(self.lrSymmetric)
            self.boolFeatures.append(self.udSymmetric)
            self.boolFeatures.append(self.d1Symmetric)
            self.boolFeatures.append(self.d2Symmetric)
            self.boolFeatures.append(self.isSquare)
            self.boolFeatures.append(self.isRectangle)
            for nPix in range(1,30):
                self.boolFeatures.append(self.nPixels==nPix)
            self.boolFeatures.append((self.nPixels%2)==0)
            self.boolFeatures.append((self.nPixels%2)==1)
    
    def hasSameShape(self, other, sameColor=False, samePosition=False, rotation=False, \
                     mirror=False, scaling=False):
        if samePosition:
            if self.position != other.position:
                return False
        if sameColor:
            m1 = self.m
            m2 = other.m
        else:
            m1 = self.shapeDummyMatrix()
            m2 = other.shapeDummyMatrix()
        if scaling and m1.shape!=m2.shape:
            def multiplyPixels(matrix, factor):
                m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
                for i,j in np.ndindex(matrix.shape):
                    for k,l in np.ndindex(factor):
                        m[i*factor[0]+k, j*factor[1]+l] = matrix[i,j]
                return m
            
            if (m1.shape[0]%m2.shape[0])==0 and (m1.shape[1]%m2.shape[1])==0:
                factor = (int(m1.shape[0]/m2.shape[0]), int(m1.shape[1]/m2.shape[1]))
                m2 = multiplyPixels(m2, factor)
            elif (m2.shape[0]%m1.shape[0])==0 and (m2.shape[1]%m1.shape[1])==0:
                factor = (int(m2.shape[0]/m1.shape[0]), int(m2.shape[1]/m1.shape[1]))
                m1 = multiplyPixels(m1, factor)
            elif rotation and (m1.shape[0]%m2.shape[1])==0 and (m1.shape[1]%m2.shape[0])==0:
                factor = (int(m1.shape[0]/m2.shape[1]), int(m1.shape[1]/m2.shape[0]))
                m2 = multiplyPixels(m2, factor)
            elif rotation and (m2.shape[0]%m1.shape[1])==0 and (m2.shape[1]%m1.shape[0])==0:
                factor = (int(m2.shape[0]/m1.shape[1]), int(m2.shape[1]/m1.shape[0]))
                m1 = multiplyPixels(m1, factor)
            else:
                return False
        if rotation and not mirror:
            if any([np.array_equal(m1, np.rot90(m2,x)) for x in range(1,4)]):
                return True
        if mirror and not rotation:
            if np.array_equal(m1, np.fliplr(m2)) or np.array_equal(m1, np.flipud(m2)):
                return True
        if mirror and rotation:
            for x in range(1, 4):
                if any([np.array_equal(m1, np.rot90(m2,x))\
                        or np.array_equal(m1, np.fliplr(np.rot90(m2,x))) for x in range(0,4)]):
                    return True               
                
        return np.array_equal(m1,m2)
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                return False
            return np.array_equal(self.m, other.m)
        else:
            return False

    """
    def __hash__(self):
        return self.m
    """    
    def isSubshape(self, other, sameColor=False, rotation=False, mirror=False):
        """
        The method checks if a shape fits inside another. Can take into account rotations and mirrors. 
        Maybe it should be updated to return the positions of subshapes instead of a boolean?
        """
        #return positions
        if rotation:
            m1 = self.m
            for x in range(1,4):
                if Shape(np.rot90(m1,x), 0, 0, 0, self.isBorder).isSubshape(other, sameColor, False, mirror):
                    return True
        if mirror == 'lr':
            if Shape(self.m[::,::-1], 0, 0, 0, self.isBorder).isSubshape(other, sameColor, rotation, False):
                return True
        if mirror == 'ud':
            if Shape(self.m[::-1,::], 0, 0, 0, self.isBorder).isSubshape(other, sameColor, rotation, False):
                return True
        if sameColor:
            if hasattr(self,'color') and hasattr(other,'color') and self.color != other.color:
                return False
        if any(other.shape[i] < self.shape[i] for i in [0,1]):
            return False
        
        for yIn in range(other.shape[1] - self.shape[1] + 1):
            for xIn in range(other.shape[0] - self.shape[0] + 1):
                if sameColor:
                    if np.all(np.logical_or((self.m == other.m[xIn: xIn + self.shape[0], yIn: yIn + self.shape[1]]),\
                                            self.m==255)):
                        return True
                else:
                    if set([tuple(np.add(ps,[xIn,yIn])) for ps in self.pixels]) <= other.pixels:
                        return True
        return False
    
    def shapeDummyMatrix(self):
        """
        Returns the smallest possible matrix containing the shape. The values
        of the matrix are ones and zeros, depending on whether the pixel is a
        shape pixel or not.
        """
        return (self.m!=255).astype(np.uint8) 
    
    def hasFeatures(self, features):
        for i in range(len(features)):
            if features[i] and not self.boolFeatures[i]:
                return False
        return True

    def getNHoles(self):
        nHoles = 0
        m = self.m
        seen = np.zeros((self.shape[0], self.shape[1]), dtype=np.bool)
        def isInHole(i,j):
            if i<0 or j<0 or i>self.shape[0]-1 or j>self.shape[1]-1:
                return False
            if seen[i,j] or m[i,j] != 255:
                return True
            seen[i,j] = True
            ret = isInHole(i+1,j)*isInHole(i-1,j)*isInHole(i,j+1)*isInHole(i,j-1)
            return ret
        for i,j in np.ndindex(m.shape):
            if m[i,j] == 255 and not seen[i,j]:
                if isInHole(i,j):
                    nHoles += 1
        return nHoles

    def isRotationInvariant(self, color=False):
        if color:
            m = np.rot90(self.m, 1)
            return np.array_equal(m, self.m)
        else:
            m2 = self.shapeDummyMatrix()
            m = np.rot90(m2, 1)
            return np.array_equal(m, m2)
        
    """
    def isFullFrame(self):
        if self.shape[0]<3 or self.shape[1]<3:
            return False
        for i in range(1, self.shape[0]-1):
            for j in range(1, self.shape[1]-1):
                if self.m[i,j] != 255:
                    return False
        if self.nPixels == 2 * (self.shape[0]+self.shape[1]-2):
            return True
        return False
    """
    def isPartialFrame(self):
        if self.shape[0] < 4 or self.shape[1] < 4 or len(self.pixels) < 4:
            return False
        if len(np.unique(self.m[1:-1,1:-1])) > 1 or self.color in np.unique(self.m[1:-1,1:-1]):
            return False
        return True
    
    def isFullFrame(self):
        if self.shape[0]<3 or self.shape[1]<3:
            return False
        for i in range(self.shape[0]):
            if self.m[i,0]==255 or self.m[i,self.shape[1]-1]==255:
                return False
        for j in range(self.shape[1]):
            if self.m[0,j]==255 or self.m[self.shape[0]-1,j]==255:
                return False
            
        # We require fullFrames to have less than 20% of the pixels inside the
        # frame of the same color of the frame
        
        if self.nPixels - 2*(self.shape[0]+self.shape[1]-2) < 0.2*(self.shape[0]-2)*(self.shape[1]-2):
            return True
        
        return False

def detectShapesByColor(x, background):
    shapes = []
    for c in range(10):
        if c == background or c not in x:
            continue
        mc = np.zeros(x.shape, dtype=int)
        mc[x==c] = c
        mc[x!=c] = 255
        x1, x2, y1, y2 = 0, mc.shape[0]-1, 0, mc.shape[1]-1
        while x1 <= x2 and np.all(mc[x1,:] == 255):
            x1 += 1 
        while x2 >= x1 and np.all(mc[x2,:] == 255):
            x2 -= 1
        while y1 <= y2 and np.all(mc[:,y1] == 255):
            y1 += 1
        while y2 >= y1 and np.all(mc[:,y2] == 255):
            y2 -= 1
        m = mc[x1:x2+1,y1:y2+1]
        s = Shape(m.copy(), x1, y1, background, False)
        shapes.append(s)
    return shapes

def detectShapes(x, background, singleColor=False, diagonals=False):
    """
    Given a numpy array x (2D), returns a list of the Shapes present in x
    """
    # Helper function to add pixels to a shape
    def addPixelsAround(i,j):
        def addPixel(i,j):
            if i < 0 or j < 0 or i > iMax or j > jMax or seen[i,j] == True:
                return
            if singleColor:
                if x[i,j] != color:
                    return
                newShape[i,j] = color
            else:
                if x[i,j] == background:
                    return
                newShape[i,j] = x[i,j]
            seen[i,j] = True                
            addPixelsAround(i,j)
        
        addPixel(i-1,j)
        addPixel(i+1,j)
        addPixel(i,j-1)
        addPixel(i,j+1)
        
        if diagonals:
            addPixel(i-1,j-1)
            addPixel(i-1,j+1)
            addPixel(i+1,j-1)
            addPixel(i+1,j+1)
            
    def crop(matrix):
        ret = matrix.copy()
        for k in range(x.shape[0]):
            if any(matrix[k,:] != 255): # -1==255 for dtype=np.uint8
                x0 = k
                break
        for k in reversed(range(x.shape[0])):
            if any(matrix[k,:] != 255): # -1==255 for dtype=np.uint8
                x1 = k
                break
        for k in range(x.shape[1]):
            if any(matrix[:,k] != 255): # -1==255 for dtype=np.uint8
                y0 = k
                break
        for k in reversed(range(x.shape[1])):
            if any(matrix[:,k] != 255): # -1==255 for dtype=np.uint8
                y1 = k
                break
        return ret[x0:x1+1,y0:y1+1], x0, y0
                
    shapes = []
    seen = np.zeros(x.shape, dtype=bool)
    iMax = x.shape[0]-1
    jMax = x.shape[1]-1
    for i, j in np.ndindex(x.shape):
        if seen[i,j] == False:
            seen[i,j] = True
            if not singleColor and x[i,j]==background:
                continue
            newShape = np.full((x.shape), -1, dtype=np.uint8)
            newShape[i,j] = x[i,j]
            if singleColor:
                color = x[i][j]
            addPixelsAround(i,j)
            m, xPos, yPos = crop(newShape)
            isBorder = xPos==0 or yPos==0 or (xPos+m.shape[0]==x.shape[0]) or (yPos+m.shape[1]==x.shape[1])
            s = Shape(m.copy(), xPos, yPos, background, isBorder)
            shapes.append(s)
    return shapes

def detectIsolatedPixels(matrix, dShapeList):
    pixList = []
    for sh in dShapeList:
        if sh.nPixels > 1 or sh.color == matrix.backgroundColor:
            continue
        else:
            cc = set()
            for i,j in np.ndindex(3, 3):
                if i - 1 + sh.position[0] < matrix.shape[0] and i - 1 + sh.position[0] >= 0 \
                        and j - 1 + sh.position[1] < matrix.shape[1] and j - 1 + sh.position[1] >= 0:
                    cc  = cc.union(set([matrix.m[i - 1 + sh.position[0],j - 1 + sh.position[1]]]))
            if len(cc) == 2:
                pixList.append(sh)
    return pixList

# %% Class Matrix
class Matrix():
    def __init__(self, m, detectGrid=True, backgroundColor=None):
        if type(m) == Matrix:
            return m
        
        self.m = np.array(m)
        
        # interesting properties:
        
        # Dimensions
        self.shape = self.m.shape
        self.nElements = self.m.size
        
        # Counter of colors
        self.colorCount = self.getColors()
        self.colors = set(self.colorCount.keys())
        self.nColors = len(self.colorCount)
        
        # Background color
        if backgroundColor==None:
            self.backgroundColor = max(self.colorCount, key=self.colorCount.get)
        else:
            self.backgroundColor = backgroundColor
        
        # Shapes
        self.shapes = detectShapes(self.m, self.backgroundColor, singleColor=True)
        self.nShapes = len(self.shapes)
        self.dShapes = detectShapes(self.m, self.backgroundColor, singleColor=True, diagonals=True)
        self.nDShapes = len(self.dShapes)
        self.fullFrames = [shape for shape in self.shapes if shape.isFullFrame]
        self.fullFrames = sorted(self.fullFrames, key=lambda x: x.shape[0]*x.shape[1], reverse=True)
        self.shapesByColor = detectShapesByColor(self.m, self.backgroundColor)
        self.partialFrames = [shape for shape in self.shapesByColor if shape.isPartialFrame]
        self.isolatedPixels = detectIsolatedPixels(self, self.dShapes)
        self.nIsolatedPixels = len(self.isolatedPixels)
        
        self.shapeColorCounter = Counter([s.color for s in self.shapes])
        self.blanks = []
        for s in self.shapes:
            if s.isRectangle and self.shapeColorCounter[s.color]==1:
                self.blanks.append(s)
            
        # Frontiers
        self.frontiers = detectFrontiers(self.m)
        self.frontierColors = [f.color for f in self.frontiers]
        if len(self.frontiers) == 0:
            self.allFrontiersEqualColor = False
        else: self.allFrontiersEqualColor = (self.frontierColors.count(self.frontiers[0]) ==\
                                         len(self.frontiers))
        # Check if it's a grid and the dimensions of the cells
        self.isGrid = False
        self.isAsymmetricGrid = False
        if detectGrid:
            for fc in set(self.frontierColors):
                possibleGrid = [f for f in self.frontiers if f.color==fc]
                possibleGrid = Grid(self.m, possibleGrid)
                if possibleGrid.nCells>1:
                    if possibleGrid.allCellsSameShape:
                        self.grid = copy.deepcopy(possibleGrid)
                        self.isGrid = True
                        self.asymmetricGrid = copy.deepcopy(possibleGrid)
                        self.isAsymmetricGrid = True
                        break
                    else:
                        self.asymmetricGrid = copy.deepcopy(possibleGrid)
                        self.isAsymmetricGrid=True
                        
        # Shape-based backgroundColor
        if not self.isGrid:
            for shape in self.shapes:
                if shape.shape==self.shape:
                    self.backgroundColor = shape.color
                    break
        # Define multicolor shapes based on the background color
        self.multicolorShapes = detectShapes(self.m, self.backgroundColor)
        self.multicolorDShapes = detectShapes(self.m, self.backgroundColor, diagonals=True)
        self.dummyMatrix = (self.m!=self.backgroundColor).astype(np.uint8) 
        # Symmetries
        self.lrSymmetric = np.array_equal(self.m, np.fliplr(self.m))
        # Up-Down
        self.udSymmetric = np.array_equal(self.m, np.flipud(self.m))
        # Diagonals (only if square)
        if self.m.shape[0] == self.m.shape[1]:
            self.d1Symmetric = np.array_equal(self.m, self.m.T)
            self.d2Symmetric = np.array_equal(np.fliplr(self.m), (np.fliplr(self.m)).T)
        else:
            self.d1Symmetric = False
            self.d2Symmetric = False
        self.totalSymmetric = self.lrSymmetric and self.udSymmetric and \
        self.d1Symmetric and self.d2Symmetric
        
        self.fullBorders = []
        for f in self.frontiers:
            if f.color != self.backgroundColor:
                if f.position==0:
                    self.fullBorders.append(f)
                elif (f.direction=='h' and f.position==self.shape[0]-1) or\
                (f.direction=='v' and f.position==self.shape[1]-1):
                    self.fullBorders.append(f)
               
        self.isVertical = False
        self.isHorizontal = False
        if len(self.frontiers)!=0:
            self.isVertical = all([f.direction=='v' for f in self.frontiers])
            self.isHorizontal = all([f.direction=='h' for f in self.frontiers])
    
    def getColors(self):
        unique, counts = np.unique(self.m, return_counts=True)
        return dict(zip(unique, counts))
    
    def getShapes(self, color=None, bigOrSmall=None, isBorder=None, diag=False):
        """
        Return a list of the shapes meeting the required specifications.
        """
        if diag:
            candidates = self.dShapes
        else:
            candidates = self.shapes
        if color != None:
            candidates = [c for c in candidates if c.color == color]
        if isBorder==True:
            candidates = [c for c in candidates if c.isBorder]
        if isBorder==False:
            candidates = [c for c in candidates if not c.isBorder]
        if len(candidates) ==  0:
            return []
        sizes = [c.nPixels for c in candidates]
        if bigOrSmall == "big":
            maxSize = max(sizes)
            return [c for c in candidates if c.nPixels==maxSize]
        elif bigOrSmall == "small":
            minSize = min(sizes)
            return [c for c in candidates if c.nPixels==minSize]
        else:
            return candidates
        
    def followsColPattern(self):
        """
        This function checks whether the matrix follows a pattern of lines or
        columns being always the same (task 771 for example).
        Meant to be used for the output matrix mainly.
        It returns a number (length of the pattern) and "row" or "col".
        """
        m = self.m.copy()
        col0 = m[:,0]
        for i in range(1,int(m.shape[1]/2)+1):
            if np.all(col0 == m[:,i]):
                isPattern=True
                for j in range(i):
                    k=0
                    while k*i+j < m.shape[1]:
                        if np.any(m[:,j] != m[:,k*i+j]):
                            isPattern=False
                            break
                        k+=1
                    if not isPattern:
                        break
                if isPattern:
                    return i
        return False
    
    def followsRowPattern(self):
        m = self.m.copy()
        row0 = m[0,:]
        for i in range(1,int(m.shape[0]/2)+1):
            if np.all(row0 == m[i,:]):
                isPattern=True
                for j in range(i):
                    k=0
                    while k*i+j < m.shape[0]:
                        if np.any(m[j,:] != m[k*i+j,:]):
                            isPattern=False
                            break
                        k+=1
                    if not isPattern:
                        break
                if isPattern:
                    return i
        return False
    
    def isUniqueShape(self, shape):
        count = 0
        for sh in self.shapes:
            if sh.hasSameShape(shape):
                count += 1
        if count==1:
            return True
        return False
    
    def getShapeAttributes(self, backgroundColor=0, singleColor=True, diagonals=True):
        '''
        Returns list of shape attributes that matches list of shapes
        Add:
            - is border
            - has neighbors
            - is reference
            - is referenced
        '''
        if singleColor: 
            if diagonals:   
                shapeList = [sh for sh in self.dShapes]
            else:   
                shapeList = [sh for sh in self.shapes]
            if len([sh for sh in shapeList if sh.color != backgroundColor]) == 0:
                return [set() for sh in shapeList]
        else:
            if diagonals: 
                shapeList = [sh for sh in self.multicolorDShapes]
            else:
                shapeList = [sh for sh in self.multicolorShapes]
            if len(shapeList) == 0:
                return [set()]
        attrList =[[] for i in range(len(shapeList))]
        if singleColor:
            cc = Counter([sh.color for sh in shapeList])
        if singleColor:
            sc = Counter([sh.nPixels for sh in shapeList if sh.color != backgroundColor])
        else:
            sc = Counter([sh.nPixels for sh in shapeList])
        largest, smallest, mcopies, mcolors = -1, 1000, 0, 0
        if singleColor:
            maxH, minH = max([sh.nHoles for sh in shapeList if sh.color != backgroundColor]),\
                            min([sh.nHoles for sh in shapeList if sh.color != backgroundColor])
        ila, ism = [], []
        for i in range(len(shapeList)):
            #color count
            if singleColor:
                if shapeList[i].color == backgroundColor:
                    attrList[i].append(-1)
                    continue
                else:
                    attrList[i].append(shapeList[i].color)
            else:
                attrList[i].append(shapeList[i].nColors)
                if shapeList[i].nColors > mcolors:
                    mcolors = shapeList[i].nColors
            #copies
            if singleColor:
                attrList[i] = [np.count_nonzero([np.all(shapeList[i].pixels == osh.pixels) for osh in shapeList])] + attrList[i]
                if attrList[i][0] > mcopies:
                    mcopies = attrList[i][0]
            else: 
                attrList[i] = [np.count_nonzero([shapeList[i] == osh for osh in shapeList])] + attrList[i]
                if attrList[i][0] > mcopies:
                    mcopies = attrList[i][0]
            #unique color?
            if singleColor:
                if cc[shapeList[i].color] == 1:
                    attrList[i].append('UnCo')
            #more of x color?
            if not singleColor:
                for c in range(10):
                    if shapeList[i].colorCount[c] > 0 and  shapeList[i].colorCount[c] == max([sh.colorCount[c] for sh in shapeList]):
                        attrList[i].append('mo'+str(c))    
            #largest?
            if len(shapeList[i].pixels) >= largest:
                ila += [i]
                if len(shapeList[i].pixels) > largest:
                    largest = len(shapeList[i].pixels)
                    ila = [i]
            #smallest?
            if len(shapeList[i].pixels) <= smallest:
                ism += [i]
                if len(shapeList[i].pixels) < smallest:
                    smallest = len(shapeList[i].pixels)
                    ism = [i]
            #unique size
            if sc[shapeList[i].nPixels] == 1 and len(sc) == 2:
                attrList[i].append('UnSi')
            #symmetric?
            if shapeList[i].lrSymmetric:
                attrList[i].append('LrSy')
            else:
                attrList[i].append('NlrSy')
            if shapeList[i].udSymmetric:
                attrList[i].append('UdSy')
            else:
                attrList[i].append('NudSy')
            if shapeList[i].d1Symmetric: 
                attrList[i].append('D1Sy')
            else:
                attrList[i].append('ND1Sy')
            if shapeList[i].d2Symmetric:
                attrList[i].append('D2Sy')
            else:
                attrList[i].append('ND2Sy')
            attrList[i].append(shapeList[i].position)
            #pixels
            if len(shapeList[i].pixels) == 1:
                attrList[i].append('PiXl')
            #holes
            if singleColor:
                if maxH>minH:
                    if shapeList[i].nHoles == maxH:
                        attrList[i].append('MoHo')
                    elif shapeList[i].nHoles == minH:
                        attrList[i].append('LeHo')           
            #is referenced by a full/partial frame?
                if any((shapeList[i].position[0] >= fr.position[0] and shapeList[i].position[1] >= fr.position[1]\
                        and shapeList[i].position[0] + shapeList[i].shape[0] <= fr.position[0] + fr.shape[0] and\
                        shapeList[i].position[1] + shapeList[i].shape[1] <= fr.position[1] + fr.shape[1] and\
                        shapeList[i].color != fr.color) for fr in self.partialFrames):
                    attrList[i].append('IsRef')
                if any((shapeList[i].position[0] >= fr.position[0] and shapeList[i].position[1] >= fr.position[1]\
                        and shapeList[i].position[0] + shapeList[i].shape[0] <= fr.position[0] + fr.shape[0] and\
                        shapeList[i].position[1] + shapeList[i].shape[1] <= fr.position[1] + fr.shape[1] and\
                        shapeList[i].color != fr.color) for fr in self.fullFrames):
                    attrList[i].append('IsFRef')
    
        if len(ism) == 1:
            attrList[ism[0]].append('SmSh')
        if len(ila) == 1:
            attrList[ila[0]].append('LaSh')
        for i in range(len(shapeList)):
            if len(attrList[i]) > 0 and attrList[i][0] == mcopies:
                attrList[i].append('MoCo')
        if not singleColor:
            for i in range(len(shapeList)):
                if len(attrList[i]) > 0 and attrList[i][1] == mcolors:
                    attrList[i].append('MoCl')
        if [l[0] for l in attrList].count(1) == 1:
            for i in range(len(shapeList)):
                if len(attrList[i]) > 0 and attrList[i][0] == 1:
                    attrList[i].append('UnSh')
                    break
        return [set(l[1:]) for l in attrList]
            

# %% Class Sample
class Sample():
    def __init__(self, s, trainOrTest, submission=False, backgroundColor=None):
        
        self.inMatrix = Matrix(s['input'], backgroundColor=backgroundColor)
        
        if trainOrTest == "train" or submission==False:
            self.outMatrix = Matrix(s['output'], backgroundColor=backgroundColor)
                    
            # We want to compare the input and the output
            # Do they have the same dimensions?
            self.sameHeight = self.inMatrix.shape[0] == self.outMatrix.shape[0]
            self.sameWidth = self.inMatrix.shape[1] == self.outMatrix.shape[1]
            self.sameShape = self.sameHeight and self.sameWidth
            
            # Is the input shape a factor of the output shape?
            # Or the other way around?
            if not self.sameShape:
                if (self.inMatrix.shape[0] % self.outMatrix.shape[0]) == 0 and \
                (self.inMatrix.shape[1] % self.outMatrix.shape[1]) == 0 :
                    self.outShapeFactor = (int(self.inMatrix.shape[0]/self.outMatrix.shape[0]),\
                                           int(self.inMatrix.shape[1]/self.outMatrix.shape[1]))
                if (self.outMatrix.shape[0] % self.inMatrix.shape[0]) == 0 and \
                (self.outMatrix.shape[1] % self.inMatrix.shape[1]) == 0 :
                    self.inShapeFactor = (int(self.outMatrix.shape[0]/self.inMatrix.shape[0]),\
                                          int(self.outMatrix.shape[1]/self.inMatrix.shape[1]))

            # Is one a subset of the other? for now always includes diagonals
            self.inSmallerThanOut = all(self.inMatrix.shape[i] <= self.outMatrix.shape[i] for i in [0,1]) and not self.sameShape
            self.outSmallerThanIn = all(self.inMatrix.shape[i] >= self.outMatrix.shape[i] for i in [0,1]) and not self.sameShape
    
            #R: Is the output a shape (faster than checking if is a subset?
    
            if self.outSmallerThanIn:
                #check if output is the size of a multicolored shape
                self.outIsInMulticolorShapeSize = any((sh.shape == self.outMatrix.shape) for sh in self.inMatrix.multicolorShapes)
                self.outIsInMulticolorDShapeSize = any((sh.shape == self.outMatrix.shape) for sh in self.inMatrix.multicolorDShapes)
            self.commonShapes, self.commonDShapes, self.commonMulticolorShapes, self.commonMulticolorDShapes = [], [], [], []
            if len(self.inMatrix.shapes) < 15 or len(self.outMatrix.shapes) < 10:
                self.commonShapes = self.getCommonShapes(diagonal=False, sameColor=True,\
                                                     multicolor=False, rotation=True, scaling=True, mirror=True)
            if len(self.inMatrix.dShapes) < 15 or len(self.outMatrix.dShapes) < 10:
                self.commonDShapes = self.getCommonShapes(diagonal=True, sameColor=True,\
                                                      multicolor=False, rotation=True, scaling=True, mirror=True)
            if len(self.inMatrix.multicolorShapes) < 15 or len(self.outMatrix.multicolorShapes) < 10:
                self.commonMulticolorShapes = self.getCommonShapes(diagonal=False, sameColor=True,\
                                                               multicolor=True, rotation=True, scaling=True, mirror=True)
            if len(self.inMatrix.multicolorDShapes) < 15 or len(self.outMatrix.multicolorDShapes) < 10:
                self.commonMulticolorDShapes = self.getCommonShapes(diagonal=True, sameColor=True,\
                                                                multicolor=True, rotation=True, scaling=True, mirror=True)

            # Which colors are there in the sample?
            self.colors = set(self.inMatrix.colors | self.outMatrix.colors)
            self.commonColors = set(self.inMatrix.colors & self.outMatrix.colors)
            self.nColors = len(self.colors)
            # Do they have the same colors?
            self.sameColors = len(self.colors) == len(self.commonColors)
            # Do they have the same number of colors?
            self.sameNumColors = self.inMatrix.nColors == self.outMatrix.nColors
            # Does output contain all input colors or viceversa?
            self.inHasOutColors = self.outMatrix.colors <= self.inMatrix.colors  
            self.outHasInColors = self.inMatrix.colors <= self.outMatrix.colors
            if self.sameShape:
                # Which pixels changes happened? How many times?
                self.changedPixels = Counter()
                self.sameColorCount = self.inMatrix.colorCount == self.outMatrix.colorCount
                for i, j in np.ndindex(self.inMatrix.shape):
                    if self.inMatrix.m[i,j] != self.outMatrix.m[i,j]:
                        self.changedPixels[(self.inMatrix.m[i,j], self.outMatrix.m[i,j])] += 1
                # Are any of these changes complete? (i.e. all pixels of one color are changed to another one)
                self.completeColorChanges = set(change for change in self.changedPixels.keys() if\
                                             self.changedPixels[change]==self.inMatrix.colorCount[change[0]] and\
                                             change[0] not in self.outMatrix.colorCount.keys())
                self.allColorChangesAreComplete = len(self.changedPixels) == len(self.completeColorChanges)
                # Does any color never change?
                self.changedInColors = set(change[0] for change in self.changedPixels.keys())
                self.changedOutColors = set(change[1] for change in self.changedPixels.keys())
                self.unchangedColors = set(x for x in self.colors if x not in set.union(self.changedInColors, self.changedOutColors))
                # Colors that stay unchanged
                self.fixedColors = set(x for x in self.colors if x not in set.union(self.changedInColors, self.changedOutColors))
            
            if self.sameShape and self.sameColorCount:
                self.sameRowCount = True
                for r in range(self.inMatrix.shape[0]):
                    _,inCounts = np.unique(self.inMatrix.m[r,:], return_counts=True)
                    _,outCounts = np.unique(self.outMatrix.m[r,:], return_counts=True)
                    if not np.array_equal(inCounts, outCounts):
                        self.sameRowCount = False
                        break
                self.sameColCount = True
                for c in range(self.inMatrix.shape[1]):
                    _,inCounts = np.unique(self.inMatrix.m[:,c], return_counts=True)
                    _,outCounts = np.unique(self.outMatrix.m[:,c], return_counts=True)
                    if not np.array_equal(inCounts, outCounts):
                        self.sameColCount = False
                        break
                    
            # Shapes in the input that are fixed
            if self.sameShape:
                self.fixedShapes = []
                for sh in self.inMatrix.shapes:
                    if sh.color in self.fixedColors:
                        continue
                    shapeIsFixed = True
                    for i,j in np.ndindex(sh.shape):
                        if sh.m[i,j] != 255:
                            if self.outMatrix.m[sh.position[0]+i,sh.position[1]+j]!=sh.m[i,j]:
                                shapeIsFixed=False
                                break
                    if shapeIsFixed:
                        self.fixedShapes.append(sh)
                    
            # Frames
            self.commonFullFrames = [f for f in self.inMatrix.fullFrames if f in self.outMatrix.fullFrames]
            if len(self.inMatrix.fullFrames)==1:
                frameM = self.inMatrix.fullFrames[0].m.copy()
                frameM[frameM==255] = self.inMatrix.fullFrames[0].background
                if frameM.shape==self.outMatrix.shape:
                    self.frameIsOutShape = True
                elif frameM.shape==(self.outMatrix.shape[0]+1, self.outMatrix.shape[1]+1):
                    self.frameInsideIsOutShape = True
            
            # Grids
            # Is the grid the same in the input and in the output?
            self.gridIsUnchanged = self.inMatrix.isGrid and self.outMatrix.isGrid \
            and self.inMatrix.grid == self.outMatrix.grid
            # Does the shape of the grid cells determine the output shape?
            if hasattr(self.inMatrix, "grid") and self.inMatrix.grid.allCellsSameShape:
                self.gridCellIsOutputShape = self.outMatrix.shape == self.inMatrix.grid.cellShape
            # Does the shape of the input determine the shape of the grid cells of the output?
            if hasattr(self.outMatrix, "grid") and self.outMatrix.grid.allCellsSameShape:
                self.gridCellIsInputShape = self.inMatrix.shape == self.outMatrix.grid.cellShape
            # Do all the grid cells have one color?
            if self.gridIsUnchanged:
                self.gridCellsHaveOneColor = self.inMatrix.grid.allCellsHaveOneColor and\
                                             self.outMatrix.grid.allCellsHaveOneColor
            # Asymmetric grids
            self.asymmetricGridIsUnchanged = self.inMatrix.isAsymmetricGrid and self.outMatrix.isAsymmetricGrid \
            and self.inMatrix.asymmetricGrid == self.outMatrix.asymmetricGrid
            if self.asymmetricGridIsUnchanged:
                self.asymmetricGridCellsHaveOneColor = self.inMatrix.asymmetricGrid.allCellsHaveOneColor and\
                self.outMatrix.asymmetricGrid.allCellsHaveOneColor
            
            # Is there a blank to fill?
            self.inputHasBlank = len(self.inMatrix.blanks)>0
            if self.inputHasBlank:
                for s in self.inMatrix.blanks:
                    if s.shape == self.outMatrix.shape:
                        self.blankToFill = s
             
            # Does the output matrix follow a pattern?
            self.followsRowPattern = self.outMatrix.followsRowPattern()
            self.followsColPattern = self.outMatrix.followsColPattern()
            
            # Full borders and horizontal/vertical
            if self.sameShape:
                self.commonFullBorders = []
                for inBorder in self.inMatrix.fullBorders:
                    for outBorder in self.outMatrix.fullBorders:
                        if inBorder==outBorder:
                            self.commonFullBorders.append(inBorder)
                
                self.isHorizontal = self.inMatrix.isHorizontal and self.outMatrix.isHorizontal
                self.isVertical = self.inMatrix.isVertical and self.outMatrix.isVertical

    def getCommonShapes(self, diagonal=True, multicolor=False, sameColor=False, samePosition=False, rotation=False, \
                     mirror=False, scaling=False):
        comSh = []
        if diagonal:
            if not multicolor:
                ishs = self.inMatrix.dShapes
                oshs = self.outMatrix.dShapes
            else:
                ishs = self.inMatrix.multicolorDShapes
                oshs = self.outMatrix.multicolorDShapes
        else:
            if not multicolor:
                ishs = self.inMatrix.shapes
                oshs = self.outMatrix.shapes
            else:
                ishs = self.inMatrix.multicolorShapes
                oshs = self.outMatrix.multicolorShapes
        #Arbitrary: shapes have size < 100 and > 3
        for ish in ishs:
            outCount = 0
            if len(ish.pixels) < 4 or len(ish.pixels) > 100:
                continue
            for osh in oshs:
                if len(osh.pixels) < 4 or len(osh.pixels) > 100:
                    continue
                if ish.hasSameShape(osh, sameColor=sameColor, samePosition=samePosition,\
                                    rotation=rotation, mirror=mirror, scaling=scaling):
                    outCount += 1
            if outCount > 0:
                comSh.append((ish, np.count_nonzero([ish.hasSameShape(ish2, sameColor=sameColor, samePosition=samePosition,\
                                    rotation=rotation, mirror=mirror, scaling=scaling) for ish2 in ishs]), outCount))
        return comSh

# %% Class Task
class Task():
    def __init__(self, t, i, submission=False, backgrounds=None):
        self.task = t
        self.index = i
        self.submission = submission
        
        if backgrounds==None:
            self.trainSamples = [Sample(s, "train", submission) for s in t['train']]
            self.testSamples = [Sample(s, "test", submission) for s in t['test']]
        else:
            self.trainSamples = [Sample(t["train"][s], "train", submission, backgrounds["train"][s]) for s in range(len(t['train']))]
            self.testSamples = [Sample(t["test"][s], "test", submission, backgrounds["test"][s]) for s in range(len(t['test']))]
        
        self.nTrain = len(self.trainSamples)
        self.nTest = len(self.testSamples)
        
        # Common properties I want to know:
        
        # Dimension:
        # Do all input/output matrices have the same shape?
        inShapes = [s.inMatrix.shape for s in self.trainSamples]
        self.sameInShape = self.allEqual(inShapes)
        if self.sameInShape:
            self.inShape = self.trainSamples[0].inMatrix.shape
        outShapes = [s.outMatrix.shape for s in self.trainSamples]
        self.sameOutShape = self.allEqual(outShapes)
        if self.sameOutShape:
            self.outShape = self.trainSamples[0].outMatrix.shape
            
        # Do all output matrices have the same shape as the input matrix?
        self.sameIOShapes = all([s.sameShape for s in self.trainSamples])
        
        # Are the input/output matrices always squared?
        self.inMatricesSquared = all([s.inMatrix.shape[0] == s.inMatrix.shape[1] \
                                      for s in self.trainSamples+self.testSamples])
        self.outMatricesSquared = all([s.outMatrix.shape[0] == s.outMatrix.shape[1] \
                                       for s in self.trainSamples])
    
        # Are shapes of in (out) matrices always a factor of the shape of the 
        # out (in) matrices?
        if all([hasattr(s, 'inShapeFactor') for s in self.trainSamples]):
            if self.allEqual([s.inShapeFactor for s in self.trainSamples]):
                self.inShapeFactor = self.trainSamples[0].inShapeFactor
            elif all([s.inMatrix.shape[0]**2 == s.outMatrix.shape[0] and \
                      s.inMatrix.shape[1]**2 == s.outMatrix.shape[1] \
                      for s in self.trainSamples]):
                self.inShapeFactor = "squared"
            elif all([s.inMatrix.shape[0]**2 == s.outMatrix.shape[0] and \
                      s.inMatrix.shape[1] == s.outMatrix.shape[1] \
                      for s in self.trainSamples]):
                self.inShapeFactor = "xSquared"
            elif all([s.inMatrix.shape[0] == s.outMatrix.shape[0] and \
                      s.inMatrix.shape[1]**2 == s.outMatrix.shape[1] \
                      for s in self.trainSamples]):
                self.inShapeFactor = "ySquared"
            elif all([s.inMatrix.shape[0]*s.inMatrix.nColors == s.outMatrix.shape[0] and \
                     s.inMatrix.shape[1]*s.inMatrix.nColors == s.outMatrix.shape[1] \
                     for s in self.trainSamples]):
                self.inShapeFactor = "nColors"
            elif all([s.inMatrix.shape[0]*(s.inMatrix.nColors-1) == s.outMatrix.shape[0] and \
                     s.inMatrix.shape[1]*(s.inMatrix.nColors-1) == s.outMatrix.shape[1] \
                     for s in self.trainSamples]):
                self.inShapeFactor = "nColors-1"
        if all([hasattr(s, 'outShapeFactor') for s in self.trainSamples]):
            if self.allEqual([s.outShapeFactor for s in self.trainSamples]):
                self.outShapeFactor = self.trainSamples[0].outShapeFactor
                
        # Is the output always smaller?
        self.outSmallerThanIn = all(s.outSmallerThanIn for s in self.trainSamples)
        self.inSmallerThanOut = all(s.inSmallerThanOut for s in self.trainSamples)
        
        # Symmetries:
        # Are all outputs LR, UD, D1 or D2 symmetric?
        self.lrSymmetric = all([s.outMatrix.lrSymmetric for s in self.trainSamples])
        self.udSymmetric = all([s.outMatrix.udSymmetric for s in self.trainSamples])
        self.d1Symmetric = all([s.outMatrix.d1Symmetric for s in self.trainSamples])
        self.d2Symmetric = all([s.outMatrix.d2Symmetric for s in self.trainSamples])
        
        # Colors
        # How many colors are there in the input? Is it always the same number?
        # How many colors are there in the output? Is it always the same number?
        self.sameNumColors = all([s.sameNumColors for s in self.trainSamples])
        self.nInColors = [s.inMatrix.nColors for s in self.trainSamples] + \
        [s.inMatrix.nColors for s in self.testSamples]
        self.sameNInColors = self.allEqual(self.nInColors)
        self.nOutColors = [s.outMatrix.nColors for s in self.trainSamples]
        self.sameNOutColors = self.allEqual(self.nOutColors)
        # Which colors does the input have? Union and intersection.
        self.inColors = [s.inMatrix.colors for s in self.trainSamples+self.testSamples]
        self.commonInColors = set.intersection(*self.inColors)
        self.totalInColors = set.union(*self.inColors)
        # Which colors does the output have? Union and intersection.
        self.outColors = [s.outMatrix.colors for s in self.trainSamples]
        self.commonOutColors = set.intersection(*self.outColors)
        self.totalOutColors = set.union(*self.outColors)
        # Which colors appear in every sample?
        self.sampleColors = [s.colors for s in self.trainSamples]
        self.commonSampleColors = set.intersection(*self.sampleColors)
        self.almostCommonColors = self.commonSampleColors.copy()
        if self.nTrain > 3:
            for color in range(10):
                if color not in self.almostCommonColors:
                    apps = 0
                    for i in range(self.nTrain):
                        if color in self.trainSamples[i].colors:
                            apps += 1
                    if apps == self.nTrain-1:
                        self.almostCommonColors.add(color)
        # Input colors of the test samples
        self.testInColors = [s.inMatrix.colors for s in self.testSamples]
        # Are there the same number of colors in every sample?
        self.sameNSampleColors = self.allEqual([len(sc) for sc in self.sampleColors]) and\
        all([len(s.inMatrix.colors | self.commonOutColors) <= len(self.sampleColors[0]) for s in self.testSamples])
        # How many colors are there in total? Which ones?
        self.colors = self.totalInColors | self.totalOutColors
        self.nColors = len(self.colors)
        # Does the output always have the same colors as the input?
        if self.sameNumColors:
            self.sameIOColors = all([i==j for i,j in zip(self.inColors, self.outColors)])
        if self.sameIOShapes:
            # Do the matrices have the same color count?
            self.sameColorCount = all([s.sameColorCount for s in self.trainSamples])
            if self.sameColorCount:
                self.sameRowCount = all([s.sameRowCount for s in self.trainSamples])
                self.sameColCount = all([s.sameColCount for s in self.trainSamples])
            # Which color changes happen? Union and intersection.
            cc = [set(s.changedPixels.keys()) for s in self.trainSamples]
            self.colorChanges = set.union(*cc)
            self.commonColorChanges = set.intersection(*cc)
            # Does any color always change? (to and from)
            self.changedInColors = [s.changedInColors for s in self.trainSamples]
            self.commonChangedInColors = set.intersection(*self.changedInColors)
            self.changedOutColors = [s.changedOutColors for s in self.trainSamples]
            self.commonChangedOutColors = set.intersection(*self.changedOutColors)
            self.commonOnlyChangedInColors = self.commonChangedInColors - set.union(*self.changedOutColors)
            # Complete color changes
            self.completeColorChanges = [s.completeColorChanges for s in self.trainSamples]
            self.commonCompleteColorChanges = set.intersection(*self.completeColorChanges)
            self.allColorChangesAreComplete = all([s.allColorChangesAreComplete for s in self.trainSamples])
            # Are there any fixed colors?
            self.fixedColors = set.intersection(*[s.fixedColors for s in self.trainSamples])
            self.fixedColors2 = set.union(*[s.fixedColors for s in self.trainSamples]) - \
            set.union(*[s.changedInColors for s in self.trainSamples]) -\
            set.union(*[s.changedOutColors for s in self.trainSamples])
            # Does any color never change?
            if self.commonChangedInColors == set(self.changedInColors[0]):
                self.unchangedColors = set(range(10)) - self.commonChangedInColors
            else:
                self.unchangedColors = [s.unchangedColors for s in self.trainSamples]
                self.unchangedColors = set.intersection(*self.unchangedColors)
                
        # Is the number of pixels changed always the same?
        """
        if self.sameIOShapes:
            self.sameChanges = self.allEqual([s.diffPixels for s in self.trainSamples])
        """
            
        #R: is output a shape in the input
        self.outIsInMulticolorShapeSize = False
        self.outIsInMulticolorDShapeSize = False

        if all([(hasattr(s, "outIsInMulticolorShapeSize") and s.outIsInMulticolorShapeSize) for s in self.trainSamples]):
             self.outIsInMulticolorShapeSize = True
        if all([(hasattr(s, "outIsInMulticolorDShapeSize") and s.outIsInMulticolorDShapeSize) for s in self.trainSamples]):
             self.outIsInMulticolorDShapeSize = True
             
        self.nCommonInOutShapes = min(len(s.commonShapes) for s in self.trainSamples)
        self.nCommonInOutDShapes = min(len(s.commonDShapes) for s in self.trainSamples) 
        self.nCommonInOutMulticolorShapes = min(len(s.commonMulticolorShapes) for s in self.trainSamples)
        self.nCommonInOutMulticolorDShapes = min(len(s.commonMulticolorDShapes) for s in self.trainSamples) 
        
        if self.sameIOShapes:
            self.fixedShapes = []
            for s in self.trainSamples:
                for shape in s.fixedShapes:
                    self.fixedShapes.append(shape)
            self.fixedShapeFeatures = []
            nFeatures = len(self.trainSamples[0].inMatrix.shapes[0].boolFeatures)
            for i in range(nFeatures):
                self.fixedShapeFeatures.append(True)
            for shape in self.fixedShapes:
                self.fixedShapeFeatures = [shape.boolFeatures[i] and self.fixedShapeFeatures[i] \
                                             for i in range(nFeatures)]
        
        # Grids:
        self.inputIsGrid = all([s.inMatrix.isGrid for s in self.trainSamples+self.testSamples])
        self.outputIsGrid = all([s.outMatrix.isGrid for s in self.trainSamples])
        self.hasUnchangedGrid = all([s.gridIsUnchanged for s in self.trainSamples])
        if all([hasattr(s, "gridCellIsOutputShape") for s in self.trainSamples]):
            self.gridCellIsOutputShape = all([s.gridCellIsOutputShape for s in self.trainSamples])
        if all([hasattr(s, "gridCellIsInputShape") for s in self.trainSamples]):
            self.gridCellIsInputShape = all([s.gridCellIsInputShape for s in self.trainSamples])
        if self.hasUnchangedGrid:
            self.gridCellsHaveOneColor = all([s.gridCellsHaveOneColor for s in self.trainSamples])
            self.outGridCellsHaveOneColor = all([s.outMatrix.grid.allCellsHaveOneColor for s in self.trainSamples])
        # Asymmetric grids
        self.inputIsAsymmetricGrid = all([s.inMatrix.isAsymmetricGrid for s in self.trainSamples+self.testSamples])
        self.hasUnchangedAsymmetricGrid = all([s.asymmetricGridIsUnchanged for s in self.trainSamples])
        if self.hasUnchangedAsymmetricGrid:
            self.assymmetricGridCellsHaveOneColor = all([s.asymmetricGridCellsHaveOneColor for s in self.trainSamples])
            self.outAsymmetricGridCellsHaveOneColor = all([s.outMatrix.asymmetricGrid.allCellsHaveOneColor for s in self.trainSamples])
        
        # Background color
        
        # Is there always a background color? Which one?
        if self.allEqual([s.inMatrix.backgroundColor for s in self.trainSamples]) and\
        self.trainSamples[0].inMatrix.backgroundColor == self.testSamples[0].inMatrix.backgroundColor:
            self.backgroundColor = self.trainSamples[0].inMatrix.backgroundColor
        elif self.hasUnchangedAsymmetricGrid and all([s.inMatrix.asymmetricGrid.nCells>6 for s in self.trainSamples]):
            self.backgroundColor = self.trainSamples[0].inMatrix.asymmetricGrid.color
            for sample in self.trainSamples:
                sample.inMatrix.backgroundColor = self.backgroundColor
                sample.outMatrix.backgroundColor = self.backgroundColor
            for sample in self.testSamples:
                sample.inMatrix.backgroundColor = self.backgroundColor
        else:
            self.backgroundColor = -1
            
        self.orderedColors = self.orderColors()
        
        # Shapes:
        # Does the task ONLY involve changing colors of shapes?
        if self.sameIOShapes:
            self.onlyShapeColorChanges = True
            for s in self.trainSamples:
                nShapes = s.inMatrix.nShapes
                if s.outMatrix.nShapes != nShapes:
                    self.onlyShapeColorChanges = False
                    break
                for shapeI in range(nShapes):
                    if not s.inMatrix.shapes[shapeI].hasSameShape(s.outMatrix.shapes[shapeI]):
                        self.onlyShapeColorChanges = False
                        break
                if not self.onlyShapeColorChanges:
                    break
            
            # Get a list with the number of pixels shapes have
            if self.onlyShapeColorChanges:
                nPixels = set()
                for s in self.trainSamples:
                    for shape in s.inMatrix.shapes:
                        nPixels.add(shape.nPixels)
                self.shapePixelNumbers =  list(nPixels)
                
        #R: Are there any common input shapes accross samples?
        self.commonInShapes = []
        for sh1 in self.trainSamples[0].inMatrix.shapes:
            if sh1.color == self.trainSamples[0].inMatrix.backgroundColor:
                continue
            addShape = True
            for s in range(1,self.nTrain):
                if not any([sh1.pixels == sh2.pixels for sh2 in self.trainSamples[s].inMatrix.shapes]):
                    addShape = False
                    break
            if addShape and sh1 not in self.commonInShapes:
                self.commonInShapes.append(sh1)

        self.commonInDShapes = []
        for sh1 in self.trainSamples[0].inMatrix.dShapes:
            if sh1.color == self.trainSamples[0].inMatrix.backgroundColor:
                continue
            addShape = True
            for s in range(1,self.nTrain):
                if not any([sh1.pixels == sh2.pixels for sh2 in self.trainSamples[s].inMatrix.dShapes]):
                    addShape = False
                    break
            if addShape:
                self.commonInDShapes.append(sh1)
        #Does the task use the information of isolated pixels?
        #if all(s.inMatrix.nIsolatedPixels)
        #Does the input always consist in two shapes?
        self.twoShapeTask = (False, False, False, False, 1)
        if all(len(s.inMatrix.multicolorDShapes)==2 for s in self.trainSamples):
            self.twoShapeTask = (True, True, True, False, 1)
            if all(s.inMatrix.multicolorDShapes[0].shape == s.inMatrix.multicolorDShapes[1].shape for s in self.trainSamples):
                self.twoShapeTask = (True, True, True, True, 1)
                
        elif all(len(s.inMatrix.multicolorShapes)==2 for s in self.trainSamples):
            self.twoShapeTask = (True, True, False, False, 1)
            if all(s.inMatrix.multicolorShapes[0].shape == s.inMatrix.multicolorShapes[1].shape for s in self.trainSamples):
                self.twoShapeTask = (True, True, False, True, 1)
        elif all(len(s.inMatrix.dShapes)==2 for s in self.trainSamples):
            self.twoShapeTask = (True, False, True, False, 1)
            if all(s.inMatrix.dShapes[0].shape == s.inMatrix.dShapes[1].shape for s in self.trainSamples):
                self.twoShapeTask = (True, False, True, True, 1)
        if self.inputIsGrid:
            if all(s.inMatrix.grid.nCells == 2 for s in self.trainSamples):
                self.twoShapeTask = (True, False, False, True, 2)
        elif  all((s.inMatrix.shape[0]*2 == s.inMatrix.shape[1] or\
                  s.inMatrix.shape[0] == s.inMatrix.shape[1]*2) for s in self.trainSamples):
                self.twoShapeTask = (True, False, False, True, 3)
                
        #Are all output matrices equal mod nonBackgroundColor?
        if self.sameOutShape:
            self.sameOutDummyMatrix = all(np.all(self.trainSamples[0].outMatrix.dummyMatrix==s.outMatrix.dummyMatrix) for s in self.trainSamples)
        # Frames
        self.hasFullFrame = all([len(s.inMatrix.fullFrames)>0 for s in self.trainSamples])
        self.hasPartialFrame = all([len(s.inMatrix.partialFrames)>0 for s in self.trainSamples])
        # Is the task about filling a blank?
        self.fillTheBlank =  all([hasattr(s, 'blankToFill') for s in self.trainSamples])
                
        # Do all output matrices follow a pattern?
        self.followsRowPattern = all([s.followsRowPattern != False for s in self.trainSamples])
        self.followsColPattern = all([s.followsColPattern != False for s in self.trainSamples])
        if self.followsRowPattern:
            self.rowPatterns = [s.outMatrix.followsRowPattern() for s in self.trainSamples]
        if self.followsColPattern:
            self.colPatterns = [s.outMatrix.followsColPattern() for s in self.trainSamples]
        
        # Full Borders / Requires vertical-horizontal rotation
        if self.sameIOShapes:
            if self.submission:
                self.hasOneFullBorder = all([len(s.commonFullBorders)==1 for s in self.trainSamples])
            else:
                self.hasOneFullBorder = all([hasattr(s, 'commonFullBorders') and len(s.commonFullBorders)==1 for s in self.trainSamples+self.testSamples])
            self.requiresHVRotation = False
            if not (self.allEqual([s.isHorizontal for s in self.trainSamples]) or \
                    self.allEqual([s.isVertical for s in self.trainSamples])):    
                self.requiresHVRotation = all([s.isHorizontal or s.isVertical for s in self.trainSamples])
        
    def allEqual(self, x):
        """
        x is a list.
        Returns true if all elements of x are equal.
        """
        if len(x) == 0:
            return False
        return x.count(x[0]) == len(x)
    
    def orderColors(self):
        """
        The aim of this function is to give the colors a specific order, in
        order to do the OHE in the right way for every sample.
        """
        orderedColors = []
        # 1: Colors that appear in every sample, input and output, and never
        # change. Only valid if t.sameIOShapes
        if self.sameIOShapes:
            for c in self.fixedColors:
                if all([c in sample.inMatrix.colors for sample in self.testSamples]):
                    orderedColors.append(c)
        # 2: Colors that appear in every sample and are always changed from,
        # never changed to.
            for c in self.commonChangedInColors:
                if c not in self.commonChangedOutColors:
                    if all([c in sample.inMatrix.colors for sample in self.testSamples]):
                        if c not in orderedColors:
                            orderedColors.append(c)
        # 3: Colors that appear in every sample and are always changed to,
        # never changed from.
            for c in self.commonChangedOutColors:
                if not all([c in sample.inMatrix.colors for sample in self.trainSamples]):
                    if c not in orderedColors:
                        orderedColors.append(c)
        # 4: Add the background color.
        if self.backgroundColor != -1:
            if self.backgroundColor not in orderedColors:
                orderedColors.append(self.backgroundColor)
        # 5: Other colors that appear in every input.
        for c in self.commonInColors:
            if all([c in sample.inMatrix.colors for sample in self.testSamples]):
                if c not in orderedColors:
                    orderedColors.append(c)
        # 6: Other colors that appear in every output.
        for c in self.commonOutColors:
            if not all([c in sample.inMatrix.colors for sample in self.trainSamples]):
                if c not in orderedColors:
                    orderedColors.append(c)
                
        # TODO Dealing with grids and frames
        
        return orderedColors   
    
#############################################################################
# %% Models
        
class OneConvModel(nn.Module):
    def __init__(self, ch=10, kernel=3, padVal = -1):
        super(OneConvModel, self).__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=kernel, bias=0)
        self.pad = nn.ConstantPad2d(int((kernel-1)/2), padVal)
        
    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.conv(self.pad(x))
        return x
    
class LinearModel(nn.Module):
    def __init__(self, inSize, outSize, ch):
        super(LinearModel, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.ch = ch
        self.fc = nn.Linear(inSize[0]*inSize[1]*ch, outSize[0]*outSize[1]*ch)
        
    def forward(self, x):
        x = x.view(1, self.inSize[0]*self.inSize[1]*self.ch)
        x = self.fc(x)
        x = x.view(1, self.ch, self.outSize[0]*self.outSize[1])
        return x
    
class LinearModelDummy(nn.Module): #(dummy = 2 channels)
    def __init__(self, inSize, outSize):
        super(LinearModelDummy, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.fc = nn.Linear(inSize[0]*inSize[1]*2, outSize[0]*outSize[1]*2, bias=0)
        
    def forward(self, x):
        x = x.view(1, self.inSize[0]*self.inSize[1]*2)
        x = self.fc(x)
        x = x.view(1, 2, self.outSize[0]*self.outSize[1])
        return x
    
class SimpleLinearModel(nn.Module):
    def __init__(self, inSize, outSize):
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(inSize, outSize)
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
def pixelCorrespondence(t):
    """
    Returns a dictionary. Keys are positions of the output matrix. Values are
    the pixel in the input matrix it corresponds to.
    Function only valid if t.sameInSahpe and t.sameOutShape
    """
    pixelsColoredAllSamples = []
    # In which positions does each color appear?
    for s in t.trainSamples:
        pixelsColored = [[] for i in range(10)]
        m = s.inMatrix.m
        for i,j in np.ndindex(t.inShape):
            pixelsColored[m[i,j]].append((i,j))
        pixelsColoredAllSamples.append(pixelsColored)
    # For each pixel in output matrix, find correspondent pixel in input matrix
    pixelMap = {}
    for i,j in np.ndindex(t.outShape):
        candidates = set()
        for s in range(t.nTrain):
            m = t.trainSamples[s].outMatrix.m
            if len(candidates) == 0:
                candidates = set(pixelsColoredAllSamples[s][m[i,j]])
            else:
                candidates = set(pixelsColoredAllSamples[s][m[i,j]]) & candidates
            if len(candidates) == 0:
                return {}
        pixelMap[(i,j)] = next(iter(candidates))
    
    return pixelMap

###############################################################################
# %% Utils

def identityM(matrix):
    """
    Function that, given Matrix, returns its corresponding numpy.ndarray m
    """
    if isinstance(matrix, np.ndarray):
        return matrix.copy()
    else:
        return matrix.m.copy()

def correctFixedColors(inMatrix, x, fixedColors, onlyChangedInColors):
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
        if m[i,j] in onlyChangedInColors:
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

def updateBestFunction(t, f, bestScore, bestFunction, checkPerfect=False, prevScore=None):
    """
    Given a task t, a partial function f, a best score and a best function, 
    this function executes f to all the matrices in t.trainSamples. If the
    resulting score is lower than bestScore, then it returns f and the new
    best score. Otherwise, it returns bestFunction again.
    """
    fun = copy.deepcopy(f)
    score = 0
    if checkPerfect:
        changedPixels = 0
    for sample in t.trainSamples:
        pred = fun(sample.inMatrix)
        score += incorrectPixels(sample.outMatrix.m, pred)
        if checkPerfect:
            changedPixels += incorrectPixels(pred, sample.inMatrix.m)
    if score < bestScore:
        bestScore = score
        bestFunction = fun
    if checkPerfect:
        if changedPixels != 0 and prevScore - changedPixels == score: 
            isPerfect = True
        else:
            isPerfect = False
        return fun, score, isPerfect
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

# %% Color symmetric pixels (task 653)

def colorSymmetricPixels(matrix, inColor, outColor, axis, includeAxis=False):
    m = matrix.m.copy()
    if axis=="lr":
        for i,j in np.ndindex((m.shape[0], int(m.shape[1]/2))):
            if m[i,j]==inColor and m[i,m.shape[1]-1-j]==inColor:
                m[i,j] = outColor
                m[i,m.shape[1]-1-j] = outColor
        if includeAxis and ((m.shape[1]%2)==1):
            j = int(m.shape[1]/2)
            for i in range(m.shape[0]):
                if m[i,j]==inColor:
                    m[i,j] = outColor
    if axis=="ud":
        for i,j in np.ndindex((int(m.shape[0]/2), m.shape[1])):
            if m[i,j]==inColor and m[m.shape[0]-1-i,j]==inColor:
                m[i,j] = outColor
                m[m.shape[0]-1-i,j] = outColor
        if includeAxis and ((m.shape[0]%2)==1):
            i = int(m.shape[0]/2)
            for j in range(m.shape[1]):
                if m[i,j]==inColor:
                    m[i,j] = outColor
    if axis=="d1":
        for i in range(m.shape[0]):
            for j in range(i):
                if m[i,j]==inColor and m[j,i]==inColor:
                    m[i,j] = outColor
                    m[j,i] = outColor
        if includeAxis:
            for i in range(m.shape[0]):
                if m[i,i]==inColor:
                    m[i,i] = outColor
    if axis=="d2":
        for i in range(m.shape[0]):
            for j in range(m.shape[0]-i-1):
                if m[i,j]==inColor and m[m.shape[1]-j-1,m.shape[0]-i-1]==inColor:
                    m[i,j] = outColor
                    m[m.shape[1]-j-1,m.shape[0]-i-1] = outColor
        if includeAxis:
            for i in range(m.shape[0]):
                if m[i, m.shape[0]-i-1]==inColor:
                   m[i, m.shape[0]-i-1] = outColor
                
    return m

def getBestColorSymmetricPixels(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    
    for cic in t.commonChangedInColors:
        for coc in t.commonChangedOutColors:
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="lr", includeAxis=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="lr", includeAxis=False)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="ud", includeAxis=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="ud", includeAxis=False)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            if all([s.inMatrix.shape[0]==s.inMatrix.shape[1] for s in t.trainSamples+t.testSamples]):
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d1", includeAxis=True)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d1", includeAxis=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d2", includeAxis=True)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d2", includeAxis=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
    return bestFunction

# %% Train and predict models  

def trainCNNDummyColor(t, k, pad):
    """
    This function trains a CNN with only one convolution of filter k and with
    padding values equal to pad.
    The training samples will have two channels: the background color and any
    other color. The training loop loops through all the non-background colors
    of each sample, treating them independently.
    This is useful for tasks like number 3.
    """
    model = OneConvModel(2, k, pad)
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
    model = OneConvModel(nChannels, k, pad)
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

def mergeMatrices(matrices, backgroundColor, mergeColor=None):
    """
    All matrices are required to have the same shape.
    """
    result = np.zeros(matrices[0].shape, dtype=np.uint8)
    if mergeColor==None:
        for i,j in np.ndindex(matrices[0].shape):
            done=False
            for m in matrices:
                if m[i,j]!=backgroundColor:
                    result[i,j] = m[i,j]
                    done=True
                    break
            if not done:
                result[i,j] = backgroundColor
    else:
        for i,j in np.ndindex(matrices[0].shape):
            colors = set()
            for m in matrices:
                if m[i,j]!=backgroundColor:
                    colors.add(m[i,j])
            if len(colors)==0:
                result[i,j] = backgroundColor
            elif len(colors)==1:
                result[i,j] = next(iter(colors))
            else:
                result[i,j] = mergeColor
    return result

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
    model = LinearModel(t.inShape, t.outShape, nChannels)
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
    model = LinearModelDummy(t.inShape, t.outShape)
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
    model = SimpleLinearModel(nInFeatures, len(colors))
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
            if shape.position[0]+i<matrix.shape[0] and shape.position[1]+j<matrix.shape[1]\
                    and shape.position[0]+i >= 0 and shape.position[1]+j >= 0:
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

def symmetrizeSubmatrix(matrix, ud=False, lr=False, rotation=False, newColor=None, subShape=None):
    """
    Given a Task.Matrix, make the non-background part symmetric
    """
    m = matrix.m.copy()
    bC = matrix.backgroundColor
    if np.all(m == bC):
        return m
    x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
    while x1 <= x2 and np.all(m[x1,:] == bC):
        x1 += 1
    while x2 >= x1 and np.all(m[x2,:] == bC):
        x2 -= 1
    while y1 <= y2 and np.all(m[:,y1] == bC):
        y1 += 1
    while y2 >= y1 and np.all(m[:,y2] == bC):
        y2 -= 1
    subMat = m[x1:x2+1,y1:y2+1].copy()
    
    if subShape == None:
        symList = []
        if ud:
            symList.append(np.flipud(subMat.copy()))
            if lr:
                symList.append(np.fliplr(subMat.copy()))
                symList.append(np.fliplr(np.flipud(subMat.copy())))
            elif lr:
                symList.append(np.fliplr(subMat.copy()))
        elif rotation:
            for x in range(1,4):
                symList.append(np.rot90(subMat.copy(),x))
        for newSym in symList:
            score, bestScore = 0, 0
            bestX, bestY = 0, 0
            for i, j in np.ndindex((m.shape[0]-newSym.shape[0]+1,m.shape[1]-newSym.shape[1]+1)):
                score = np.count_nonzero(np.logical_or(m[i:i+newSym.shape[0],j:j+newSym.shape[1]] == newSym, newSym == bC))
                if score > bestScore:
                    bestX, bestY = i, j
                    bestScore = score
            for i, j in np.ndindex(newSym.shape):
                if newSym[i,j] != bC:
                    if newColor == None:
                        m[bestX+i, bestY+j] = newSym[i,j]
                    else:
                        if m[bestX+i,bestY+j] == bC:
                           m[bestX+i, bestY+j] = newColor 
    else:
        found = False
        for x in range(subMat.shape[0]-subShape.shape[0]+1):
            for y in range(subMat.shape[1]-subShape.shape[1]+1):
                #if np.all(m[x1+x:x1+x+subShape.shape[0],y1+y:y1+y+subShape.shape[1]]==subShape.m):
                if np.all(np.equal(m[x1+x:x1+x+subShape.shape[0],y1+y:y1+y+subShape.shape[1]] == bC,subShape.m == 255)):
                    found = True   
                    break
            if found:
                break
        if not found:
            return m
        if ud and lr:
            if 2*x+x1+subShape.shape[0] > m.shape[0] or 2*y+y1+subShape.shape[0]> m.shape[1]:
                return m
            for i in range(subMat.shape[0]):
                for j in range(subMat.shape[1]):
                    if subMat[i][j] != bC:
                        m[2*x+x1+subShape.shape[0]-i-1,y1+j] = subMat[i,j]
                        m[x1+i,2*y+y1+subShape.shape[0]-j-1] = subMat[i,j]
                        m[2*x+x1+subShape.shape[0]-i-1,2*y+y1+subShape.shape[0]-j-1] = subMat[i,j]
        elif rotation:
            if x1+y+x+subShape.shape[0] > m.shape[0] or y1+x+y+subShape.shape[1] > m.shape[1]\
                or y1+y-x+subMat.shape[0] >= m.shape[0] or x1+x-y+subMat.shape[1] >= m.shape[1]\
                or x1+2*x+subShape.shape[0] > m.shape[0] or y1+2*y+subShape.shape[0] > m.shape[1]:
                return m
            for i in range(subMat.shape[0]):
                for j in range(subMat.shape[1]):
                    if subMat[i,j] != bC:
                        m[x1+x+subShape.shape[0]+y-j-1,y1+y-x+i] = subMat[i,j]
                        m[x1+x-y+j,y1+y+subShape.shape[0]+x-i-1] = subMat[i,j]
                        m[x1+2*x+subShape.shape[0]-i-1,y1+2*y+subShape.shape[0]-j-1] = subMat[i,j]        
    return m

def getBestSymmetrizeSubmatrix(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    rotation, lr, ud = False, False, False
    croppedSamples = [cropAllBackground(s.outMatrix) for s in t.trainSamples]
    if all(np.all(np.flipud(m)==m) for m in croppedSamples):
        lr = True
    if all(np.all(np.fliplr(m)==m) for m in croppedSamples):    
        ud = True    
    if all(m.shape[0]==m.shape[1] and np.all(np.rot90(m)==m) for m in croppedSamples):
        rotation = True
    for sh in t.commonInDShapes:
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeSubmatrix,\
                                                        lr=lr,ud=ud,rotation=rotation,subShape=sh), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeSubmatrix,lr=lr,\
                                                            ud=ud,rotation=rotation), bestScore, bestFunction)
    return bestFunction

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

def revertColorOrder(matrix):
    m = matrix.m.copy()
    colors = [color for color,count in sorted(matrix.colorCount.items(), key=lambda item:item[1])]
    colorDict = {}
    for i in range(len(colors)):
        colorDict[colors[i]] = colors[len(colors)-i-1]
    for i,j in np.ndindex(m.shape):
        m[i,j] = colorDict[m[i,j]]
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
            if (matrix.shapes[sh].color in fixedColors):# or \
            #(matrix.shapes[sh].hasFeatures(fixedShapeFeatures)):
                continue
            if hasFeatures(featureList[sh], ccwf[color]):
                m = changeColorShapes(m, [matrix.shapes[sh]], color)
                #break
    return m


# %% Change pixels with features

def pixelRecolor(t):
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
    
def executePixelRecolor(matrix, Best_Dict, Best_v, Best_Q1, Best_Q2):
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

# %% Extend Color

def extendColor(matrix, direction, cic, fixedColors, color=None, sourceColor=None,\
                deleteExtensionColors=set(), deleteIfBorder=False, \
                breakAtFixedColor=False, mergeColor=None):
    #m = matrix.m.copy()
    
    if sourceColor==None:
        sourceColor=color
        
    matrices = []
    
    # Vertical
    if direction=='all' or direction=='hv' or direction=='v' or direction=='u':
        for j in range(matrix.shape[1]):
            colorCells=False
            start = matrix.shape[0]-1
            for i in reversed(range(matrix.shape[0])):
                if color==None and sourceColor==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = i
                if colorCells:
                    if matrix.m[i,j] in cic:
                        if color==None:
                            m[i,j] = sourceColor
                        else:
                            m[i,j] = color
                    elif m[i,j]!=sourceColor and breakAtFixedColor=="any":
                        sourceColor = m[i,j]
                    elif matrix.m[i,j] in fixedColors and breakAtFixedColor:
                        break
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   i==0 and deleteIfBorder):
                    currentI = i
                    for i in range(currentI, start):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
    if direction=='all' or direction=='hv' or direction=='v' or direction=='d':
        for j in range(matrix.shape[1]):
            colorCells=False
            start = 0
            for i in range(matrix.shape[0]):
                if color==None and sourceColor==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = i
                if colorCells and matrix.m[i,j] in fixedColors and breakAtFixedColor:
                    break
                if colorCells:
                    if matrix.m[i,j] in cic:
                        if color==None:
                            m[i,j] = sourceColor
                        else:
                            m[i,j] = color
                    elif m[i,j]!=sourceColor and breakAtFixedColor=="any":
                        sourceColor = m[i,j]
                    elif matrix.m[i,j] in fixedColors and breakAtFixedColor:
                        break
                        
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   i==m.shape[0]-1 and deleteIfBorder):
                    currentI = i+1
                    for i in reversed(range(start, currentI)):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
             
    # Horizontal
    if direction=='all' or direction=='hv' or direction=='h' or direction=='l':
        for i in range(matrix.shape[0]):
            colorCells=False
            start = matrix.shape[1]-1
            for j in reversed(range(matrix.shape[1])):
                if color==None and sourceColor==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = j
                if colorCells:
                    if matrix.m[i,j] in cic:
                        if color==None:
                            m[i,j] = sourceColor
                        else:
                            m[i,j] = color
                    elif m[i,j]!=sourceColor and breakAtFixedColor=="any":
                        sourceColor = m[i,j]
                    elif matrix.m[i,j] in fixedColors and breakAtFixedColor:
                        break
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   j==0 and deleteIfBorder):
                    currentJ = j
                    for j in range(currentJ, start):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
    if direction=='all' or direction=='hv' or direction=='h' or direction=='r':
        for i in range(matrix.shape[0]):
            colorCells=False
            start = 0
            for j in range(matrix.shape[1]):
                if color==None and sourceColor==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = j
                if colorCells: 
                    if matrix.m[i,j] in cic:
                        if color==None:
                            m[i,j] = sourceColor
                        else:
                            m[i,j] = color
                    elif m[i,j]!=sourceColor and breakAtFixedColor=="any":
                        sourceColor = m[i,j]
                    elif matrix.m[i,j] in fixedColors and breakAtFixedColor:
                        break
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   j==m.shape[1]-1 and deleteIfBorder):
                    currentJ = j+1
                    for j in reversed(range(start, currentJ)):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
              
    if len(matrices)>0:
        m = mergeMatrices(matrices, next(iter(cic)), mergeColor=mergeColor)
    else:
        m = matrix.m.copy()
                
    # Diagonal
    if direction=='all' or  direction=='diag' or direction=='d1' or direction=='d2':
        if direction=='diag' or direction=='all':
            directions = ['d1', 'd2']
        else:
            directions = [direction]
        for direction in directions:
            for transpose in [True, False]:
                if transpose and direction=='d1':
                    matrix.m = np.rot90(matrix.m, 2).T
                    m = np.rot90(m, 2).T
                if transpose and direction=='d2':
                    matrix.m = matrix.m.T
                    m = m.T
                if direction=='d2':
                    matrix.m = np.fliplr(matrix.m)
                    m = np.fliplr(m)
                for i in range(-matrix.shape[0]+1, matrix.shape[1]):
                    diag = np.diagonal(matrix.m, i)
                    colorCells=False
                    for j in range(len(diag)):
                        if color==None and sourceColor==None:
                            if i<=0:
                                if matrix.m[-i+j,j] not in (fixedColors|cic):
                                    sourceColor = matrix.m[-i+j,j]
                            else:
                                if matrix.m[j,i+j] not in (fixedColors|cic):
                                    sourceColor = matrix.m[j,i+j]
                        if i<=0:
                            if matrix.m[-i+j,j]==sourceColor:
                                colorCells=True
                            if colorCells:
                                if matrix.m[-i+j,j] in cic:
                                    if color==None:
                                        m[-i+j,j] = sourceColor
                                    else:
                                        m[-i+j,j] = color
                                elif matrix.m[-i+j,j]!=sourceColor and breakAtFixedColor=="any":
                                    sourceColor = m[-i+j,j]
                                elif matrix.m[-i+j,j] in fixedColors and breakAtFixedColor:
                                    break
                            if colorCells and ((matrix.m[-i+j,j] in deleteExtensionColors) or \
                                               j==len(diag)-1 and deleteIfBorder):
                                for j in range(len(diag)):
                                    m[-i+j,j] = matrix.m[-i+j,j]
                                break
                        else:
                            if matrix.m[j,i+j]==sourceColor:
                                colorCells=True
                            if colorCells:
                                if matrix.m[j,i+j] in cic:
                                    if color==None:
                                        m[j,i+j] = sourceColor
                                    else:
                                        m[j,i+j] = color
                                elif matrix.m[j,i+j]!=sourceColor and breakAtFixedColor=="any":
                                    sourceColor = m[j,i+j]
                                elif matrix.m[j,i+j] in fixedColors and breakAtFixedColor:
                                    break
                            if colorCells and ((matrix.m[j,i+j] in deleteExtensionColors) or \
                                               j==len(diag)-1 and deleteIfBorder):
                                for j in range(len(diag)):
                                    m[j,i+j] = matrix.m[j,i+j]
                                break
                    if color==None:
                        sourceColor=None
                if direction=='d2':
                    matrix.m = np.fliplr(matrix.m)
                    m = np.fliplr(m)
                if transpose and direction=='d2':
                    matrix.m = matrix.m.T
                    m = m.T
                if transpose and direction=='d1':
                    matrix.m = np.rot90(matrix.m, 2).T
                    m = np.rot90(m, 2).T

    return m    

def getBestExtendColor(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    
    mergeColors = t.commonOutColors-t.totalInColors
    if len(mergeColors) == 1:
        mergeColor = next(iter(mergeColors))
    else:
        mergeColor = None
    
    cic = t.commonChangedInColors
    if len(cic)==0:
        return bestFunction
    fixedColors = t.fixedColors
    for d in ['h', 'v']:#['r', 'l', 'h', 'u', 'd', 'v', 'hv', 'd1', 'd2', 'diag', 'all']:
        for dib,bafc in product([True, False], [True, False, "any"]):
            f = partial(extendColor, direction=d, cic=cic, fixedColors=fixedColors,\
                        deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(extendColor, direction=d, cic=cic, fixedColors=fixedColors,\
                        deleteExtensionColors=fixedColors,\
                        deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            for coc in t.commonChangedOutColors:    
                f = partial(extendColor, color=coc, direction=d, cic=cic, fixedColors=fixedColors,\
                            deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(extendColor, color=coc, direction=d, cic=cic, fixedColors=fixedColors,\
                            deleteExtensionColors=fixedColors,\
                            deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                for fc in t.fixedColors:
                    f = partial(extendColor, color=coc, direction=d, cic=cic, sourceColor=fc, \
                                fixedColors=fixedColors,\
                                deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                    f = partial(extendColor, color=coc, direction=d, cic=cic, sourceColor=fc, \
                                fixedColors=fixedColors, deleteExtensionColors=fixedColors,\
                                deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                
    return bestFunction

# %% Fill rectangleInside
def fillRectangleInside(matrix, rectangleColor, fillColor):
    m = matrix.m.copy()
    for shape in matrix.shapes:
        if shape.isRectangle and shape.color==rectangleColor:
            if shape.shape[0] > 2 and shape.shape[1] > 2:
                rect = np.full((shape.shape[0]-2, shape.shape[1]-2), fillColor, dtype=np.uint8)
                m[shape.position[0]+1:shape.position[0]+shape.shape[0]-1,\
                  shape.position[1]+1:shape.position[1]+shape.shape[1]-1] = rect
    return m
    
# %% Move shapes    

def moveShape(matrix, shape, background, direction, until = -1, nSteps = 100, \
              keepOriginal=False):
    """
    'direction' can be l, r, u, d, ul, ur, dl, dr
    (left, right, up, down, horizontal, vertical, diagonal1, diagonal2)
    'until' can be a color or -1, which will be interpreted as border
    If 'until'==-2, then move until the shape encounters anything
    """
    m = matrix.copy()
    if not keepOriginal:
        m = changeColorShapes(m, [shape], background)
    s = copy.deepcopy(shape)
    if nSteps=="shapeX":
        nSteps = shape.shape[0]
    if nSteps=="shapeY":
        nSteps = shape.shape[1]
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
    
def moveAllShapes(matrix, background, direction, until, nSteps=100, color=None, \
                  keepOriginal=False):
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
    if len(shapesToMove) > 15:
        return m
    for s in shapesToMove:
        newMatrix = m.copy()
        if direction == "any":
            for d in ['l', 'r', 'u', 'd', 'ul', 'ur', 'dl', 'dr']:
                newMatrix = moveShape(m, s, background, d, until, keepOriginal=keepOriginal)
                if not np.all(newMatrix == m):
                    return newMatrix
                    break
        else:
            m = moveShape(m, s, background, direction, until, nSteps, keepOriginal=keepOriginal)
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
    if len(matrix.shapes) > 25:
        return m
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

def getBestMoveShapes(t, candidate):
    """
    This functions tries to find, for a given task t, the best way to move
    shapes.
    """
    directions = ['l', 'r', 'u', 'd', 'any']#, 'ul', 'ur', 'dl', 'dr', 'any']
    bestScore = 1000
    bestFunction = partial(identityM)
    
    doSingleColor = all([len(s.inMatrix.shapes) < 50 for s in t.trainSamples])
    doDSingleColor = all([len(s.inMatrix.dShapes) < 15 for s in t.trainSamples])
    doMulticolor = all([len(s.inMatrix.multicolorShapes) < 15 for s in t.trainSamples])
        
    # Move all shapes in a specific direction, until a non-background thing is touched
    for d in directions:
        if doSingleColor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="singleColor")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        if doDSingleColor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="diagonalSingleColor")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        if doMulticolor:
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
        
    if doSingleColor:
        colorsToChange = list(t.colors - t.fixedColors - set({t.backgroundColor}))
        ctc = [[c] for c in colorsToChange] + [colorsToChange] # Also all colors
        for c in ctc:
            for d in directions:
                moveUntil = colorsToChange + [-1] + [-2] #Border, any
                for u in moveUntil:
                    f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                    direction=d, until=u)
                    bestFunction, bestScore, isPerfect = updateBestFunction(t, f, bestScore, bestFunction, \
                                                                            checkPerfect=True, prevScore=candidate.score)
                    if isPerfect:
                        return bestFunction
                    if bestScore==0:
                        return bestFunction
                for nSteps in range(1, 5):
                    f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                direction=d, until=-2, nSteps=nSteps)
                    bestFunction, bestScore, isPerfect = updateBestFunction(t, f, bestScore, bestFunction,\
                                                                            checkPerfect=True, prevScore=candidate.score)
                    if isPerfect:
                        return bestFunction
                    if bestScore==0:
                        return bestFunction
                    f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                direction=d, until=-2, nSteps=nSteps, keepOriginal=True)
                    bestFunction, bestScore, isPerfect = updateBestFunction(t, f, bestScore, bestFunction,\
                                                                            checkPerfect=True, prevScore=candidate.score)
                    if isPerfect:
                        return bestFunction
                    if bestScore==0:
                        return bestFunction
                
    
        if t.backgroundColor != -1 and hasattr(t, 'fixedColors'):
            colorsToMove = t.almostCommonColors - set([t.backgroundColor]) - t.fixedColors
            for ctm in colorsToMove:
                for uc in t.unchangedColors:
                    f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                                     background=t.backgroundColor, until=uc)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                    
                    f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                                background=t.backgroundColor, until=uc, diagonals=True)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                
        if all([len(sample.fixedShapes)>0 for sample in t.trainSamples]):
            f = partial(moveAllShapesToClosest, background=t.backgroundColor,\
                        fixedShapeFeatures = t.fixedShapeFeatures)
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

# %% Connect Pixels

def connectPixels(matrix, pixelColor=None, connColor=None, fixedColors=set(),\
                  allowedChanges={}, lineExclusive=False, diagonal=False):
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
    
    # Diagonal         
    if diagonal:
        for d in [1, 2]:
            if d==2:
                matrix = np.fliplr(matrix)
                m = np.fliplr(m)
            for i in range(-matrix.shape[0]-1, matrix.shape[1]):
                diag = np.diagonal(matrix, i)
                lowLimit = 0
                while lowLimit < len(diag) and diag[lowLimit] != pixelColor:
                    lowLimit += 1
                lowLimit += 1
                upLimit = len(diag)-1
                while upLimit > lowLimit and diag[upLimit] != pixelColor:
                    upLimit -= 1
                if upLimit > lowLimit:
                    if lineExclusive:
                        for j in range(lowLimit, upLimit):
                            if i<=0:
                                if matrix[-i+j,j] == pixelColor:
                                    lowLimit = upLimit
                                    break
                            else:
                                if matrix[j,i+j] == pixelColor:
                                    lowLimit = upLimit
                                    break
                    for j in range(lowLimit, upLimit):
                        if i<=0:
                            if connColor != None:
                                if matrix[-i+j,j] != pixelColor and matrix[-i+j,j] not in fixedColors:
                                    m[-i+j,j] = connColor
                            else:
                                if matrix[-i+j,j] in allowedChanges.keys():
                                    m[-i+j,j] = allowedChanges[matrix[-i+j,j]]
                        else:
                            if connColor != None:
                                if matrix[j,i+j] != pixelColor and matrix[j,i+j] not in fixedColors:
                                    m[j,i+j] = connColor
                            else:
                                if matrix[j,i+j] in allowedChanges.keys():
                                    m[j,i+j] = allowedChanges[matrix[j,i+j]]
            if d==2:
                matrix = np.fliplr(matrix)
                m = np.fliplr(m)

    return m

def connectAnyPixels(matrix, pixelColor=None, connColor=None, fixedColors=set(),\
                     allowedChanges={}, lineExclusive=False, diagonal=False):
    m = matrix.m.copy()
    if pixelColor==None:
        if connColor==None:
            for c in matrix.colors - set([matrix.backgroundColor]):
                m = connectPixels(m, c, c, lineExclusive=lineExclusive, diagonal=diagonal)
            return m
        else:
            for c in matrix.colors - set([matrix.backgroundColor]):
                m = connectPixels(m, c, connColor, lineExclusive=lineExclusive, diagonal=diagonal)
            return m
    else:
        if len(allowedChanges)>0:
            m = connectPixels(m, pixelColor, allowedChanges=allowedChanges,\
                              lineExclusive=lineExclusive, diagonal=diagonal)
        else:
            m = connectPixels(m, pixelColor, connColor, fixedColors, lineExclusive=lineExclusive,\
                              diagonal=diagonal)
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

def flipShape(matrix, shape, axis, background):
    # Axis can be lr, ud
    m = matrix.copy()
    smallM = np.ones((shape.shape[0], shape.shape[1]), dtype=np.uint8) * background
    for c in shape.pixels:
        smallM[c] = shape.color
    if axis == "lr":
        smallM = np.fliplr(smallM)
    if axis == "ud":
        smallM = np.flipud(smallM)
    for i,j in np.ndindex(smallM.shape):
        m[shape.position[0]+i, shape.position[1]+j] = smallM[i,j]
    return m

def flipAllShapes(matrix, axis, color, background, byColor=False, diagonal=False):#, multicolor=False):
    m = matrix.m.copy()
    if byColor:
        shapesToMirror = [s for s in matrix.shapesByColor if s.color in color]
    #elif multicolor:
    #    if diagonal:
    #        shapesToMirror = [s for s in matrix.multicolorDShapes]
    #    else:
    #        shapesToMirror = [s for s in matrix.multicolorShapes]
    else:
        if diagonal:
            shapesToMirror = [s for s in matrix.dShapes if s.color in color]
        else:
            shapesToMirror = [s for s in matrix.shapes if s.color in color]
    for s in shapesToMirror:
        m = flipShape(m, s, axis, background)
    return m

def getBestFlipAllShapes(t):
    bestFunction = partial(identityM)
    if t.backgroundColor == -1:
        return bestFunction
    bestScore = 1000
    colors = set([0,1,2,3,4,5,6,7,8,9]) - set([t.backgroundColor])
    for d in ["lr", "ud"]:
        #for multicolor in [True, False]:
        for diagonal in [True, False]:
            bestFunction, bestScore = updateBestFunction(t, partial(flipAllShapes, axis=d, color=colors,\
                                    background=t.backgroundColor, diagonal=diagonal), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(flipAllShapes, axis=d, color=colors,\
                                background=t.backgroundColor, byColor=True), bestScore, bestFunction)
    return bestFunction

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
        matrices.append(Matrix(m))
        
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
    
def extendMatrix(matrix, color, position="tl", xShape=None, yShape=None, isSquare=False, goodDimension=None):
    """
    Given a matrix, xShape(>matrix.shape[0]), yShape(>matrix.shape[1]) and a color,
    this function extends the matrix using the dimensions given by xShape and
    yShape by coloring the extra pixels with the given color.
    If xShape or yShape are not given, then they will be equal to matrix.shape[0]
    of matrix.shape[1], respectively.
    The position of the input matrix in the output matrix can be given by
    specifying "tl", "tr", "bl" or "br" (top-left/top-right/bot-left/bot-right).
    The default is top-left.
    """
    if isSquare:
        if goodDimension=='x':
            xShape = matrix.shape[0]
            yShape = matrix.shape[0]
        if goodDimension=='y':
            xShape = matrix.shape[1]
            yShape = matrix.shape[1]
    if xShape==None:
        xShape = matrix.shape[0]
    if yShape==None:
        yShape = matrix.shape[1]
    m = np.full((xShape, yShape), color, dtype=np.uint8)
    if position=="tl":
        m[0:matrix.shape[0], 0:matrix.shape[1]] = matrix.m.copy()
    elif position=="tr":
        m[0:matrix.shape[0], yShape-matrix.shape[1]:yShape] = matrix.m.copy()
    elif position=="bl":
        m[xShape-matrix.shape[0]:xShape, 0:matrix.shape[1]] = matrix.m.copy()
    else:
        m[xShape-matrix.shape[0]:xShape, yShape-matrix.shape[1]:yShape] = matrix.m.copy()
    
    return m

def getBestExtendMatrix(t):
    """
    If t.backgroundColor!=-1 and if it makes sense.
    """
    if t.backgroundColor==-1:
        totalColorCount = Counter()
        for s in t.trainSamples:
            totalColorCount += s.inMatrix.colorCount
        background = max(totalColorCount.items(), key=operator.itemgetter(1))[0]
    else:
        background=t.backgroundColor
    bestScore = 1000
    bestFunction = partial(identityM)
    
    # Define xShape and yShape:
    xShape = None
    yShape = None
    isSquare=False
    goodDimension=None
    
    # If the outShape is given, easy
    if t.sameOutShape:
        xShape=t.outShape[0]
        yShape=t.outShape[1]
    # If the output xShape is always the same and the yShape keeps constant, pass the common xShape
    elif len(set([s.outMatrix.shape[0] for s in t.trainSamples]))==1:
        if all([s.outMatrix.shape[1]==s.inMatrix.shape[1] for s in t.trainSamples]):
            xShape=t.trainSamples[0].outMatrix.shape[0]  
    # If the output yShape is always the same and the xShape keeps constant, pass the common yShape
    elif len(set([s.outMatrix.shape[1] for s in t.trainSamples]))==1:
        if all([s.outMatrix.shape[0]==s.inMatrix.shape[0] for s in t.trainSamples]):
            yShape=t.trainSamples[0].outMatrix.shape[1] 
    # If the matrix is always squared, and one dimension (x or y) is fixed, do this:
    elif all([s.outMatrix.shape[0]==s.outMatrix.shape[1] for s in t.trainSamples]):
        isSquare=True
        if all([s.outMatrix.shape[1]==s.inMatrix.shape[1] for s in t.trainSamples]):
            goodDimension='y'     
        elif all([s.outMatrix.shape[0]==s.inMatrix.shape[0] for s in t.trainSamples]):
            goodDimension='x'      
    
    for position in ["tl", "tr", "bl", "br"]:
        f = partial(extendMatrix, color=background, xShape=xShape, yShape=yShape,\
                    position=position, isSquare=isSquare, goodDimension=goodDimension)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction
    
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
def getBestMultiplyMatrix(t, falseColor):  
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
    
    # Conditions
    def trueCondition(matrix, pixel):
        return True
    def maxColor(matrix, pixel):
        x = [k for k, v in sorted(matrix.colorCount.items(), key=lambda item: item[1], reverse=True)]
        if len(x)<2 or matrix.colorCount[x[0]]!=matrix.colorCount[x[1]]:
            return pixel==max(matrix.colorCount, key=matrix.colorCount.get)
        else:
            return False
    def minColor(matrix,pixel):
        x = [k for k, v in sorted(matrix.colorCount.items(), key=lambda item: item[1])]
        if len(x)<2 or matrix.colorCount[x[-1]]!=matrix.colorCount[x[-2]]:
            return pixel==min(matrix.colorCount, key=matrix.colorCount.get)
        else:
            return False
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

def doBestMultiplyMatrix(matrix, opCond, falseColor):
    factor = matrix.shape
    m = np.full(tuple(s * f for s, f in zip(matrix.shape, factor)), falseColor, dtype=np.uint8)
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
        m = Matrix(mat)
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
        m = Matrix(mat)
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
        m = Matrix(mat)
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
        m = Matrix(mat)
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

def colorAppearingXTimes(matrix, times):
    m = np.zeros(matrix.grid.shape, dtype=np.uint8)
    for i,j in np.ndindex(matrix.grid.shape):
        for k,v in matrix.grid.cells[i][j][0].colorCount.items():
            if v==times:
                m[i,j] = k
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

# %% crop all shapes
    
def cropAllShapes(matrix, background, diagonal=False):
    if diagonal:
        shapes = [shape for shape in matrix.dShapes if shape.color!=background]
    else:
        shapes = [shape for shape in matrix.shapes if shape.color!=background]
    shapes = sorted(shapes, key=lambda x: x.nPixels, reverse=True)
    
    if len(shapes)==0:
        return matrix.m.copy()
    
    m = shapes[0].m.copy()
    for i,j in np.ndindex(m.shape):
        if m[i,j]==255:
            m[i,j] = background
    
    outMatrix = Matrix(m)
    if diagonal:
        outShapes = [shape for shape in outMatrix.dShapes if shape.color==background]
    else:
        outShapes = [shape for shape in outMatrix.shapes if shape.color==background]
    
    for s in outShapes:
        if s.color==background:
            for shape in shapes:
                if shape.hasSameShape(s):
                    m = changeColorShapes(m, [s], shape.color)
                    break
    return m

# %% Stuff added by Roderic

def getBestSymmetrizeAllShapes(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    for cc in set.intersection(*t.inColors).union(set([-1])):
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc, byColor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc, context=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc, context=True, byColor=True), bestScore, bestFunction)

    return bestFunction

def symmetrizeAllShapes(matrix, diagonal=True, multicolor=True, targetColor=-1, axis=None,\
                        context=False, lr = True, ud = True, byColor=False):
    m = matrix.m.copy()
    bC = matrix.backgroundColor
    if byColor:
        shList = [sh for sh in matrix.shapesByColor if (sh.shape[0]<m.shape[0] and sh.shape[1]<m.shape[1])]
    else:    
        if not multicolor: 
            if diagonal:   
                shList = [sh for sh in matrix.dShapes]
            else:   
                shList = [sh for sh in matrix.shapes]
        else:
            if diagonal: 
                shList = [sh for sh in matrix.multicolorDShapes]
            else:
                shList = [sh for sh in matrix.multicolorShapes]
    if targetColor > -1:
        shList = [sh for sh in shList if hasattr(sh, 'color') and sh.color == targetColor]
    for sh in shList:
        if context:
            shM = m[sh.position[0]:sh.position[0]+sh.shape[0], sh.position[1]:sh.position[1]+sh.shape[1]]
        else:
            shM = sh.m.copy()
        if lr:
            shMlr = np.fliplr(shM)
            for i,j in np.ndindex(sh.shape):
                if shM[i,j] == bC or shM[i,j] == 255:
                    shM[i,j] = shMlr[i,j]
        if ud:
            shMud = np.flipud(shM)
            for i,j in np.ndindex(sh.shape):
                if shM[i,j] == bC or shM[i,j] == 255:
                    shM[i,j] = shMud[i,j]
        if context:
            m[sh.position[0]:sh.position[0]+sh.shape[0], sh.position[1]:sh.position[1]+sh.shape[1]] = shM
        else:
            newInsert = copy.deepcopy(sh)
            newInsert.m = shM
            m = insertShape(m, newInsert)
    return m

#paint grids
def paintGridLikeBackground(matrix):
    """
    Ignores the grid by paiting it in the background color (most repeated color or second if first coincides with grid color).
    In some cases cells have more than one color but it is still worth ignoring the grid. 
    """
    m = matrix.m.copy()
    bC = max(matrix.colorCount,key=matrix.colorCount.get)
    if matrix.isGrid:
        m[m==matrix.grid.color] = bC
    elif matrix.isAsymmetricGrid:
        m[m==matrix.asymmetricGrid.color] = bC
    return m

#HOW DO I PASS THE ARGUMENTS????
def paintGridLikeOriginal(matrix, grid):
    """
    Repaint a grid previously painted in the background color.
    """
    m = matrix.m.copy()
    m = insertShape(m, grid)
    return m   

def downsizeMode(matrix, newShape, falseColor=None):
    """
    Given a matrix and a shape, this function returns a new matrix with the
    given shape. The elements of the return matrix are given by the colors of 
    each of the submatrices. If a submatrix has more than one color the mode is
    chosen.
    """
    if (matrix.shape[0]%newShape[0])!=0 or (matrix.shape[1]%newShape[1])!=0:
        return matrix.m.copy()
    xBlock = int(matrix.shape[0]/newShape[0])
    yBlock = int(matrix.shape[1]/newShape[1])
    m = np.full(newShape, matrix.backgroundColor, dtype=np.uint8)
    for i,j in np.ndindex(newShape[0], newShape[1]):
        colorCount = Counter(matrix.m[i*xBlock: (i+1)*xBlock, j*yBlock: (j+1)*yBlock].flatten())
        m[i,j] = max(colorCount, key=colorCount.get)
    return m    

def isDeleteTask(t):    
    if hasattr(t, 'colorChanges') and t.backgroundColor in [c[1] for c in t.colorChanges]:
        return True
    return False

def getBestDeleteShapes(t, multicolor=False, diagonal=True):
    attrs = set(['LaSh','SmSh','MoCl','MoCo','PiXl'])
    bestScore = 1000
    bestFunction = partial(identityM)
    for attr in attrs:
        bestFunction, bestScore = updateBestFunction(t, partial(deleteShapes, diagonal=diagonal, multicolor=multicolor,attributes=set([attr])), bestScore, bestFunction)
    return bestFunction

def getDeleteAttributes(t, diagonal=True):
    bC = max(0, t.backgroundColor)
    if diagonal:
        if t.nCommonInOutDShapes == 0:
            return set()
        attrs = set.union(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=True)[s.inMatrix.dShapes.index(sh[0])]\
                    for s in t.trainSamples for sh in s.commonDShapes])
        nonAttrs = set()
        c = 0
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=True)
            for shi in range(len(s.inMatrix.shapes)):
                if any(s.inMatrix.dShapes[shi] == sh2[0] for sh2 in s.commonDShapes) or s.inMatrix.dShapes[shi].color == bC:
                    continue
                else:
                    if c == 0:
                        nonAttrs = shAttrs[shi]
                        c += 1
                    else:
                        nonAttrs = nonAttrs.intersection(shAttrs[shi])
                        c += 1
    else:
        if t.nCommonInOutShapes == 0:
            return set()
        attrs = set.union(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=False)[s.inMatrix.shapes.index(sh[0])]\
                    for s in t.trainSamples for sh in s.commonShapes])
        nonAttrs = set()
        c = 0
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=False)
            for shi in range(len(s.inMatrix.shapes)):
                if any(s.inMatrix.shapes[shi] == sh2[0] for sh2 in s.commonShapes) or s.inMatrix.shapes[shi].color == bC:
                    continue
                else:
                    if c == 0:
                        nonAttrs = shAttrs[shi]
                        c += 1
                    else:
                        nonAttrs = nonAttrs.intersection(shAttrs[shi])
                        c += 1
    return set(nonAttrs - attrs)

def deleteShapes(matrix, attributes, diagonal, multicolor):
    m = matrix.m.copy()
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes]
        else:   
            shList = [sh for sh in matrix.shapes]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    attrList = matrix.getShapeAttributes(matrix.backgroundColor, not multicolor, diagonal)
    for shi in range(len(shList)):
        if len(attrList[shi].intersection(attributes)) > 0:
            m = deleteShape(m, shList[shi], matrix.backgroundColor)
    return m

def getBestArrangeShapes(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    if hasattr(t, 'outShape'):
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes, outShape=t.outShape,\
                                                            diagonal=True, multicolor=False), bestScore, bestFunction)
    elif t.outIsInMulticolorShapeSize:
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes, diagonal=True,\
                                                                multicolor=True,outShape='LaSh'), bestScore, bestFunction)
    else:
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes,shByColor=True,outShape='LaSh'), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes,shByColor=True, fullFrames=True,outShape='LaSh'), bestScore, bestFunction)
    return bestFunction

def arrangeShapes (matrix, outShape = None, multicolor=True, diagonal=True, shByColor=False,\
                   fullFrames=False, outDummyMatrix=None, outDummyColor=0):
    def completeFrames(shape,rotate=False,fill=False):
        'version of symmetrize shape intended for frame-like shapes' 
        m = shape.m.copy()
        if m.shape[0]>m.shape[1]:   
            newm = np.full((m.shape[0], m.shape[0]), fill_value=255)
            sideL = m.shape[0]
        else:
            newm = np.full((m.shape[1], m.shape[1]), fill_value=255)
            sideL = m.shape[1]
        for i,j in np.ndindex(m.shape):
            if m[i,j] != 255:
                if newm[i,j] == 255:
                    newm[i,j] = m[i,j]
                if newm[sideL - j - 1,i] == 255:
                    newm[sideL - j - 1,i] = m[i,j]
                if newm[sideL - i - 1,sideL - j - 1] == 255:
                    newm[sideL - i - 1,sideL - j - 1] = m[i,j]
                if newm[j,sideL - i - 1] == 255 and m[i,j]:
                    newm[j,sideL - i - 1] = m[i,j]
        newSh = copy.deepcopy(shape)
        newSh.m = newm
        newSh.shape = newm.shape
        return newSh
    
    def tessellateShapes (mat, shL, n, bC, rotation=False):
        m = mat.copy()
        arrFound = False
        rot = 1
        """
        Attempts to tessellate matrix mat with background color bC shapes is list sh.
        """
        if rotation:
            rot = 4
            if len(shL[n].pixels)==1:
                rot = 1
            if len(shL[n].pixels)==2:
                rot = 2
        for x in range(rot):
            sh = copy.deepcopy(shL[n])
            sh.m = np.rot90(sh.m,x).copy()
            sh.shape = sh.m.shape
            if mat.shape[0] < sh.shape[0] or mat.shape[1] < sh.shape[1]:
                continue
            for i, j in np.ndindex(tuple(mat.shape[k] - sh.shape[k] + 1 for k in (0,1))):
                if np.all(np.logical_or(m[i: i+sh.shape[0], j: j+sh.shape[1]] == bC, sh.m == 255)):
                    for k, l in np.ndindex(sh.shape):
                        if sh.m[k,l] != 255:
                            m[i+k,j+l] = sh.m[k,l]
                    if n == len(shL) - 1:
                        return m, True
                    m, arrFound = tessellateShapes(m, shL, n+1, bC, rotation)
                    if arrFound:
                        return m, True
                    if not arrFound:
                        for k, l in np.ndindex(sh.shape):
                            if sh.m[k,l] != 255:
                                m[i+k,j+l] = bC
        return m, False
    if shByColor:
        shList = [sh for sh in matrix.shapesByColor]
        if fullFrames:
            shList = [completeFrames(sh) for sh in matrix.shapesByColor]
    else:
        if not multicolor: 
            if diagonal:   
                shList = [sh for sh in matrix.dShapes if sh.color != matrix.backgroundColor]
            else:   
                shList = [sh for sh in matrix.shapes if sh.color != matrix.backgroundColor]
        else:
            if diagonal: 
                shList = [sh for sh in matrix.multicolorDShapes]
            else:
                shList = [sh for sh in matrix.multicolorShapes]
    if len(shList) < 2 or len(shList)>7:
        return matrix.m.copy()
    if outDummyMatrix is None:
        shList.sort(key=lambda x: x.shape[0]*x.shape[1], reverse=True)
        if outShape == 'LaSh':
            outShape = shList[0].shape    
        if outShape == None:
            outShape = matrix.shape
        if all((sh.shape[0]<=outShape[0] and sh.shape[1]<=outShape[1]) for sh in shList) and\
                                sum(len(sh.pixels) for sh in shList) <= outShape[0]*outShape[1]:
            m, tessellate = tessellateShapes(np.full(outShape, fill_value=matrix.backgroundColor),shList,\
                                             0,matrix.backgroundColor)
            if tessellate:
                return m
            m, tessellate = tessellateShapes(np.full(outShape, fill_value=matrix.backgroundColor),shList,\
                                             0,matrix.backgroundColor,rotation=True)
            if tessellate:
                return m
    else:
        m = np.full(outDummyMatrix.shape,fill_value=outDummyColor)
        shC = Counter([sh.shape for sh in shList])
        pSh = shC.most_common(1)[0][0]
        if pSh[0]>m.shape[0] or pSh[1]>m.shape[1] or m.shape[0]%pSh[0] != 0 or m.shape[1]%pSh[1] != 0:
            return matrix.m.copy()
        for i, j in np.ndindex(m.shape[0]//pSh[0], m.shape[1]//pSh[1]):
            for k, l in np.ndindex(matrix.shape[0]-pSh[0]+1, matrix.shape[1]-pSh[1]+1):
                if np.all((matrix.m[k:k+pSh[0],l:l+pSh[1]] == outDummyColor)==(outDummyMatrix[i*pSh[0]:(i+1)*pSh[0],j*pSh[1]:(j+1)*pSh[1]]==0)):
                    m[i*pSh[0]:(i+1)*pSh[0],j*pSh[1]:(j+1)*pSh[1]] = matrix.m[k:k+pSh[0],l:l+pSh[1]]
                    break
        return m
    return matrix.m.copy()
    
def getBestReplicateOneShape(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    if t.outSmallerThanIn or (not t.sameIOShapes):
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='pixelwise', reshape=True,\
                                                                multicolor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='horizontal', reshape=True,\
                                                                multicolor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='vertical', reshape=True,\
                                                                multicolor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='pixelwise', reshape=True,\
                                                                multicolor=False, paintLikePix=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='horizontal', reshape=True,\
                                                                multicolor=False, paintLikePix=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='vertical', reshape=True,\
                                                                multicolor=False, paintLikePix=True), bestScore, bestFunction)
    deleteO = [False]
    deleteA = [False]
    if isDeleteTask(t):
        deleteO += [True]
        deleteA += [True]
    if t.sameIOShapes:
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='pixelwise', multicolor=False,\
                                                                paintLikePix=True), bestScore, bestFunction)
        for deleteOriginal in deleteO:
            for deleteAnchor in deleteA:
                bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, deleteOriginal=deleteOriginal,\
                                                                deleteAnchor=deleteAnchor), bestScore, bestFunction)
    return bestFunction

def replicateOneShape(matrix, diagonal=True, multicolor=True, deleteOriginal=False,\
                      deleteAnchor=False, lay=False, reshape=False,overlap=0, paintLikePix=False):
    m = matrix.m.copy()
    #first find the shape or shapes to replicate
    if diagonal:
        if multicolor:
            shList = [sh for sh in matrix.multicolorDShapes if len(sh.pixels)>1]
        else:
            shList = [sh for sh in matrix.dShapes if (len(sh.pixels)>1 and sh.color != matrix.backgroundColor)]
    else:
        if multicolor:
            shList = [sh for sh in matrix.multicolorShapes if len(sh.pixels)>1]
        else:
            shList = [sh for sh in matrix.shapes if (len(sh.pixels)>1 and sh.color != matrix.backgroundColor)]
    pixList = matrix.isolatedPixels#[pix for pix in matrix.dShapes if len(pix.pixels)==1]
    if len(shList) != 1 or len(pixList) == 0:
        return m
    repSh = shList[0]
    if lay != False:
        if lay == 'pixelwise':
            if len(pixList) < 2:
                return m
            if len(set([p.position[0] for p in pixList])) == 1:
                pixList.sort(key=lambda x: x.position[1])
                steps = [(0, pixList[i].position[1] - pixList[i-1].position[1] - 1) for i in range(1,len(pixList))]
                direction = (0, (pixList[1].position[1] - pixList[0].position[1] - 1)//abs(pixList[1].position[1] - pixList[0].position[1] - 1))
            elif len(set([p.position[1] for p in pixList])) == 1:
                pixList.sort(key=lambda x: x.position[0])
                steps = [(pixList[i].position[0] - pixList[i-1].position[0] - 1, 0) for i in range(1,len(pixList))]
                direction = ((pixList[1].position[0] - pixList[0].position[0] - 1)//abs(pixList[1].position[0] - pixList[0].position[0] - 1), 0)
            else:
                return m
            if paintLikePix and repSh.color == pixList[-1].color:
                    steps = steps[::-1]
                    steps = [(-st[0], -st[1]) for st in steps]
                    direction = (-direction[0], -direction[1])
                    pixList = pixList[::-1]
            if reshape:
                m = np.full((repSh.shape[0]*(1 + (len(pixList)-1)*abs(direction[0])), repSh.shape[1]*(1 + (len(pixList)-1)*abs(direction[1]))),\
                            fill_value = matrix.backgroundColor)
                for i in range(len(pixList)):
                    newInsert = copy.deepcopy(repSh)
                    newInsert.position = (i*repSh.shape[0]*direction[0], i*repSh.shape[1]*direction[1])
                    if paintLikePix:
                        newInsert.m[repSh.m == repSh.color] = pixList[i].color
                    m = insertShape(m, newInsert)
                deleteOriginal = False
            else:
                pos = repSh.position
                for (p,i) in zip(steps, [j for j in range(1,len(pixList))]):
                    pos = (pos[0] + direction[0]*repSh.shape[0] + p[0], pos[1] + direction[1]*repSh.shape[1] + p[1])
                    newInsert = copy.deepcopy(repSh)
                    newInsert.position = pos
                    if paintLikePix:
                        newInsert.m[repSh.m == repSh.color] = pixList[i].color
                    m = insertShape(m, newInsert)
        elif lay == 'horizontal': 
            m = np.full((repSh.shape[0], len(pixList)*repSh.shape[1]), fill_value = matrix.backgroundColor)
            deleteOriginal = False
            for i in range(len(pixList)):
                newInsert = copy.deepcopy(repSh)
                newInsert.position = (0, i*repSh.shape[1])
                m = insertShape(m, newInsert)
        elif lay == 'vertical': 
            m = np.full((len(pixList)*repSh.shape[0], repSh.shape[1]), fill_value = matrix.backgroundColor)
            deleteOriginal = False
            for i in range(len(pixList)):
                newInsert = copy.deepcopy(repSh)
                newInsert.position = (i*repSh.shape[0], 0)
                m = insertShape(m, newInsert)
    else:
        for pix in pixList:
            if (pix.position[0] >= repSh.position[0]) and (pix.position[1] >= repSh.position[1]) \
                and (pix.position[0] < repSh.position[0]+repSh.shape[0]) and (pix.position[1] < repSh.position[1]+repSh.shape[1]):
                continue
            newInsert = copy.deepcopy(repSh)
            if pix.m[0,0] in repSh.m:
                newInsert = copy.deepcopy(repSh)
                for i, j in np.ndindex(repSh.shape):
                    if repSh.m[i,j] == pix.m[0,0]:
                        newInsert.position = (pix.position[0]-i, pix.position[1]-j)
                        break
            else:
                newInsert.position = (pix.position[0] - (repSh.shape[0]-1)//2, pix.position[1] - (repSh.shape[1]-1)//2)
            m = insertShape(m, newInsert)
            if deleteAnchor:
                m = deleteShape(m, pix, matrix.backgroundColor)
    if deleteOriginal:
        m = deleteShape(m, repSh, matrix.backgroundColor)
    return m

def getBestMoveToPanel(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    for fit in [True, False]:
        for igPan in [True, False]:
            for uniq in [True, False]:
                bestFunction, bestScore = updateBestFunction(t, partial(moveToPanel,fit=fit,\
                                                ignorePanel=igPan), bestScore, bestFunction)
    return bestFunction

def moveToPanel(matrix, diagonal=True,fit=False, ignorePanel=False, cropPanel=True, uniq=True):
    m = matrix.m.copy()
    shList = [sh for sh in matrix.multicolorDShapes if len(sh.pixels)>1]
    if len(shList) < 2 or len(shList) > 8:
        return m
    shList.sort(key=lambda x: x.shape[0]*x.shape[1],reverse=True)
    panel = shList[0]
    shList = shList[1:]
    if fit and hasattr(panel, 'color'):
        pC = panel.color
        for sh in shList:
            found = False
            for x in range(4):
                rotSh = np.rot90(sh.m, x).copy()
                if panel.shape[0]-rotSh.shape[0]+1<0 or panel.shape[1]-rotSh.shape[1]+1<0:
                    continue
                for i, j in np.ndindex(panel.shape[0]-rotSh.shape[0]+1,panel.shape[1]-rotSh.shape[1]+1):
                    if np.all((rotSh==pC)==(panel.m[i:i+rotSh.shape[0],j:j+rotSh.shape[1]]==255)):
                        newInsert = copy.deepcopy(sh)
                        newInsert.m = rotSh
                        newInsert.shape = rotSh.shape
                        newInsert.position = (panel.position[0]+i, panel.position[1]+j)
                        m = insertShape(m, newInsert)
                        found = True
                        break
                if found:
                    break
    else:
        pixList = [pix for pix in matrix.dShapes if len(pix.pixels)==1]
        pixList = [pix for pix in pixList if all(pix.position[i]>=panel.position[i]\
                                        and pix.position[i]<panel.position[i]+panel.shape[i] for i in [0,1])]
        if len(pixList)==0:
            return m
        newInsList = []
        for pix in pixList:
            for sh in shList:
                if pix.m[0,0] in sh.m:
                    newInsert = copy.deepcopy(sh)
                    if sh.nColors == 1:
                        newInsert.position = (pix.position[0] - (sh.shape[0]-1)//2, pix.position[1] - (sh.shape[1]-1)//2)
                    else:
                        for i, j in np.ndindex(sh.shape):
                            if sh.m[i,j] == pix.m[0,0]:
                                newInsert.position = (pix.position[0]-i, pix.position[1]-j) 
                    newInsList.append(newInsert)
                    if uniq:
                        break
            if ignorePanel:
                m = deleteShape(m, panel, matrix.backgroundColor)
            for sh in newInsList:
                m = insertShape(m, sh)
    if cropPanel:
        return m[panel.position[0]:panel.position[0]+panel.shape[0],\
                 panel.position[1]:panel.position[1]+panel.shape[1]]
    return m
        
#overlapSubmatrices 
def printShapes(matrices, base=0, backgroundColor=0):
    """
    This function returns the result of printing one matrices on the other.
    The matrices are shape matrices and may contain 255.
    """
    if base == 1:
        matrices = matrices[::-1]
    m = np.zeros(matrices[0].shape, dtype=int)
    for i,j in np.ndindex(m.shape):
        if matrices[0][i,j] != 255 and matrices[0][i,j] != backgroundColor:
            if matrices[1][i,j] != 255:
                m[i,j] = matrices[1][i,j]
            else:
                m[i,j] = matrices[0][i,j]
        else:
            m[i,j] = backgroundColor
    return m 

def multiplyMatrices(matrices, outShape=None, background=0, base=0, color=0):
    """
    Copy m1 matrix into every pixel of m2 if it is not background. Output has shape
    m1.shape*m2.shape. base and color are arguments to swap m1 and m2 and to choose
    color appropriately
    """
    if base == 1:
        matrices = matrices[::-1]
    m1 = matrices[0].copy()
    m2 = matrices[1].copy()
    s1 = m1.shape
    s2 = m2.shape
    m = np.full((m1.shape[0]*m2.shape[0],m1.shape[1]*m2.shape[1]), fill_value=background)
    if color == 0:
        for i,j in np.ndindex(s1):
            if m1[i,j] != background:
                m[i*s2[0]:(i+1)*s2[0], j*s2[1]:(j+1)*s2[1]] = m2
    else:
       for i,j in np.ndindex(s1):
            if m1[i,j] != background:
                for k,l in np.ndindex(m2.shape):
                    if m2[k,l] != background:
                        m[i*s2[0]+k, j*s2[1]+l] = m1[i,j] 
    return m   

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
    return overlapMatrices(submat, colorHierarchy)

def overlapMatrices(matrices, colorHierarchy):
    """
    Overlaps matrices of a given shape according to the color hierarchy
    """
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        m[i,j] = colorHierarchy[max([colorHierarchy.index(x[i,j]) for x in matrices])]
    return m

def overlapShapes(matrix, diagonal=True, multicolor=True, byColor=False, hierarchy=[0,1,2,3,4,5,6,7,8,9]):
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes]
        else:   
            shList = [sh for sh in matrix.shapes]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    if byColor:
        shList = [sh for sh in matrix.shapesByColor]
    shList = [sh for sh in shList if sh.isRectangle]
    if len(set([sh.shape for sh in shList])) != 1:
        return matrix.m.copy()
    return overlapMatrices([sh.m for sh in shList],hierarchy)

#Cropshape
def getCropAttributes(t, diagonal, multicolor, sameColor=True):
    bC = max(0, t.backgroundColor)
    if diagonal and not multicolor:
        if t.nCommonInOutDShapes == 0:
            return set()
        attrs = set.intersection(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=True)[s.inMatrix.dShapes.index(s.commonDShapes[0][0])] for s in t.trainSamples])
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=True)
            for shi in range(len(s.inMatrix.dShapes)):
                if s.inMatrix.dShapes[shi] == s.commonDShapes[0][0]:
                    continue
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])
                    
    if not diagonal and not multicolor:
        if t.nCommonInOutShapes == 0:
            return set()
        attrs = set.intersection(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=False)[s.inMatrix.shapes.index(s.commonShapes[0][0])] for s in t.trainSamples])
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=False)
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
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=False, diagonals=False)
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
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=False, diagonals=True)
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
    bC = max(0, t.backgroundColor)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,True, False),\
                                                           backgroundColor=bC, singleColor=True, diagonals=True), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,False, False),\
                                                           backgroundColor=bC, singleColor=True, diagonals=False), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,True, True),\
                                                           backgroundColor=bC, singleColor=False, diagonals=True), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,False, True),\
                                                           backgroundColor=bC, singleColor=False, diagonals=False), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    for attr in ['LaSh', 'MoCo', 'MoCl', 'UnSh', 'UnSi']:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=True, diagonals=True), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=True, diagonals=False), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=False, diagonals=True), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=True, diagonals=False), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction
    
def cropShape(matrix, attributes, backgroundColor=0, singleColor=True, diagonals=True, context=False):
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
    if context:
        bestShape = shapeList[bestShapes[0]]
        m = matrix.m[bestShape.position[0]:bestShape.position[0]+bestShape.shape[0], bestShape.position[1]:bestShape.position[1]+bestShape.shape[1]]
        return m.copy() 
    else:
        bestShape = shapeList[bestShapes[0]].m.copy()
        bestShape[bestShape==255]=backgroundColor
    return bestShape
    
#Crop a shape using a reference shape or set of shapes
def getBestCropReference(t):
    bestFunction = partial(identityM)
    bestScore = 1000
    for sh in t.commonInShapes:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference, refShape=sh,\
                                refType='subshape', multicolor=True, diagonal=False), bestScore, bestFunction)
    if len(t.commonInShapes) == 1:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference, refShape=t.commonInShapes[0],\
                                refType='mark', multicolor=False, diagonal=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference, refShape=t.commonInShapes[0],\
                                refType='mark', multicolor=False, diagonal=False), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference,\
                                refType='pixels', maxOrMin='min',multicolor=True, diagonal=False), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference,\
                                refType='pixels', maxOrMin='max',multicolor=True, diagonal=False), bestScore, bestFunction)
    #add referenced by frames
    return bestFunction

#Crop a shape using a reference shape or set of shapes
def cropShapeReference(matrix, refShape=None, refType='subshape', maxOrMin='max', sameColor=True, multicolor=True, diagonal=False):
    if multicolor:
        if diagonal:
            shList = matrix.multicolorDShapes
        else:
            shList = matrix.multicolorShapes
    else:
        if diagonal:
            shList = matrix.dShapes
        else:
            shList = matrix.shapes
    if refType == 'subshape':
        bestSh, bestScore = None, 0
        for sh in shList:
            if hasattr(sh, 'subshapes') and refShape in sh.subshapes:
                score = np.count_nonzero([refShape == sh2 for sh2 in sh.subshapes])
                if score > bestScore:
                    bestSh = sh
                    bestScore = score
        if bestSh == None:
            return matrix.m.copy()
        return bestSh.m
    elif refType == 'pixels':
        if maxOrMin == 'max':
            bestSh, bestScore = None, 0
        else:
            bestSh, bestScore = None, 1000
        for sh in shList:
            if hasattr(sh, 'subshapes'):
                score = len([p for p in sh.subshapes if len(p.pixels) == 1])
                if maxOrMin == 'max' and score > bestScore:
                    bestSh = sh
                    bestScore = score
                if maxOrMin == 'min' and score < bestScore:
                    bestSh = sh
                    bestScore = score
        if bestSh == None:
            return matrix.m.copy()
        return bestSh.m              
    elif refType == 'frame':
        return matrix.m.copy()
    elif refType == 'mark':
        foundRef = False
        for sh in shList:
            if sh == refShape:
                refShape = sh
                foundRef = True
                break
        if not foundRef:
            return matrix.m
        #otherwise return closest to reference
        bestShape = None
        dist = 1000
        refPos = refShape.position
        refPixels = [(p[0]+refPos[0], p[1]+refPos[1]) for p in refShape.pixels]
        for sh in shList:
            if sh == refShape or sh.color == matrix.backgroundColor:
                continue
            for p2 in [(p[0]+sh.position[0], p[1]+sh.position[1]) for p in sh.pixels]:
                if min(abs((p[0]-p2[0]))+abs((p[1]-p2[1])) for p in refPixels) < dist:
                    bestShape = sh
                    dist = min(abs((p[0]-p2[0])+(p[1]-p2[1])) for p in refPixels)
        if bestShape == None:
            return matrix.m.copy()
        bestShape=bestShape.m
        bestShape[bestShape==255]=matrix.backgroundColor
        return bestShape         

def cropAllBackground(matrix):
    m = matrix.m.copy()
    bC = matrix.backgroundColor
    if np.all(m == bC):
        return m
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
    
###############################################################################
###############################################################################
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
        x.append(partial(downsizeMode,newShape=outShape))
        if t.backgroundColor!=-1:
            x.append(partial(downsize, newShape=outShape, falseColor=t.backgroundColor))

    ###########################################################################
    # sameIOShapes
    if candTask.sameIOShapes:
        
        #######################################################################
        
        # ColorMap
        ncc = len(candTask.colorChanges)
        if len(set([cc[0] for cc in candTask.colorChanges])) == ncc and ncc != 0:
            x.append(partial(colorMap, cMap=dict(candTask.colorChanges)))
            
        x.append(partial(revertColorOrder))
            
        # Symmetrize
        if all([len(x)==1 for x in candTask.changedInColors]):
            color = next(iter(candTask.changedInColors[0]))
            axis = []
            if candTask.lrSymmetric:
                axis.append("lr")
            if candTask.udSymmetric:
                axis.append("ud")
            if all([sample.inMatrix.shape[0]==sample.inMatrix.shape[1] for sample in candTask.testSamples]):
                if candTask.d1Symmetric:
                    axis.append("d1")
                if candTask.d2Symmetric:
                    axis.append("d2")
            x.append(partial(symmetrize, axis=axis, color=color))
            if candTask.totalOutColors==1:
                for fc in candTask.fixedColors:
                    x.append(partial(symmetrize, axis=axis, refColor=fc,\
                                     outColor=next(iter(candTask.totalOutColors))))
    
        # Color symmetric pixels
        x.append(getBestColorSymmetricPixels(candTask))
    
        # Complete rectangles
        if candTask.backgroundColor!=-1 and len(candTask.fixedColors)==1 and \
        len(candTask.colorChanges)==1:
            sc = next(iter(candTask.fixedColors))
            nc = next(iter(candTask.colorChanges))[1]
            x.append(partial(completeRectangles, sourceColor=sc, newColor=nc))
        
        #######################################################################
        # For LinearShapeModel we need to have the same shapes in the input
        # and in the output, and in the exact same positions.
        # This model predicts the color of the shape in the output.
        
        if candTask.onlyShapeColorChanges:
            ccwf = getColorChangesWithFeatures(candTask)
            fsf = candTask.fixedShapeFeatures
            x.append(partial(changeShapesWithFeatures, ccwf=ccwf, fixedColors=candTask.fixedColors,\
                             fixedShapeFeatures=fsf))
            
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
        #if candTask.sameNSampleColors and all(["predictCNN" not in str(op.func) for op in c.ops]):
        #    x.append(getBestSameNSampleColorsCNN(candTask))

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
            x.append(getBestFlipAllShapes(candTask))
                                
        #######################################################################
        # Other sameIOShapes functions
        # Move shapes
        x.append(getBestMoveShapes(candTask, candidate=c))
        
        pr = pixelRecolor(candTask)
        if len(pr)!=1:
            x.append(partial(executePixelRecolor, Best_Dict=pr[0], Best_v=pr[1], Best_Q1=pr[2], Best_Q2=pr[3]))
        
            
        # extendColor
        x.append(getBestExtendColor(candTask))
            
        # fillRectangleInside
        for cic in candTask.commonChangedInColors:
            for coc in candTask.commonChangedOutColors:
                x.append(partial(fillRectangleInside, rectangleColor=cic, fillColor=coc))
            

        # Connect Pixels
        x.append(partial(connectAnyPixels))
        x.append(partial(connectAnyPixels, diagonal=True))
        if all([len(x)==1 for x in candTask.changedInColors]):
            x.append(partial(connectAnyPixels, connColor=next(iter(candTask.changedOutColors[0]))))
            x.append(partial(connectAnyPixels, connColor=next(iter(candTask.changedOutColors[0])), \
                             diagonal=True))

        fc = candTask.fixedColors
        #if hasattr(t, "fixedColors"):
        #    tfc = candTask.fixedColors
        #else:
        #    tfc = set()
        for pc in candTask.colors - candTask.commonChangedInColors:
            for cc in candTask.commonChangedOutColors:
                x.append(partial(connectAnyPixels, pixelColor=pc, \
                                 connColor=cc, fixedColors=fc))
                x.append(partial(connectAnyPixels, pixelColor=pc, \
                                 connColor=cc, fixedColors=fc, diagonal=True))
        for pc in candTask.colors - candTask.commonChangedInColors:
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges)))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges), \
                             diagonal=True))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges),\
                             lineExclusive=True))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges),\
                             lineExclusive=True, diagonal=True))
                
        for cc in candTask.commonColorChanges:
            for cc in candTask.commonColorChanges:
                for border, bs in product([True, False, None], ["big", "small", None]):
                    x.append(partial(changeShapes, inColor=cc[0], outColor=cc[1],\
                                     bigOrSmall=bs, isBorder=border))
        
        #replicate/symmterize/other shape related tasks
        x.append(getBestSymmetrizeSubmatrix(candTask))
        x.append(getBestReplicateOneShape(candTask))
        x.append(getBestSymmetrizeAllShapes(candTask))
        

        #delete shapes
        if isDeleteTask(candTask) and all(t.backgroundColor == c[1] for c in candTask.colorChanges):
            x.append(getBestDeleteShapes(candTask, True, True))
                    
    ###########################################################################
    # Cases in which the input has always the same shape, and the output too
    if candTask.sameInShape and candTask.sameOutShape and \
    all(candTask.trainSamples[0].inMatrix.shape == s.inMatrix.shape for s in candTask.testSamples):
            
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
        
        pixelMap = pixelCorrespondence(candTask)
        if len(pixelMap) != 0:
            x.append(partial(mapPixels, pixelMap=pixelMap, outShape=candTask.outShape))
        
    ###########################################################################
    # Other cases
    
    if candTask.inSmallerThanOut and t.inSmallerThanOut:
        x.append(getBestExtendMatrix(candTask))
        
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
            totalColorCount = Counter()
            for sample in t.trainSamples:
                for color in sample.outMatrix.colorCount.keys():
                    totalColorCount[color] += sample.outMatrix.colorCount[color]
            falseColor = max(totalColorCount.items(), key=operator.itemgetter(1))[0]
            opCond = getBestMultiplyMatrix(candTask, falseColor)
            x.append(partial(doBestMultiplyMatrix, opCond=opCond, falseColor=falseColor))
            
            
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
       
    if candTask.inputIsGrid:
        if all([s.inMatrix.grid.shape==s.outMatrix.shape for s in candTask.trainSamples]):
            for times in range(1, 6):
                x.append(partial(colorAppearingXTimes, times=times))
            x.append(partial(maxColorFromCell))
               
    # Cropshape
    x.append(partial(overlapShapes))
    if candTask.outSmallerThanIn:
        if candTask.backgroundColor!=-1:
            x.append(partial(cropAllShapes, background=candTask.backgroundColor, diagonal=True))
            x.append(partial(cropAllShapes, background=candTask.backgroundColor, diagonal=False))
        
        bestCrop = getBestCropShape(candTask)
        if 'attributes' in bestCrop.keywords.keys():
            for attr in bestCrop.keywords['attributes']:
                if isinstance(attr, str) and attr[:2] == 'mo' and len(bestCrop.keywords['attributes']) > 1:
                    continue
                newCrop = copy.deepcopy(bestCrop)
                newCrop.keywords['attributes'] = set([attr])
                x.append(newCrop)
                
        x.append(getBestCropReference(candTask))  
    
        for attrs in [set(['LaSh']),set(['UnCo'])]:
            x.append(partial(cropShape, attributes=attrs, backgroundColor=max(0,candTask.backgroundColor),\
                                 singleColor=True, diagonals=True)) 
            x.append(partial(cropShape, attributes=attrs, backgroundColor=max(0,candTask.backgroundColor),\
                                 singleColor=True, diagonals=True, context=True)) 
        if candTask.outIsInMulticolorShapeSize:
            x.append(getBestMoveToPanel(candTask))
    if all([len(s.inMatrix.multicolorShapes)==1 for s in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropOnlyMulticolorShape, diagonals=False))
    if all([len(s.inMatrix.multicolorDShapes)==1 for s in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropOnlyMulticolorShape, diagonals=True))
    
    if candTask.sameIOShapes:        
        if candTask.sameNSampleColors and all(["predictCNN" not in str(op.func) for op in c.ops]):
            x.append(getBestSameNSampleColorsCNN(candTask))
    
    # startOps

    x.append(partial(paintGridLikeBackground))
    x.append(partial(cropAllBackground))
    
    # minimize
    if not candTask.sameIOShapes:
        x.append(partial(minimize))
    
    return x

###############################################################################
###############################################################################
# Submission Setup
    
class Candidate():
    """
    Objects of the class Candidate store the information about a possible
    candidate for the solution.

    ...
    Attributes
    ----------
    ops: list
        A list containing the operations to be performed to the input matrix
        in order to get to the solution. The elements of the list are partial
        functions (from functools.partial).
    score: int
        The score of the candidate. The score is defined as the sum of the
        number incorrect pixels when applying ops to the input matrices of the
        train samples of the task.
    tasks: list
        A list containing the tasks (in its original format) after performing
        each of the operations in ops, starting from the original inputs.
    t: Task.Task
        The Task.Task object corresponding to the current status of the task.
        This is, the status after applying all the operations of ops to the
        input matrices of the task.
    """
    def __init__(self, ops, tasks, score=1000, predictions=np.zeros((2,2))):
        self.ops = ops
        self.score = score
        self.tasks = tasks
        self.t = None
        self.predictions = predictions
        self.checked = False

    def __lt__(self, other):
        """
        A candidate is better than another one if its score is lower.
        """
        if self.score == other.score:
            return len(self.ops) < len(other.ops)
        return self.score < other.score

    def generateTask(self):
        """
        Assign to the attribute t the Task.Task object corresponding to the
        current task status.
        """
        self.t = Task(self.tasks[-1], 'dummyIndex', submission=True)

class Best3Candidates():
    """
    An object of this class stores the three best candidates of a task.

    ...
    Attributes
    ----------
    candidates: list
        A list of three elements, each one of them being an object of the class
        Candidate.
    """
    def __init__(self, Candidate1, Candidate2, Candidate3):
        self.candidates = [Candidate1, Candidate2, Candidate3]

    def maxCandidate(self):
        """
        Returns the index of the candidate with highest score.
        """
        x = 0
        if self.candidates[1] > self.candidates[0]:
            x = 1
        if self.candidates[2] > self.candidates[x]:
            x = 2
        return x

    def addCandidate(self, c):
        """
        Given a candidate c, this function substitutes c with the worst
        candidate in self.candidates only if it's a better candidate (its score
        is lower).
        """
        if all([self.candidates[i].score < c.score for i in range(3)]):
            return
        
        for i in range(3):
            if all([np.array_equal(self.candidates[i].predictions[x], c.predictions[x]) \
                    for x in range(len(c.predictions))]):
                return
        iMaxCand = self.maxCandidate()
        for i in range(3):
            if c < self.candidates[iMaxCand]:
                c.generateTask()
                self.candidates[iMaxCand] = c
                break

    def allPerfect(self):
        return all([c.score==0 for c in self.candidates])

    def getOrderedIndices(self):
        """
        Returns a list of 3 indices (from 0 to 2) with the candidates ordered
        from best to worst.
        """
        orderedList = [0]
        if self.candidates[1] < self.candidates[0]:
            orderedList.insert(0, 1)
        else:
            orderedList.append(1)
        if self.candidates[2] < self.candidates[orderedList[0]]:
            orderedList.insert(0, 2)
        elif self.candidates[2] < self.candidates[orderedList[1]]:
            orderedList.insert(1, 2)
        else:
            orderedList.append(2)
        return orderedList

# Crop task if necessary
def getCroppingPosition(matrix):
    bC = matrix.backgroundColor
    x, xMax, y, yMax = 0, matrix.m.shape[0]-1, 0, matrix.m.shape[1]-1
    while x <= xMax and np.all(matrix.m[x,:] == bC):
        x += 1
    while y <= yMax and np.all(matrix.m[:,y] == bC):
        y += 1
    return [x,y]
    
def needsCropping(t):
    # Only to be used if t.sameIOShapes
    for sample in t.trainSamples:
        if sample.inMatrix.backgroundColor != sample.outMatrix.backgroundColor:
            return False
        if getCroppingPosition(sample.inMatrix) != getCroppingPosition(sample.outMatrix):
            return False
        inMatrix = cropAllBackground(sample.inMatrix)
        outMatrix = cropAllBackground(sample.outMatrix)
        if inMatrix.shape!=outMatrix.shape or sample.inMatrix.shape==inMatrix.shape:
            return False
    return True

def cropTask(t, task):
    positions = {"train": [], "test": []}
    backgrounds = {"train": [], "test": []}
    for s in range(t.nTrain):
        task["train"][s]["input"] = cropAllBackground(t.trainSamples[s].inMatrix).tolist()
        task["train"][s]["output"] = cropAllBackground(t.trainSamples[s].outMatrix).tolist()
        backgrounds["train"].append(t.trainSamples[s].inMatrix.backgroundColor)
        positions["train"].append(getCroppingPosition(t.trainSamples[s].inMatrix))
    for s in range(t.nTest):
        task["test"][s]["input"] = cropAllBackground(t.testSamples[s].inMatrix).tolist()
        backgrounds["test"].append(t.testSamples[s].inMatrix.backgroundColor)
        positions["test"].append(getCroppingPosition(t.testSamples[s].inMatrix))
        if not t.submission:
            task["test"][s]["output"] = cropAllBackground(t.testSamples[s].outMatrix).tolist()
    return positions, backgrounds

def recoverCroppedMatrix(matrix, outShape, position, backgroundColor):
    m = np.full(outShape, backgroundColor, dtype=np.uint8)
    m[position[0]:position[0]+matrix.shape[0], position[1]:position[1]+matrix.shape[1]] = matrix.copy()
    return m
    
def needsRecoloring(t):
    """
    This method determines whether the task t needs recoloring or not.
    It needs recoloring if every color in an output matrix appears either
    in the input or in every output matrix.
    Otherwise a recoloring doesn't make sense.
    If this function returns True, then orderTaskColors should be executed
    as the first part of the preprocessing of t.
    """
    for sample in t.trainSamples:
        for color in sample.outMatrix.colors:
            if (color not in sample.inMatrix.colors) and (color not in t.commonOutColors):
                return False
    return True

def orderTaskColors(t):
    """
    Given a task t, this function generates a new task (as a dictionary) by
    recoloring all the matrices in a specific way.
    The goal of this function is to impose that if two different colors
    represent the exact same thing in two different samples, then they have the
    same color in both of the samples.
    Right now, the criterium to order colors is:
        1. Common colors ordered according to Task.Task.orderColors
        2. Colors that appear both in the input and the output
        3. Colors that only appear in the input
        4. Colors that only appear in the output
    In steps 2-4, if there is more that one color satisfying that condition, 
    the ordering will happen according to the colorCount.
    """
    def orderColors(trainOrTest):
        if trainOrTest=="train":
            samples = t.trainSamples
        else:
            samples = t.testSamples
        for sample in samples:
            sampleColors = t.orderedColors.copy()
            sortedColors = [k for k, v in sorted(sample.inMatrix.colorCount.items(), key=lambda item: item[1])]
            for c in sortedColors:
                if c not in sampleColors:
                    sampleColors.append(c)
            if trainOrTest=="train" or t.submission==False:
                sortedColors = [k for k, v in sorted(sample.outMatrix.colorCount.items(), key=lambda item: item[1])]
                for c in sortedColors:
                    if c not in sampleColors:
                        sampleColors.append(c)
                    
            rel, invRel = relDicts(sampleColors)
            if trainOrTest=="train":
                trainRels.append(rel)
                trainInvRels.append(invRel)
            else:
                testRels.append(rel)
                testInvRels.append(invRel)
                
            inMatrix = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
            for c in sample.inMatrix.colors:
                inMatrix[sample.inMatrix.m==c] = invRel[c]
            if trainOrTest=='train' or t.submission==False:
                outMatrix = np.zeros(sample.outMatrix.shape, dtype=np.uint8)
                for c in sample.outMatrix.colors:
                    outMatrix[sample.outMatrix.m==c] = invRel[c]
                if trainOrTest=='train':
                    task['train'].append({'input': inMatrix.tolist(), 'output': outMatrix.tolist()})
                else:
                    task['test'].append({'input': inMatrix.tolist(), 'output': outMatrix.tolist()})
            else:
                task['test'].append({'input': inMatrix.tolist()})
        
    task = {'train': [], 'test': []}
    trainRels = []
    trainInvRels = []
    testRels = []
    testInvRels = []
    
    orderColors("train")
    orderColors("test")
    
    return task, trainRels, trainInvRels, testRels, testInvRels

def recoverOriginalColors(matrix, rel):
    """
    Given a matrix, this function is intended to recover the original colors
    before being modified in the orderTaskColors function.
    rel is supposed to be either one of the trainRels or testRels outputs of
    that function.
    """
    m = matrix.copy()
    for i,j in np.ndindex(matrix.shape):
        if matrix[i,j] in rel.keys(): # TODO Task 162 fails. Delete this when fixed
            m[i,j] = rel[matrix[i,j]][0]
    return m

def ignoreGrid(t, task, inMatrix=True, outMatrix=True):
    for s in range(t.nTrain):
        if inMatrix:
            m = np.zeros(t.trainSamples[s].inMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.trainSamples[s].inMatrix.grid.cells[i][j][0].colors))
            task["train"][s]["input"] = m.tolist()
        if outMatrix:
            m = np.zeros(t.trainSamples[s].outMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.trainSamples[s].outMatrix.grid.cells[i][j][0].colors))
            task["train"][s]["output"] = m.tolist()
    for s in range(t.nTest):
        if inMatrix:
            m = np.zeros(t.testSamples[s].inMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].inMatrix.grid.cells[i][j][0].colors))
            task["test"][s]["input"] = m.tolist()
        if outMatrix and not t.submission:
            m = np.zeros(t.testSamples[s].outMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].outMatrix.grid.cells[i][j][0].colors))
            task["test"][s]["output"] = m.tolist()

def recoverGrid(t, x, s):
    realX = t.testSamples[s].inMatrix.m.copy()
    cells = t.testSamples[s].inMatrix.grid.cells
    for cellI in range(len(cells)):
        for cellJ in range(len(cells[0])):
            cellShape = cells[cellI][cellJ][0].shape
            position = cells[cellI][cellJ][1]
            for k,l in np.ndindex(cellShape):
                realX[position[0]+k, position[1]+l] = x[cellI,cellJ]
    return realX

def ignoreAsymmetricGrid(t, task):
    for s in range(t.nTrain):
        m = np.zeros(t.trainSamples[s].inMatrix.asymmetricGrid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.trainSamples[s].inMatrix.asymmetricGrid.cells[i][j][0].colors))
        task["train"][s]["input"] = m.tolist()
        m = np.zeros(t.trainSamples[s].outMatrix.asymmetricGrid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.trainSamples[s].outMatrix.asymmetricGrid.cells[i][j][0].colors))
        task["train"][s]["output"] = m.tolist()
    for s in range(t.nTest):
        m = np.zeros(t.testSamples[s].inMatrix.asymmetricGrid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.testSamples[s].inMatrix.asymmetricGrid.cells[i][j][0].colors))
        task["test"][s]["input"] = m.tolist()
        if not t.submission:
            m = np.zeros(t.testSamples[s].outMatrix.asymmetricGrid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].outMatrix.asymmetricGrid.cells[i][j][0].colors))
            task["test"][s]["output"] = m.tolist()

def recoverAsymmetricGrid(t, x, s):
    realX = t.testSamples[s].inMatrix.m.copy()
    cells = t.testSamples[s].inMatrix.asymmetricGrid.cells
    for cellI in range(len(cells)):
        for cellJ in range(len(cells[0])):
            cellShape = cells[cellI][cellJ][0].shape
            position = cells[cellI][cellJ][1]
            for k,l in np.ndindex(cellShape):
                realX[position[0]+k, position[1]+l] = x[cellI,cellJ]
    return realX

def rotateTaskWithOneBorder(t, task):
    rotTask = copy.deepcopy(task)
    rotations = {'train': [], 'test': []}
    for s in range(t.nTrain):
        border = t.trainSamples[s].commonFullBorders[0]
        if border.direction=='h' and border.position==0:
            rotations['train'].append(1)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 1).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 1).tolist()
        elif border.direction=='v' and border.position==t.trainSamples[s].inMatrix.shape[1]-1:
            rotations['train'].append(2)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 2).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 2).tolist()
        elif border.direction=='h' and border.position==t.trainSamples[s].inMatrix.shape[0]-1:
            rotations['train'].append(3)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 3).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 3).tolist()
        else:
            rotations['train'].append(0)
    
    for s in range(t.nTest):
        if t.submission:
            hasBorder=False
            for border in t.testSamples[s].inMatrix.fullBorders:
                if border.color!=t.testSamples[s].inMatrix.backgroundColor:
                    if border.direction=='h' and border.position==0:
                        rotations['test'].append(1)
                        rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
                    elif border.direction=='v' and border.position==t.testSamples[s].inMatrix.shape[1]-1:
                        rotations['test'].append(2)
                        rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 2).tolist()
                    elif border.direction=='h' and border.position==t.testSamples[s].inMatrix.shape[0]-1:
                        rotations['test'].append(3)
                        rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 3).tolist()
                    else:
                        rotations['test'].append(0)
                    hasBorder=True
                    break
            if not hasBorder:
                return False
        else:
            border = t.testSamples[s].commonFullBorders[0]
            if border.direction=='h' and border.position==0:
                rotations['test'].append(1)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 1).tolist()
            elif border.direction=='v' and border.position==t.testSamples[s].inMatrix.shape[1]-1:
                rotations['test'].append(2)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 2).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 2).tolist()
            elif border.direction=='h' and border.position==t.testSamples[s].inMatrix.shape[0]-1:
                rotations['test'].append(3)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 3).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 3).tolist()
            else:
                rotations['test'].append(0)
        
    return rotTask, rotations

def rotateHVTask(t, task):
    rotTask = copy.deepcopy(task)
    rotations = {'train': [], 'test': []}
    
    for s in range(t.nTrain):
        if t.trainSamples[s].isVertical:
            rotations['train'].append(1)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 1).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 1).tolist()
        else:
            rotations['train'].append(0)
    
    for s in range(t.nTest):
        if t.submission:
            if t.testSamples[s].inMatrix.isHorizontal:
                rotations['test'].append(0)
            elif t.testSamples[s].inMatrix.isVertical:
                rotations['test'].append(1)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
            else:
                return False, False
        else:
            if t.testSamples[s].isHorizontal:
                rotations['test'].append(0)
            elif t.testSamples[s].isVertical:
                rotations['test'].append(1)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 1).tolist()
            else:
                return False, False
            
    return rotTask, rotations

def recoverRotations(matrix, trainOrTest, s, rotations):
    if rotations[trainOrTest][s] == 1:
        m = np.rot90(matrix, 3)
    elif rotations[trainOrTest][s] == 2:
        m = np.rot90(matrix, 2)
    elif rotations[trainOrTest][s] == 3:
        m = np.rot90(matrix, 1)
    else:
        m = matrix.copy()        
    return m

def tryOperations(t, c, cTask, b3c, firstIt=False):
    """
    Given a Task.Task t and a Candidate c, this function applies all the
    operations that make sense to the input matrices of c. After a certain
    operation is performed to all the input matrices, a new candidate is
    generated from the resulting output matrices. If the score of the candidate
    improves the score of any of the 3 best candidates, it will be saved in the
    variable b3c, which is an object of the class Best3Candidates.
    """
    if c.checked or c.score==0 or b3c.allPerfect():
        return
    startOps = ("switchColors", "cropShape", "cropAllBackground", "minimize", \
                "maxColorFromCell", "deleteShapes", "replicateShapes","colorByPixels",\
                "paintGridLikeBackground") # applyEvolve?
    repeatIfPerfect = ("extendColor", "moveAllShapes")
    possibleOps = getPossibleOperations(t, c)
    for op in possibleOps:
        for s in range(t.nTrain):
            cTask["train"][s]["input"] = op(c.t.trainSamples[s].inMatrix).tolist()
            if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
                cTask["train"][s]["input"] = correctFixedColors(\
                     c.t.trainSamples[s].inMatrix.m,\
                     np.array(cTask["train"][s]["input"]),\
                     c.t.fixedColors, c.t.commonOnlyChangedInColors).tolist()
        newPredictions = []
        for s in range(t.nTest):
            newOutput = op(c.t.testSamples[s].inMatrix)
            newPredictions.append(newOutput)
            cTask["test"][s]["input"] = newOutput.tolist()
            if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
                cTask["test"][s]["input"] = correctFixedColors(\
                     c.t.testSamples[s].inMatrix.m,\
                     np.array(cTask["test"][s]["input"]),\
                     c.t.fixedColors, c.t.commonOnlyChangedInColors).tolist()
        cScore = sum([incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                            t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        changedPixels = sum([incorrectPixels(c.t.trainSamples[s].inMatrix.m, \
                                                  np.array(cTask["train"][s]["input"])) for s in range(t.nTrain)])
        newCandidate = Candidate(c.ops+[op], c.tasks+[copy.deepcopy(cTask)], cScore,\
                                 predictions=newPredictions)
        b3c.addCandidate(newCandidate)
        if firstIt and str(op)[28:60].startswith(startOps):
            if all([np.array_equal(np.array(cTask["train"][s]["input"]), \
                                   t.trainSamples[s].inMatrix.m) for s in range(t.nTrain)]):
                continue
            newCandidate.generateTask()
            tryOperations(t, newCandidate, cTask, b3c)
        elif str(op)[28:60].startswith(repeatIfPerfect) and c.score - changedPixels == cScore and changedPixels != 0:
            newCandidate.generateTask()
            tryOperations(t, newCandidate, cTask, b3c)
   # Mark the candidate as checked to not repeat all the operations
    c.checked = True
            
def getPredictionsFromTask(originalT, task):
    taskNeedsRecoloring = needsRecoloring(originalT)
    if taskNeedsRecoloring:
        task, trainRels, trainInvRels, testRels, testInvRels = orderTaskColors(originalT)
        t = Task(task, task_id, submission=True)
    else:
        t = originalT

    cTask = copy.deepcopy(task)

    if t.sameIOShapes:
        taskNeedsCropping = needsCropping(t)
    else:
        taskNeedsCropping = False
    if taskNeedsCropping:
        cropPositions, backgrounds = cropTask(t, cTask)
        t2 = Task(cTask, task_id, submission=True, backgrounds=backgrounds)
    elif t.hasUnchangedGrid:
        if t.gridCellsHaveOneColor:
            ignoreGrid(t, cTask) # This modifies cTask, ignoring the grid
            t2 = Task(cTask, task_id, submission=True)
        elif t.outGridCellsHaveOneColor:
            ignoreGrid(t, cTask, inMatrix=False)
            t2 = Task(cTask, task_id, submission=True)
        else:
            t2 = t
    elif t.hasUnchangedAsymmetricGrid and t.assymmetricGridCellsHaveOneColor:
        ignoreAsymmetricGrid(t, cTask)
        t2 = Task(cTask, task_id, submission=True)
    else:
        t2 = t
        
    if t2.sameIOShapes:
        hasRotated = False
        if t2.hasOneFullBorder:
            hasRotated, rotateParams = rotateTaskWithOneBorder(t2, cTask)
        elif t2.requiresHVRotation:
            hasRotated, rotateParams = rotateHVTask(t2, cTask)
        if hasRotated!=False:
            cTask = hasRotated.copy()
            t2 = Task(cTask, task_id, submission=True)

    cScore = sum([incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                         t2.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
    dummyPredictions = [sample.inMatrix.m for sample in t2.testSamples]
    c = Candidate([], [task], score=cScore, predictions=dummyPredictions)
    c.t = t2
    b3c = Best3Candidates(c, c, c)

    # Generate the three candidates with best possible score
    prevScore = sum([c.score for c in b3c.candidates])
    firstIt = True
    while True:
        copyB3C = copy.deepcopy(b3c)
        for c in copyB3C.candidates:
            if c.score == 0:
                continue
            tryOperations(t2, c, cTask, b3c, firstIt)
            if firstIt:
                firstIt = False
                break
        score = sum([c.score for c in b3c.candidates])
        if score >= prevScore:
            break
        else:
            prevScore = score
            
    taskPredictions = []
    
    # Once the best 3 candidates have been found, make the predictions
    for s in range(t.nTest):
        taskPredictions.append([])
        for c in b3c.candidates:
            #print(c.ops)
            x = t2.testSamples[s].inMatrix.m.copy()
            for opI in range(len(c.ops)):
                newX = c.ops[opI](Matrix(x))
                if t2.sameIOShapes and len(t2.fixedColors) != 0:
                    x = correctFixedColors(x, newX, t2.fixedColors, t2.commonOnlyChangedInColors)
                else:
                    x = newX.copy()
            if t2.sameIOShapes and hasRotated!=False:
                x = recoverRotations(x, "test", s, rotateParams)
            if taskNeedsCropping:
                x = recoverCroppedMatrix(x, originalT.testSamples[s].inMatrix.shape, \
                                         cropPositions["test"][s], t.testSamples[s].inMatrix.backgroundColor)
            elif t.hasUnchangedGrid and (t.gridCellsHaveOneColor or t.outGridCellsHaveOneColor):
                x = recoverGrid(t, x, s)
            elif t.hasUnchangedAsymmetricGrid and t.assymmetricGridCellsHaveOneColor:
                x = recoverAsymmetricGrid(t, x, s)
            if taskNeedsRecoloring:
                x = recoverOriginalColors(x, testRels[s])
            
            taskPredictions[s].append(x)
            
    return taskPredictions, b3c
        
###############################################################################
# %% Main Loop and submission
            
#if submission.index[0] == '00576224_0':
#    submission.to_csv('submission.csv', index=False)
#    exit()

submission = pd.DataFrame(columns=['output'])
submission.index.name = 'output_id'
    
for task_id, task in data.items():   
                    
    originalT = Task(task, task_id, submission=True)
        
    predictions, b3c = getPredictionsFromTask(originalT, task.copy())

    finalPredictions = predictions

    pred = []
    for pair_id in range(originalT.nTest):
        for i in range(len(finalPredictions[pair_id])):
            pred.append(flattener(finalPredictions[pair_id][i].astype(int).tolist()))
    predictions = pred

    if len(predictions) == 0:
        pred = '|0| |0| |0|'
    elif len(predictions) == 1:
        pred = predictions[0] + ' ' + predictions[0] + ' ' + predictions[0]
    elif len(predictions) == 2:
        pred =  predictions[0] + ' ' + predictions[1] + ' ' + predictions[0]
    elif len(predictions) == 3:
        pred = predictions[0] + ' ' + predictions[1] + ' ' + predictions[2]
    
    for pair_id in range(originalT.nTest):
        submission.loc[task_id+'_'+str(pair_id), 'output'] = pred
    
submission.to_csv('submission.csv')
        