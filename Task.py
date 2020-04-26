import numpy as np
from collections import Counter
import copy

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
        
        self.nHoles = self.getNHoles()
        
        """
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
            for nPix in range(1,11):
                self.boolFeatures.append(self.nPixels==nPix)
            self.boolFeatures.append((self.nPixels%2)==0)
            self.boolFeatures.append((self.nPixels%2)==1)
        """
    
    def hasSameShape(self, other, sameColor=False, samePosition=False, rotation=False):
        if samePosition:
            if self.position != other.position:
                return False
        if sameColor:
            m1 = self.m
            m2 = other.m
        else:
            m1 = self.shapeDummyMatrix()
            m2 = other.shapeDummyMatrix()
        if rotation:
            if any([np.array_equal(m1, np.rot90(m2,x)) for x in range(1,4)]):
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
    
    """
    def hasFeatures(self, features):
        for i in range(len(features)):
            if features[i] and not self.boolFeatures[i]:
                return False
        return True
    """

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
    """
            # Is the shape in the border?
            isBorder = False
            if any([c[0] == 0 or c[1] == 0 or c[0] == iMax or c[1] == jMax for c in newShape]):
                isBorder = True
            
            # Now: What kind of shape is it???
            if len(newShape) == 1:
                s = Pixel(np.array(newShape), color, isBorder)
            else:
                # List containing the number of shape pixels surrounding each pixel
                nSurroundingPixels = []
                for p in newShape:
                    psp = 0
                    if [p[0]-1, p[1]] in newShape:
                        psp += 1
                    if [p[0]+1, p[1]] in newShape:
                        psp += 1
                    if [p[0], p[1]+1] in newShape:
                        psp += 1
                    if [p[0], p[1]-1] in newShape:
                        psp += 1
                    nSurroundingPixels.append(psp)
                # Check for loops and frames
                # If len(newShape) == 4, then it is a 2x2 square, not a loop!
                if all([s==2 for s in nSurroundingPixels]) and len(newShape) != 4:
                    maxI = max([p[0] for p in newShape])
                    minI = min([p[0] for p in newShape])
                    maxJ = max([p[1] for p in newShape])
                    minJ = min([p[1] for p in newShape])
                    isFrame = True
                    for p in newShape:
                        if p[0] not in [maxI, minI] and p[1] not in [maxJ, minJ]:
                            isFrame = False
                            s = Loop(np.array(newShape), color, isBorder)
                            break
                    if isFrame:
                        s = Frame(np.array(newShape), color, isBorder)
                # Check for lines and Frontiers
                spCounter = Counter(nSurroundingPixels)
                if len(spCounter) == 2 and 1 in spCounter.keys() and spCounter[1] == 2:
                    if len(set([p[0] for p in newShape])) == 1 or len(set([p[0] for p in newShape])):
                        if newShape[0][0] == newShape[1][0]:
                            s = Line(np.array(newShape), color, isBorder, 'v')
                        else:
                            s = Line(np.array(newShape), color, isBorder, 'h')
                    else:
                        s = Path(np.array(newShape), color, isBorder)
            if 's' not in locals(): 
                s = GeneralShape(np.array(newShape), color, isBorder)
            shapes.append(s)
            del s
    
    return shapes    

class Pixel(Shape):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        self.nHoles=0
        self.isSquare=True
        self.isRectangle=True
        
class Path(Shape):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        self.isSquare=False
        self.isRectangle=False
        self.nHoles=0

class Line(Path):
    def __init__(self, pixels, color, isBorder, orientation):
        super().__init__(pixels, color, isBorder)
        self.orientation = orientation
        
class Loop(Shape):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        self.nHoles=1
        self.isSquare=False
            self.isRectangle=False
        
class Frame(Loop):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        
class GeneralShape(Shape):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        
        self.isRectangle = self.nPixels == (self.xLen+1) * (self.yLen+1)
        self.isSquare = self.isRectangle and self.xLen == self.yLen
        
        # Number of holes
        self.nHoles = self.getNHoles()
        
    """
        
# %% Class Matrix
class Matrix():
    def __init__(self, m, detectGrid=True):
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
        self.backgroundColor = max(self.colorCount, key=self.colorCount.get)
        
        # Shapes
        self.shapes = detectShapes(self.m, self.backgroundColor, singleColor=True)
        self.nShapes = len(self.shapes)
        self.dShapes = detectShapes(self.m, self.backgroundColor, singleColor=True, diagonals=True)
        self.nDShapes = len(self.dShapes)
        self.multicolorShapes = detectShapes(self.m, self.backgroundColor)
        self.multicolorDShapes = detectShapes(self.m, self.backgroundColor, diagonals=True)
        #R: Since black is the most common background color. 
        #self.nonBMulticolorShapes = detectShapes(self.m, 0)
        #self.nonBMulticolorDShapes = detectShapes(self.m, 0, diagonals=True)
        # Non-background shapes
        #self.notBackgroundShapes = [s for s in self.shapes if s.color != self.backgroundColor]
        #self.nNBShapes = len(self.notBackgroundShapes)
        #self.notBackgroundDShapes = [s for s in self.dShapes if s.color != self.backgroundColor]
        #self.nNBDShapes = len(self.notBackgroundDShapes)
        
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
        if detectGrid:
            for fc in set(self.frontierColors):
                possibleGrid = [f for f in self.frontiers if f.color==fc]
                possibleGrid = Grid(self.m, possibleGrid)
                if possibleGrid.allCellsSameShape and possibleGrid.nCells > 1:
                    self.grid = copy.deepcopy(possibleGrid)
                    self.isGrid = True
                    break 
                
        # Frames
        self.fullFrames = []
        for shape in self.shapes:
            if shape.shape[0]>2 and shape.shape[1]>2:
                if not np.any(shape.m[1:shape.shape[0]-1,1:shape.shape[1]-1]==shape.color):
                    self.fullFrames.append(shape)
        
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
    
    """
    def shapeHasFeatures(self, index, features):
        for i in range(len(features)):
            if features[i] and not self.shapeFeatures[index][i]:
                return False
        return True
    """
    
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
        else:
            if diagonals: 
                shapeList = [sh for sh in self.multicolorDShapes]
            else:
                shapeList = [sh for sh in self.multicolorShapes]
        attrList =[[] for i in range(len(shapeList))]
        if singleColor:
            cc = Counter([sh.color for sh in shapeList])
        if singleColor:
            sc = Counter([sh.nPixels for sh in shapeList if sh.color != backgroundColor])
        else:
            sc = Counter([sh.nPixels for sh in shapeList])
        largest, smallest, mcopies, mcolors = -1, 1000, 0, 0
        ila, ism = [], []
        for i in range(len(shapeList)):
            #color count
            if singleColor:
                if shapeList[i].color == backgroundColor:
                    attrList[i].append(-1)
                    continue
            else:
                attrList[i].append(shapeList[i].nColors)
                if shapeList[i].nColors > mcolors:
                    mcolors = shapeList[i].nColors
            #copies 
            attrList[i] = [np.count_nonzero([np.all(shapeList[i].pixels == osh.pixels) for osh in shapeList])] + attrList[i]
            if attrList[i][0] > mcopies:
                mcopies = attrList[i][0]
            #unique color?
            if singleColor:
                if cc[shapeList[i].color] == 1:
                    attrList[i].append('UnCo')
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
    def __init__(self, s):
        
        self.inMatrix = Matrix(s['input'])
        self.outMatrix = Matrix(s['output'])
                
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
        """
        if self.sameShape:
            self.diffMatrix = Matrix((self.inMatrix.m - self.outMatrix.m).tolist())
            self.diffPixels = np.count_nonzero(self.diffMatrix.m)
        """
        # Is one a subset of the other? for now always includes diagonals
        self.inSmallerThanOut = all(self.inMatrix.shape[i] <= self.outMatrix.shape[i] for i in [0,1]) and not self.sameShape
        self.outSmallerThanIn = all(self.inMatrix.shape[i] >= self.outMatrix.shape[i] for i in [0,1]) and not self.sameShape

        #R: Is the output a shape (faster than checking if is a subset?

        if self.outSmallerThanIn:
            #check if output is the size of a multicolored shape
            self.outIsInMulticolorShapeSize = any((sh.shape == self.outMatrix.shape) for sh in self.inMatrix.multicolorShapes)
            self.outIsInMulticolorDShapeSize = any((sh.shape == self.outMatrix.shape) for sh in self.inMatrix.multicolorDShapes)

        self.commonShapes = self.getCommonShapes(diagonal=False, sameColor=True)
        self.commonDShapes = self.getCommonShapes(diagonal=True, sameColor=True)
        self.commonMulticolorShapes = self.getCommonShapes(diagonal=False, sameColor=False)
        self.commonMulticolorDShapes = self.getCommonShapes(diagonal=True, sameColor=False)

        # Shapes in the input that are fixed
        if self.sameShape:
            self.fixedShapes = []
            for sh in self.inMatrix.shapes:
                shapeIsFixed = True
                for i,j in np.ndindex(sh.shape):
                    if sh.m[i,j] != 255:
                        if self.outMatrix.m[sh.position[0]+i,sh.position[1]+j]!=sh.m[i,j]:
                            shapeIsFixed=False
                            break
                if shapeIsFixed:
                    self.fixedShapes.append(sh)
        
        """
        # Is the output a subset of the input?
        self.inSubsetOfOutIndices = set()
        if self.inSmallerThanOut:
            for i, j in np.ndindex((self.outMatrix.shape[0] - self.inMatrix.shape[0] + 1, self.outMatrix.shape[1] - self.inMatrix.shape[1] + 1)):
                if np.all(self.inMatrix.m == self.outMatrix.m[i:i+self.inMatrix.shape[0], j:j+self.inMatrix.shape[1]]):
                    self.inSubsetOfOutIndices.add((i, j))
        # Is the input a subset of the output?
        self.outSubsetOfInIndices = set()
        if self.outSmallerThanIn:
            for i, j in np.ndindex((self.inMatrix.shape[0] - self.outMatrix.shape[0] + 1, self.inMatrix.shape[1] - self.outMatrix.shape[1] + 1)):
                if np.all(self.outMatrix.m == self.inMatrix.m[i:i+self.outMatrix.shape[0], j:j+self.outMatrix.shape[1]]):
                    self.outSubsetOfInIndices.add((i, j))
            #Is output a single input shape?
            if len(self.outSubsetOfInIndices) == 1:
                #modify to compute background correctly
                for sh in self.outMatrix.shapes:
                    if sh.m.size == self.outMatrix.m.size:
                        osh = sh
                        self.outIsShape = True
                        self.outIsShapeAttributes = []
                        for ish in self.inMatrix.shapes:
                            if ish.m == osh.m:
                                break
                        self.outIsShapeAttributes = attribute_list(ish, self.inMatrix)
                        break
        """
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
                                         self.changedPixels[change]==self.inMatrix.colorCount[change[0]])
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
        
        # Is there a blank to fill?
        self.inputHasBlank = len(self.inMatrix.blanks)>0
        if self.inputHasBlank:
            for s in self.inMatrix.blanks:
                if s.shape == self.outMatrix.shape:
                    self.blankToFill = s
                    
         
        # Does the output matrix follow a pattern?
        self.followsRowPattern = self.outMatrix.followsRowPattern()
        self.followsColPattern = self.outMatrix.followsColPattern()
        
    def getCommonShapes(self, diagonal=True, sameColor=False):
        comSh = []
        if diagonal:
            if sameColor:
                ishs = self.inMatrix.dShapes
                oshs = self.outMatrix.dShapes
            else:
                ishs = self.inMatrix.multicolorDShapes
                oshs = self.outMatrix.multicolorDShapes
        else:
            if sameColor:
                ishs = self.inMatrix.shapes
                oshs = self.outMatrix.shapes
            else:
                ishs = self.inMatrix.multicolorShapes
                oshs = self.outMatrix.multicolorShapes
        #Arbitrary: shapes have size < 30.         
        for ish in ishs:
            outCount = 0
            if len(ish.pixels) == 1 or len(ish.pixels) > 30:
                continue
            for osh in oshs:
                if len(osh.pixels) == 1 or len(osh.pixels) > 30:
                    continue
                if sameColor:
                     if ish == osh:
                        outCount += 1
                        break
                else:
                   if ish.pixels == osh.pixels:
                       outCount += 1
            if outCount > 0:
                comSh.append((ish, outCount))
        return comSh

# %% Class Task
class Task():
    def __init__(self, t, i):
        self.task = t
        self.index = i
        
        self.trainSamples = [Sample(s) for s in t['train']]
        self.testSamples = [Sample(s) for s in t['test']]
        
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
        
        # Is the output always smaller?
        self.outSmallerThanIn = all(s.outSmallerThanIn for s in self.trainSamples)
        self.inSmallerThanOut = all(s.inSmallerThanOut for s in self.trainSamples)
            
                
        # Check for I/O subsets
        """
        self.inSubsetOfOut = self.trainSamples[0].inSubsetOfOutIndices
        for s in self.trainSamples:
            self.inSubsetOfOut = set.intersection(self.inSubsetOfOut, s.inSubsetOfOutIndices)
        self.outSubsetOfIn = self.trainSamples[0].outSubsetOfInIndices
        for s in self.trainSamples:
            self.outSubsetOfIn = set.intersection(self.outSubsetOfIn, s.outSubsetOfInIndices)
        """
        
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
            # Complete color changes
            self.completeColorChanges = [s.completeColorChanges for s in self.trainSamples]
            self.commonCompleteColorChanges = set.intersection(*self.completeColorChanges)
            self.allColorChangesAreComplete = all([s.allColorChangesAreComplete for s in self.trainSamples])
            # Are there any fixed colors?
            self.fixedColors = set.intersection(*[s.fixedColors for s in self.trainSamples])
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
        
        # Is there always a background color? Which one?
        if self.allEqual([s.inMatrix.backgroundColor for s in self.trainSamples]) and\
        self.trainSamples[0].inMatrix.backgroundColor == self.testSamples[0].inMatrix.backgroundColor:
            self.backgroundColor = self.trainSamples[0].inMatrix.backgroundColor
        else:
            self.backgroundColor = -1
            
        #R: is output a shape in the input
        self.outIsInMulticolorShapeSize = False
        self.outIsInMulticolorDShapeSize = False

        if all([(hasattr(s, "outIsInMulticolorShapeSize") and s.outIsInMulticolorShapeSize) for s in self.trainSamples]):
             self.outIsInMulticolorShapeSize = True
        if all([(hasattr(s, "outIsInMulticolorDShapeSize") and s.outIsInMulticolorDShapeSize) for s in self.trainSamples]):
             self.outIsInMulticolorDShapeSize = True
             
        """
        if all([hasattr(s, 'outIsInDShape') for s in self.trainSamples]) and all(len(s.outIsInDShape)==1 for s in self.trainSamples):
            self.outIsInDShape = True
        if all([hasattr(s, 'outIsInShape') for s in self.trainSamples]) and all(len(s.outIsInShape)>0 for s in self.trainSamples):
            self.outIsInShape = True
        if all([hasattr(s, 'outIsInNonBMulticolorDShape') for s in self.trainSamples]) and all(len(s.outIsInNonBMulticolorDShape)>0 for s in self.trainSamples) and self.backgroundColor == 0:
            self.outIsInMulticolorDShape = True
        if all([hasattr(s, 'outIsInNonBMulticolorShape') for s in self.trainSamples]) and all(len(s.outIsInNonBMulticolorShape)>0 for s in self.trainSamples) and self.backgroundColor == 0:
            self.outIsInMulticolorShape = True
        if all([hasattr(s, 'outIsInMulticolorShape') for s in self.trainSamples]) and all(len(s.outIsInMulticolorShape)>0 for s in self.trainSamples):
            self.outIsInMulticolorShape = True
        if all([hasattr(s, 'outIsInMulticolorShape') for s in self.trainSamples]) and all(len(s.outIsInMulticolorShape)>0 for s in self.trainSamples):
            self.outIsInMulticolorShape = True
        """
        self.nCommonInOutShapes = min(len(s.commonShapes) for s in self.trainSamples)
        self.nCommonInOutDShapes = min(len(s.commonDShapes) for s in self.trainSamples) 
        self.nCommonInOutMulticolorShapes = min(len(s.commonMulticolorShapes) for s in self.trainSamples)
        self.nCommonInOutMulticolorDShapes = min(len(s.commonMulticolorDShapes) for s in self.trainSamples) 
        
        """
        if len(self.commonInColors) == 1 and len(self.commonOutColors) == 1 and \
        next(iter(self.commonInColors)) == next(iter(self.commonOutColors)):
            self.backgroundColor = next(iter(self.commonInColors))
        else:
            self.backgroundColor = -1
        """
        
        """
        # Shape features
        self.shapeFeatures = []
        for s in self.trainSamples:
            self.shapeFeatures += s.shapeFeatures
        """
        
        """
        if self.sameIOShapes:
            self.fixedShapes = []
            for s in self.trainSamples:
                for sh in s.fixedShapes:
                    self.fixedShapes.append(sh)
            self.fixedShapeFeatures = []
            nFeatures = len(self.trainSamples[0].inMatrix.shapes[0].boolFeatures)
            for i in range(nFeatures):
                self.fixedShapeFeatures.append(True)
            for sh in self.fixedShapes:
                self.fixedShapeFeatures = [sh.boolFeatures[i] and self.fixedShapeFeatures[i] \
                                             for i in range(nFeatures)]
        """
        
        
        self.orderedColors = self.orderColors()
        
        # Grids:
        self.hasUnchangedGrid = all([s.gridIsUnchanged for s in self.trainSamples])
        if all([hasattr(s, "gridCellIsOutputShape") for s in self.trainSamples]):
            self.gridCellIsOutputShape = all([s.gridCellIsOutputShape for s in self.trainSamples])
        if all([hasattr(s, "gridCellIsInputShape") for s in self.trainSamples]):
            self.gridCellIsInputShape = all([s.gridCellIsInputShape for s in self.trainSamples])
        if self.hasUnchangedGrid:
            self.gridCellsHaveOneColor = all([s.gridCellsHaveOneColor for s in self.trainSamples])
        
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
                if not any([sh1==sh2 for sh2 in self.trainSamples[s].inMatrix.shapes]):
                    addShape = False
                    break
            if addShape:
                self.commonInShapes.append(sh1)

        self.commonInDShapes = []
        for sh1 in self.trainSamples[0].inMatrix.dShapes:
            if sh1.color == self.trainSamples[0].inMatrix.backgroundColor:
                continue
            addShape = True
            for s in range(1,self.nTrain):
                if not any([sh1==sh2 for sh2 in self.trainSamples[s].inMatrix.dShapes]):
                    addShape = False
                    break
            if addShape:
                self.commonInDShapes.append(sh1)
                
        # Frames
        self.hasFullFrame = all([len(s.inMatrix.fullFrames)>0 for s in self.trainSamples])

        # Is the task about filling a blank?
        self.fillTheBlank =  all([hasattr(s, 'blankToFill') for s in self.trainSamples])
                
        # Do all output matrices follow a pattern?
        self.followsRowPattern = all([s.followsRowPattern != False for s in self.trainSamples])
        self.followsColPattern = all([s.followsColPattern != False for s in self.trainSamples])
        if self.followsRowPattern:
            self.rowPatterns = [s.outMatrix.followsRowPattern() for s in self.trainSamples]
        if self.followsColPattern:
            self.colPatterns = [s.outMatrix.followsColPattern() for s in self.trainSamples]
        
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
                if c in self.testSamples[0].inMatrix.colors:
                    orderedColors.append(c)
        # 2: Colors that appear in every sample and are always changed from,
        # never changed to.
            for c in self.commonChangedInColors:
                if c not in self.commonChangedOutColors:
                    if c in self.testSamples[0].inMatrix.colors:
                        if c not in orderedColors:
                            orderedColors.append(c)
        # 3: Colors that appear in every sample and are always changed to,
        # never changed from.
            for c in self.commonChangedOutColors:
                if c not in self.commonChangedInColors:
                    if c not in orderedColors:
                        orderedColors.append(c)
        # 4: Add the background color.
        if self.backgroundColor != -1:
            if self.backgroundColor not in orderedColors:
                orderedColors.append(self.backgroundColor)
        # 5: Other colors that appear in every input.
        for c in self.commonInColors:
            if c in self.testSamples[0].inMatrix.colors:
                if c not in orderedColors:
                    orderedColors.append(c)
        # 6: Other colors that appear in every output.
        for c in self.commonOutColors:
            if c not in orderedColors:
                orderedColors.append(c)
                
        # TODO Dealing with grids and frames
        
        return orderedColors       