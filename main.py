# %% Setup
# This first cell needs to be executed before doing anything else
import json
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors

import Models
import Task
import Utils
from collections import Counter

# Load all the data. It needs to be in the folder 'data'
data_path = Path('data')
train_path = data_path / 'training'
valid_path = data_path / 'evaluation'
test_path = data_path / 'test'

train_tasks = { task.stem: json.load(task.open()) for task in train_path.iterdir() }
valid_tasks = { task.stem: json.load(task.open()) for task in valid_path.iterdir() }

# Correct wrong cases:
# 025d127b
for i in range(9, 12):
    for j in range(3, 8):
        train_tasks['025d127b']['train'][0]['output'][i][j] = 0
for i in range(7, 10):
    for j in range(3, 6):
        train_tasks['025d127b']['train'][0]['output'][i][j] = 2
train_tasks['025d127b']['train'][0]['output'][8][4] = 0
# ef135b50
train_tasks['ef135b50']['test'][0]['output'][6][4] = 9
# bd14c3bf
for i in range(3):
    for j in range(5):
        if valid_tasks['bd14c3bf']['test'][0]['input'][i][j] == 1:
            valid_tasks['bd14c3bf']['test'][0]['input'][i][j] = 2
# a8610ef7
for i in range(6):
    for j in range(6):
        if valid_tasks['a8610ef7']['test'][0]['output'][i][j] == 8:
            valid_tasks['a8610ef7']['test'][0]['output'][i][j] = 5
valid_tasks['a8610ef7']['train'][3]['input'][0][1] = 2
valid_tasks['a8610ef7']['train'][3]['input'][5][1] = 2
# 54db823b
valid_tasks['54db823b']['train'][0]['output'][2][3] = 3
valid_tasks['54db823b']['train'][0]['output'][2][4] = 9
# e5062a87
for j in range(3, 7):
    train_tasks['e5062a87']['train'][1]['output'][1][j] = 2
# 1b60fb0c
train_tasks['1b60fb0c']['train'][1]['output'][8][8] = 0
train_tasks['1b60fb0c']['train'][1]['output'][8][9] = 0
# 82819916
train_tasks['82819916']['train'][0]['output'][4][5] = 4
# fea12743
for i in range(11, 16):
    for j in range(6):
        if valid_tasks['fea12743']['train'][0]['output'][i][j] == 2:
            valid_tasks['fea12743']['train'][0]['output'][i][j] = 8
# 42a50994
train_tasks['42a50994']['train'][0]['output'][1][0] = 8
train_tasks['42a50994']['train'][0]['output'][0][1] = 8
# f8be4b64
for j in range(19):
    if valid_tasks['f8be4b64']['test'][0]['output'][12][j] == 0:
        valid_tasks['f8be4b64']['test'][0]['output'][12][j] = 1
valid_tasks['f8be4b64']['test'][0]['output'][12][8] = 0
# d511f180
train_tasks['d511f180']['train'][1]['output'][2][2] = 9
# 10fcaaa3
train_tasks['10fcaaa3']['train'][1]['output'][4][7] = 8
# cbded52d
train_tasks['cbded52d']['train'][0]['input'][4][6] = 1
# 11852cab
train_tasks['11852cab']['train'][0]['input'][1][2] = 3
# 868de0fa
for j in range(2, 9):
    train_tasks['868de0fa']['train'][2]['input'][9][j] = 0
    train_tasks['868de0fa']['train'][2]['input'][10][j] = 1
    train_tasks['868de0fa']['train'][2]['input'][15][j] = 0
    train_tasks['868de0fa']['train'][2]['input'][16][j] = 1
train_tasks['868de0fa']['train'][2]['input'][15][2] = 1
train_tasks['868de0fa']['train'][2]['input'][15][8] = 1
# 6d58a25d
train_tasks['6d58a25d']['train'][0]['output'][10][0] = 0
train_tasks['6d58a25d']['train'][2]['output'][6][13] = 4
# a9f96cdd
train_tasks['a9f96cdd']['train'][3]['output'][1][3] = 0
# 48131b3c
valid_tasks['48131b3c']['train'][2]['output'][4][4] = 0
# 150deff5
aux = train_tasks['150deff5']['train'][2]['output'].copy()
train_tasks['150deff5']['train'][2]['output'] = train_tasks['150deff5']['train'][2]['input'].copy()
train_tasks['150deff5']['train'][2]['input'] = aux
# 17cae0c1
for i in range(3):
    for j in range(3, 6):
        valid_tasks['17cae0c1']['test'][0]['output'][i][j] = 9
# e48d4e1a
train_tasks['e48d4e1a']['train'][3]['input'][0][9] = 5
train_tasks['e48d4e1a']['train'][3]['output'][0][9] = 0
# 8fbca751
valid_tasks['8fbca751']['train'][1]['output'][1][3] = 2
valid_tasks['8fbca751']['train'][1]['output'][2][3] = 8
# 4938f0c2
for i in range(12):
    for j in range(6,13):
        if train_tasks['4938f0c2']['train'][2]['input'][i][j]==2:
            train_tasks['4938f0c2']['train'][2]['input'][i][j] = 0
for i in range(5,11):
    for j in range(7):
        if train_tasks['4938f0c2']['train'][2]['input'][i][j]==2:
            train_tasks['4938f0c2']['train'][2]['input'][i][j] = 0
# 9aec4887
train_tasks['9aec4887']['train'][0]['output'][1][4] = 8
# b0f4d537
for i in range(9):
    valid_tasks['b0f4d537']['train'][0]['output'][i][3] = 0
    valid_tasks['b0f4d537']['train'][0]['output'][i][4] = 1
valid_tasks['b0f4d537']['train'][0]['output'][2][3] = 3
valid_tasks['b0f4d537']['train'][0]['output'][2][4] = 3
valid_tasks['b0f4d537']['train'][0]['output'][5][3] = 2
# aa300dc3
valid_tasks['aa300dc3']['train'][1]['input'][1][7] = 5
valid_tasks['aa300dc3']['train'][1]['output'][1][7] = 5
valid_tasks['aa300dc3']['train'][1]['input'][8][2] = 5
valid_tasks['aa300dc3']['train'][1]['output'][8][2] = 5



# allTasks stores the tasks as given originally. It is a dictionary, and its
# keys are the ids of the tasks
allTasks = train_tasks.copy()
allTasks.update(valid_tasks)

#df = pd.read_csv('info.csv', index_col=0)

# index is a list containing the 800 ids of the tasks
#index = list(allTasks.keys())
with open('index.pickle','rb') as f:
    index = pickle.load(f)

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
    if type(task)==int:
        task = allTasks[index[task]]
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

# For formatting the output
def flattener(pred):
    """
    This function formats the output. Only to be used when submitting to Kaggle
    """
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

# %% Current approach
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
    def __init__(self, ops, tasks, score=1000):
        self.ops = ops
        self.score = score
        self.tasks = tasks
        self.t = None

    def __lt__(self, other):
        """
        A candidate is better than another one if its score is lower.
        """
        return self.score < other.score

    def generateTask(self):
        """
        Assign to the attribute t the Task.Task object corresponding to the
        current task status.
        """
        self.t = Task.Task(self.tasks[-1], 'dummyIndex')

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
        iMaxCand = self.maxCandidate()
        for i in range(3):
            if c < self.candidates[iMaxCand]:
                c.generateTask()
                self.candidates[iMaxCand] = c
                break

    def allPerfect(self):
        return all([c.score==0 for c in self.candidates])
    
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
                    
            rel, invRel = Utils.relDicts(sampleColors)
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

def ignoreGrid(t, task):
    for s in range(t.nTrain):
        m = np.zeros(t.trainSamples[s].inMatrix.grid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.trainSamples[s].inMatrix.grid.cells[i][j][0].colors))
        task["train"][s]["input"] = m.tolist()
        m = np.zeros(t.trainSamples[s].outMatrix.grid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.trainSamples[s].outMatrix.grid.cells[i][j][0].colors))
        task["train"][s]["output"] = m.tolist()
    for s in range(t.nTest):
        m = np.zeros(t.testSamples[s].inMatrix.grid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.testSamples[s].inMatrix.grid.cells[i][j][0].colors))
        task["test"][s]["input"] = m.tolist()
        m = np.zeros(t.testSamples[s].outMatrix.grid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.testSamples[s].outMatrix.grid.cells[i][j][0].colors))
        task["test"][s]["output"] = m.tolist()

def recoverGrid(t, x):
    realX = t.testSamples[s].inMatrix.m.copy()
    cells = t.testSamples[s].inMatrix.grid.cells
    for cellI in range(len(cells)):
        for cellJ in range(len(cells[0])):
            cellShape = cells[cellI][cellJ][0].shape
            position = cells[cellI][cellJ][1]
            for k,l in np.ndindex(cellShape):
                realX[position[0]+k, position[1]+l] = x[cellI,cellJ]
    return realX

def tryOperations(t, c, firstIt=False):
    """
    Given a Task.Task t and a Candidate c, this function applies all the
    operations that make sense to the input matrices of c. After a certain
    operation is performed to all the input matrices, a new candidate is
    generated from the resulting output matrices. If the score of the candidate
    improves the score of any of the 3 best candidates, it will be saved in the
    variable b3c, which is an object of the class Best3Candidates.
    """
    if c.score==0 or b3c.allPerfect():
        return
    startOps = ("switchColors", "cropShape", "cropOnlyMulticolorShape", "minimize", \
                "maxColorFromCell")
    #repeatIfPerfect = ("changeShapes")
    possibleOps = Utils.getPossibleOperations(t, c)
    for op in possibleOps:
        for s in range(t.nTrain):
            cTask["train"][s]["input"] = op(c.t.trainSamples[s].inMatrix).tolist()
            if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
                cTask["train"][s]["input"] = Utils.correctFixedColors(\
                     c.t.trainSamples[s].inMatrix.m,\
                     np.array(cTask["train"][s]["input"]),\
                     c.t.fixedColors).tolist()
        for s in range(t.nTest):
            cTask["test"][s]["input"] = op(c.t.testSamples[s].inMatrix).tolist()
            if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
                cTask["test"][s]["input"] = Utils.correctFixedColors(\
                     c.t.testSamples[s].inMatrix.m,\
                     np.array(cTask["test"][s]["input"]),\
                     c.t.fixedColors).tolist()
        cScore = sum([Utils.incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                            t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        #changedPixels = sum([Utils.incorrectPixels(c.t.trainSamples[s].inMatrix.m, \
        #                                           np.array(cTask["train"][s]["input"])) for s in range(t.nTrain)])
        newCandidate = Candidate(c.ops+[op], c.tasks+[copy.deepcopy(cTask)], cScore)
        b3c.addCandidate(newCandidate)
        if firstIt and str(op)[28:60].startswith(startOps):
            newCandidate.generateTask()
            tryOperations(t, newCandidate)
        #elif str(op)[28:60].startswith(repeatIfPerfect) and c.score - changedPixels == cScore and changedPixels != 0:
        #    newCandidate.generateTask()
        #    tryOperations(t, newCandidate)

class Solution():
    def __init__(self, index, taskId, ops):
        self.index = index
        self.taskId = taskId
        self.ops = ops

# %% Solution Loop
solved = []
solvedIds = []
evolveTasks = [6,11,23,27,46,50,57,59,65,69,73,80,83,93,94,97,98,104,118,119,135,140,147,167,\
               170,189,198,201,224,229,231,236,242,247,254,255,257,267,279,282,283,285,287,298,322,\
               327,330,335,344,347,348,357,377,386,428,429,449,457,469,482,496,505,507,517,525,\
               526,531,552,573,577,579,585,605,607,629,631,633,646,648,661,678,679,693,703,706,731,748,\
               749,750,754,790,791,793,796,797]
count = 0
sameColorCountTasks = [3,7,29,31,43,52,77,86,121,127,139,149,153,154,178,227,240,\
                       244,249,269,300,352,372,379,389,434,447,456,501,502,512,\
                       516,545,555,556,560,567,601,613,615,638,641,660,719,733,\
                       737,741,743,746,756,781,782,784]
scctSolved = [7,31,52,86,139,149,154,178,240,249,269,372,379,556,719,741]

tasksWithFrames = [28, 74, 87, 90, 95, 104, 131, 136, 137, 142, 153, 158, 181, 182,\
                   188, 200, 204, 207, 208, 223, 227, 232, 237, 244, 245, 258, 272,\
                   273, 289, 296, 307, 309, 334, 345, 370, 382, 386, 389, 395, 419,\
                   437, 443, 445, 450, 451, 456, 458, 460, 462, 463, 466, 470, 472,\
                   475, 494, 495, 498, 535, 536, 588, 589, 593, 597, 610, 611, 618,\
                   621, 622, 623, 624, 625, 630, 634, 638, 640, 648, 650, 652, 669,\
                   672, 677, 678, 690, 699, 704, 710, 722, 726, 737, 742, 745, 758,\
                   760, 768, 779]

scctSolved = [7,31,52,86,139,149,154,178,240,249,269,372,379,556,719,741]
cropTasks = [13,30,35,48,56,78,110,120,133,173,176,206,217,262,270,289,299,345,383,488,576,578,635,712,727,785]
replicateShapes = [68, 645, 367, 421, 207, 362, 431, 494, 524]
#, 190, 367, 421, 431, 524
count=0
originalCount = 0
# 92,130,567,29,34,52,77,127
# 7,24,31,249,269,545,719,741,24,788
for idx in tqdm(range(800), position=0, leave=True):
    taskId = index[idx]
    task = allTasks[taskId]
    originalT = Task.Task(task, taskId)
       
    if needsRecoloring(originalT):
        task, trainRels, trainInvRels, testRels, testInvRels = orderTaskColors(originalT)
        t = Task.Task(task, taskId)
    else:
        t = originalT

    cTask = copy.deepcopy(task)
    if t.hasUnchangedGrid and t.gridCellsHaveOneColor:
        ignoreGrid(t, cTask) # This modifies cTask, ignoring the grid
        t2 = Task.Task(cTask, taskId)
    else:
        t2 = t

    cScore = sum([Utils.incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                         t2.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
    c = Candidate([], [task], score=cScore)
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
            tryOperations(t2, c, firstIt)
            if firstIt:
                firstIt = False
                break
        score = sum([c.score for c in b3c.candidates])
        if score >= prevScore:
            break
        else:
            prevScore = score

    # Once the best 3 candidates have been found, make the predictions
    for s in range(t.nTest):
        for c in b3c.candidates:
            print(c.ops)
            x = t2.testSamples[s].inMatrix.m.copy()
            for opI in range(len(c.ops)):
                newX = c.ops[opI](Task.Matrix(x))
                if t2.sameIOShapes and len(t2.fixedColors) != 0:
                    x = Utils.correctFixedColors(x, newX, t.fixedColors)
                else:
                    x = newX.copy()
            if t.hasUnchangedGrid and t.gridCellsHaveOneColor:
                x = recoverGrid(t, x)
            if needsRecoloring(originalT):
                x = recoverOriginalColors(x, testRels[s])
            plot_sample(originalT.testSamples[s], x)
            if Utils.incorrectPixels(x, t.testSamples[s].outMatrix.m) == 0:
                #print(idx)
                print(idx, c.ops)
                plot_task(task)
                count += 1
                break
                solved.append(Solution(idx, taskId, c.ops))
                solvedIds.append(idx)
                #break
