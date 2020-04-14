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
train_tasks['cbded52d']['train'][0]['input'][4][7] = 2
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
aux = train_tasks['150deff5']['test'][0]['output']
train_tasks['150deff5']['test'][0]['output'] = train_tasks['150deff5']['test'][0]['input']
train_tasks['150deff5']['test'][0]['input'] = aux
# 17cae0c1
for i in range(3):
    for j in range(3, 6):
        valid_tasks['17cae0c1']['test'][0]['output'][i][j] = 9
# e48d4e1a
train_tasks['e48d4e1a']['train'][3]['input'][0][9] = 5
train_tasks['e48d4e1a']['train'][3]['output'][0][9] = 0

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

def tryOperations(t, c):
    """
    Given a Task.Task t and a Candidate c, this function applies all the
    operations that make sense to the input matrices of c. After a certain
    operation is performed to all the input matrices, a new candidate is
    generated from the resulting output matrices. If the score of the candidate
    improves the score of any of the 3 best candidates, it will be saved in the
    variable b3c, which is an object of the class Best3Candidates.
    """
    possibleOps = Utils.getPossibleOperations(t, c)
    for op in possibleOps:
        cScore = 0
        for s in range(t.nTrain):
            cTask["train"][s]["input"] = op(c.t.trainSamples[s].inMatrix).tolist()
            if t.sameIOShapes and len(t.unchangedColors) != 0:
                cTask["train"][s]["input"] = Utils.correctUnchangedColors(\
                     c.t.trainSamples[s].inMatrix.m,\
                     np.array(cTask["train"][s]["input"]),\
                     t.unchangedColors).tolist()
        for s in range(t.nTest):
            cTask["test"][s]["input"] = op(c.t.testSamples[s].inMatrix).tolist()
            if t.sameIOShapes and len(t.unchangedColors) != 0:
                cTask["test"][s]["input"] = Utils.correctUnchangedColors(\
                     c.t.testSamples[s].inMatrix.m,\
                     np.array(cTask["test"][s]["input"]),\
                     t.unchangedColors).tolist()
        cScore += sum([Utils.incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                          t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        newCandidate = Candidate(c.ops+[op], c.tasks+[copy.deepcopy(cTask)], cScore)
        b3c.addCandidate(newCandidate)

class Solution():
    def __init__(self, index, taskId, ops):
        self.index = index
        self.taskId = taskId
        self.ops = ops

# %% Solution Loop
solved = []

targetedTasks = [6,11,23,27,46,50,57,65,69,73,80,83,93,94,104,118,135,140,167,\
                 170,189,198,224,229,242,254,255,257,267,279,282,285,287,298,322,\
                 330,335,344,347,348,377,386,428,449,457,482,496,507,517,525,\
                 526,531,552,573,579,585,607,629,631,648,678,703,706,731,731,\
                 750,790,791,796,797]

count = 0
for idx in tqdm([719], position=0, leave=True):
    taskId = index[idx]
    task = allTasks[taskId]
    t = Task.Task(task, taskId)

    cTask = copy.deepcopy(task)
    if t.hasUnchangedGrid and t.gridCellsHaveOneColor:
        ignoreGrid(t, cTask) # This modifies cTask, ignoring the grid
        t2 = Task.Task(cTask, taskId)
    else:
        t2 = t

    c = Candidate([], [task])
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
            tryOperations(t2, c)
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
                if t2.sameIOShapes and len(t2.unchangedColors) != 0:
                    x = Utils.correctUnchangedColors(x, newX, t.unchangedColors)
                else:
                    x = newX.copy()
            if t.hasUnchangedGrid and t.gridCellsHaveOneColor:
                x = recoverGrid(t, x)
            plot_sample(t.testSamples[s], x)
            if Utils.incorrectPixels(x, t.testSamples[s].outMatrix.m) == 0:
                print(idx)
                print(str(c.ops)[18:50])
                plot_task(task)
                #break
                #solved.append(Solution(idx, taskId, c.ops))
                #break
