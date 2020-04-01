# %% Setup
import json
import numpy as np
import pandas as pd
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors

import Models
import Task
import Utils

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


allTasks = train_tasks.copy()
allTasks.update(valid_tasks)

df = pd.read_csv('info.csv', index_col=0)
index = list(allTasks.keys())

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
    if predict is None:
        plot_pictures([sample.inMatrix.m, sample.outMatrix.m], ['Input', 'Output'])
    else:
        plot_pictures([sample.inMatrix.m, sample.outMatrix.m, predict], ['Input', 'Output', 'Predict'])

def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp==i)
    return img

def plot_task2(task):
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
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

# %% Create DataFrame with TF info about each task

# Add new fields to the df
df["LinearModel"] = np.nan

def solve(task, model, dummify=False):
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()            
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
    for e in range(num_epochs):
        optimizer.zero_grad()
        loss = 0.0

        for s in task.trainSamples:
            # predict output from input
            x = s.inMatrix.m
            if dummify:
                x = inp2img(x).tolist()
            x = torch.tensor(x).unsqueeze(0).float()
            y = torch.tensor(s.outMatrix.m).long().unsqueeze(0)
            y_pred = model(x)
            loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

@torch.no_grad()
def predict(model, task):
    predictions = []
    for s in task.testSamples:
        x = s.inMatrix.m
        x = torch.from_numpy(x).unsqueeze(0).float()
        #x = torch.tensor(x).unsqueeze(0).float()
        pred = model(x).squeeze().numpy()
        #pred = model(x).argmax(1).squeeze().numpy()
        predictions.append(pred)
    return predictions

# Fill the matrix
#for i in tqdm(index, position=0, leave=True):
for idx in tqdm(range(0, 10), position=0, leave=True): 
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task)
    
    # Models
    if t.sameInShape and t.sameOutShape:
        if all(t.trainSamples[0].inMatrix.shape == s.inMatrix.shape for s in t.testSamples):       
            model = Models.LinearModel(t.trainSamples[0].inMatrix.shape, t.trainSamples[0].outMatrix.shape, 10)
            model = solve(t, model, True)
            pred = predict(model, t)
            score = [correctCells(pred, s.outMatrix.m) for s in t.testSamples]
            df.loc[i, "LinearModel"] = all(s == 0 for s in score)
        
len(np.where(df['LinearModel'] == True)[0])

# Read and write csv
df.to_csv('info.csv')
df2 = pd.read_csv('info.csv', index_col=0)

# New recursive approach
df["RecConvMultiModelPad1"] = np.nan
#for i in tqdm(index, position=0, leave=True):
correctK7 = []
for idx in tqdm(range(0, 800), position=0, leave=True): 
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
    
    if t.sameIOShapes:
        models = []
        num_epochs = 100
        losses = np.zeros(num_epochs)
        inTrain = [s.inMatrix.m for s in t.trainSamples]
        outTrain = [s.outMatrix.m for s in t.trainSamples]
        for nModel in range(10):
            kernel = 7# 3 if (nModel%2 != 0) else 5
            model = Models.OneConvModel(10, kernel, -1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            criterion = nn.CrossEntropyLoss()
            for e in range(num_epochs):
                optimizer.zero_grad()
                loss = 0.0
                for i in range(len(inTrain)):
                    x = inp2img(inTrain[i])
                    x = torch.tensor(x).unsqueeze(0).float()
                    y = torch.tensor(outTrain[i]).long().unsqueeze(0)
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                losses[e] = loss.item()
            models.append(model)
            newTrain = [model(torch.tensor(inp2img(x)).unsqueeze(0).float()).argmax(1).squeeze().numpy() for x in inTrain]
            if all(np.all(newTrain[i] == inTrain[i]) for i in range(len(inTrain))):
                break
            else:
                inTrain = newTrain
            
        predictions = []
        score = []
        with torch.no_grad():
            for s in t.testSamples:
                x = s.inMatrix.m
                for mm in models:
                    x = torch.tensor(inp2img(x)).unsqueeze(0).float()
                    x = mm(x)#.argmax(1)
                    x = x.argmax(1).squeeze().numpy()
                    score.append(correctCells(x, t.testSamples[0].outMatrix.m))
                pred = x
                predictions.append(pred)
            score = [correctCells(predictions[i], t.testSamples[i].outMatrix.m) for i in range(t.nTest)]
            #df.loc[i, "RecConvMultiModel"] = all(s == 0 for s in score)
            if all(s == 0 for s in score):
                correctK7.append(index[idx])

# %% Other things

# Symmetrization
solved = []
candidates = []
df["Symmetrize"] = np.nan
for i in tqdm(index, position=0, leave=True):
    task = allTasks[i]
    t = Task.Task(task, i)
    if t.lrSymmetric or t.udSymmetric or t.d1Symmetric:
        if len(t.changingColors) == 1:
            result = Models.Symmetrize(t, next(iter(t.changingColors)))
            for j in range(len(result)):
                if np.all(result[j] == t.testSamples[j].outMatrix.m):
                    df.loc[i, "Symmetrize"] = True
                
# %% Fully connected layer to determine 1-on-1 correspondence
# Solve with one fully connected layer
result = []
predictions = []
for i in tqdm(index):
    task = allTasks[i]
    t = Task.Task(task)
    
scores = []
predictions = []
for idx in tqdm(range(0, 10), position=0, leave=True):    
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
    if t.sameInShape and t.sameOutShape:
        model = Models.LinearModel(t.trainSamples[0].inMatrix.shape, t.trainSamples[0].outMatrix.shape, 1)
        num_epochs = 100
        criterion = nn.MSELoss()
        losses = np.zeros(num_epochs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0

            for s in t.trainSamples:
                x = s.inMatrix.m
                x = torch.tensor(x).unsqueeze(0).float()
                        
                y = torch.tensor(s.outMatrix.m).float().unsqueeze(0)
                y = y.view(-1, y.numel())
                             
                y_pred = model(x)
                loss += criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            for p in model.parameters():
                p.data.clamp_(min=0, max=1)
            with torch.no_grad():
                model.fc.weight *=  1 / model.fc.weight.max()
            losses[e] = loss.item()  
          
        with torch.no_grad():
            for s in t.testSamples:
                x = s.inMatrix.m
                x = torch.from_numpy(x).unsqueeze(0).float()
                pred = model(x).squeeze().numpy()
                pred = np.reshape(np.ceil(pred), t.outShape, order='F').astype(int)
                scores.append((i, correctCells(pred, s.outMatrix.m)))
                predictions.append((i, pred))
        
        
df['SameIOShapes'].iloc[0:10]
    
# %% Try 1-to-1 color and position correspondence
# Only valid if all shapes are equal
results = []
for idx in tqdm(range(0, 800), position=0, leave=True):    
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)                                  
    if t.sameIOShapes:
        result = Models.color1_2_1(t)
        if len(result) != 0:
            results.append((i, result))

df["color1_2_1"] = np.nan
df.loc['b1948b0a','color1_2_1'] = True
df.loc['c8f0f002','color1_2_1'] = True

# %% pixelCorrespondence

solved = []
for idx in tqdm(range(0, 800), position=0, leave=True):    
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i) 
    
    if t.sameInShape and t.sameOutShape and \
    all(t.trainSamples[0].inMatrix.shape == s.inMatrix.shape for s in t.testSamples):
        correspondence = Models.pixelCorrespondence(t)
        if len(correspondence) != 0:
            for s in t.testSamples:
                m =  s.inMatrix.m
                pred = np.zeros(t.outShape)
                for i,j in np.ndindex(t.outShape):
                    pred[i,j] = m[correspondence[i,j]]
                #plot_sample(s, pred)
                if np.all(s.outMatrix.m == pred):
                   solved.append(t.index)
    

        
# %% Predict output shape
df["Shape"] = np.nan
for i in tqdm(index, position=0, leave=True):
    task = allTasks[i]
    t = Task.Task(task, i)
    
    if t.sameOutShape:
        outShape = t.trainSamples[0].outMatrix.shape
        if t.sameInShape and outShape == t.trainSamples[0].inMatrix.shape:
            success = True
            for s in t.testSamples:
                if s.outMatrix.shape != s.inMatrix.shape:
                    success = False
                    break
            df.loc[i, "Shape"] = success
            continue
        else:
            success = True
            for s in t.testSamples:
                if s.outMatrix.shape != outShape:
                    success = False
                    break
            df.loc[i, "Shape"] = success
            continue
    
    if t.sameIOShapes:
        success = True
        for s in t.testSamples:
            if s.inMatrix.shape != s.outMatrix.shape:
                success = False
                break
        df.loc[i, "Shape"] = success
        continue
    
    df.loc[i, "Shape"] = False
    
# %% 1-on-1 correspondence, including color
scores = []
predictions = []
for idx in tqdm(range(0, 800), position=0, leave=True):    
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
    if t.sameInShape and t.sameOutShape:
        outShape = t.outShape
        inShape = t.inShape
        correspondence = {}
        solutionPossible = True
        for iOut, jOut in np.ndindex(outShape):
            corrFound = False
            for iIn, jIn in np.ndindex(inShape):
                corrCell = True
                for s in t.trainSamples:
                    if s.outMatrix.m[iOut, jOut] != s.inMatrix.m[iIn, jIn]:
                        corrCell = False
                        break
                if corrCell:
                    correspondence[(iOut, jOut)] = (iIn, jIn)
                    corrFound = True
                    break
            if not corrFound:
                solutionPossible = False
                break
            
        if solutionPossible:
            predictions = []
            for s in t.testSamples:
                pred = np.zeros(outShape)
                for iOut, jOut in np.ndindex(outShape):
                    pred[iOut, jOut] = s.inMatrix.m[correspondence[(iOut,jOut)][0], correspondence[(iOut,jOut)][1]]
                scores.append((i, correctCells(pred, s.outMatrix.m)))
                
# %% Adaptive convolutions
correctAC = []
for idx in tqdm(range(0, 800), position=0, leave=True): 
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
    
    # Models to try in every iteration
    # [Kernel, Value of padding]
    modelsToTry = [[3, 0], [3, -1], [5, 0], [5, -1], [7, 0], [7, -1]]
    
    if t.sameIOShapes:
        finalModels = []
        num_epochs = 20
        inTrain = [s.inMatrix.m for s in t.trainSamples]
        outTrain = [s.outMatrix.m for s in t.trainSamples]
        for nModel in range(5):
            candidateModels = []
            modelScores = []
            for mtt in modelsToTry:
                model = Models.OneConvModel(10, mtt[0], mtt[1])
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                criterion = nn.CrossEntropyLoss()
                for e in range(num_epochs):
                    optimizer.zero_grad()
                    loss = 0.0
                    for i in range(len(inTrain)):
                        x = inp2img(inTrain[i])
                        x = torch.tensor(x).unsqueeze(0).float()
                        y = torch.tensor(outTrain[i]).long().unsqueeze(0)
                        y_pred = model(x)
                        loss += criterion(y_pred, y)
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    sc = 0
                    for iS in range(len(inTrain)):
                        x = inp2img(inTrain[iS])
                        x = torch.tensor(x).unsqueeze(0).float()
                        x = model(x).argmax(1).squeeze().numpy()
                        sc += correctCells(x, outTrain[iS])
                    candidateModels.append(model)
                    modelScores.append(sc)
            nextModel = candidateModels[modelScores.index(min(modelScores))]
            finalModels.append(nextModel)
            inTrain = [nextModel(torch.tensor(inp2img(x)).unsqueeze(0).float()).argmax(1).squeeze().numpy() for x in inTrain]
            
        predictions = []
        score = []
        with torch.no_grad():
            for s in t.testSamples:
                x = s.inMatrix.m
                for fm in finalModels:
                    x = torch.tensor(inp2img(x)).unsqueeze(0).float()
                    x = fm(x)#.argmax(1)
                    x = x.argmax(1).squeeze().numpy()
                pred = x
                predictions.append(pred)
            score = [correctCells(predictions[i], t.testSamples[i].outMatrix.m) for i in range(t.nTest)]
            if all(s == 0 for s in score):
                correctAC.append(index[idx])
                
# %% Fully connected, but well done
scores = []
predictions = []
for idx in tqdm(range(0, 800), position=0, leave=True):    
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
    if t.sameInShape and t.sameOutShape:
        if all(t.trainSamples[0].inMatrix.shape == s.inMatrix.shape for s in t.testSamples):
            model = Models.LinearModel(t.trainSamples[0].inMatrix.shape, t.trainSamples[0].outMatrix.shape, 10)
            num_epochs = 10
            criterion = nn.CrossEntropyLoss()
            losses = np.zeros(num_epochs)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            for e in range(num_epochs):
                optimizer.zero_grad()
                loss = 0.0
    
                for s in t.trainSamples:
                    x = torch.tensor(inp2img(s.inMatrix.m)).unsqueeze(0).float()
                    y = torch.tensor(s.outMatrix.m).long().unsqueeze(0)
                    y = y.view(1, -1)
                                 
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
    
                loss.backward()
                optimizer.step()
                losses[e] = loss.item()
              
            with torch.no_grad():
                for s in t.testSamples:
                    x = inp2img(s.inMatrix.m)
                    x = torch.from_numpy(x).unsqueeze(0).float()
                    pred = model(x).argmax(1).squeeze().numpy()
                    pred = np.reshape(pred, t.outShape).astype(int)
                    scores.append((i, correctCells(pred, s.outMatrix.m)))
                    predictions.append((i, pred))
                    
# %% exploring deBackgroundized samples
scores = []
predictions = []

i = index[1]
task = allTasks[i]
t = Task.Task(task, i)

if t.sameInShape and t.sameOutShape:
    inShape = t.trainSamples[0].inMatrix.shape
    outShape = t.trainSamples[0].outMatrix.shape
    model = Models.LinearModel(inShape, outShape, 2)
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros(num_epochs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    trainIn, trainOut = Utils.deBackgroundize(t)
    rel, invRel = Utils.relDicts([0, 1])
    trainIn = [Utils.dummify(x, 2, rel) for x in trainIn]
    for e in range(num_epochs):
        optimizer.zero_grad()
        loss = 0.0

        for nSample in range(len(trainIn)):
            x = torch.tensor(trainIn[nSample]).unsqueeze(0).float()
            y = torch.tensor(trainOut[nSample]).long().unsqueeze(0)
            y = y.view(1, -1)
                                 
            y_pred = model(x)
            loss += criterion(y_pred, y)
    
        loss.backward()
        optimizer.step()
        losses[e] = loss.item()
        
    with torch.no_grad():
        for s in t.testSamples:
            x = Utils.deBackgroundizeMatrix(s.inMatrix.m, t.backgroundColor)
            x = Utils.dummify(x, 2, rel)
            x = torch.from_numpy(x).unsqueeze(0).float()
            pred = model(x).argmax(1).view(outShape[0], outShape[1]).numpy()
            pred *= list(s.inMatrix.colorCount.keys())[1] 
            scores.append((i, Utils.correctCells(pred, s.outMatrix.m)))
            predictions.append((i, pred))
            
# %% Exploring dummifying each color one by one (always 2 output classes)           

# Solves 9 with kernel=5, pad=-1:
# ['025d127b','25ff71a9','25ff71a9','6f8cd79b','794b24be','91714a58',
#  'd037b0a7','66e6c45b','fc754716']
            
# Solves 7 with kernel=3, pad=-1:
# ['25ff71a9','25ff71a9','4347f46a','6f8cd79b','7f4411dc','66e6c45b',
#  'fc754716']

# Solves 11 with kernel=7, pad=-1:
# ['25ff71a9','25ff71a9','3618c87e','6f8cd79b','794b24be','91714a58',
#  'c3f564a4','d037b0a7','e9afcf9a','66e6c45b','fc754716']
            
# Solves 7 with kernel=9, pad=0:
# ['25ff71a9','3618c87e','4347f46a','5582e5ca','c3f564a4','d037b0a7',
#  'e9afcf9a']
# Only one new one, but because of the pad, not the kernel
         
# %% This solves task 29
solved = [] 
for idx in tqdm(range(0, 800), position=0, leave=True):    
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
    
    if t.sameIOShapes:
        if len(t.unchangedColors) != 0:
            colors = []
            for c in t.unchangedColors:
                colors.append(c)
            if t.backgroundColor != -1 and\
            t.backgroundColor not in t.unchangedColors:
                colors.append(t.backgroundColor)
            
            model = Models.SimpleLinearModel(4, 2)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            num_epochs = 100
            losses = np.zeros(num_epochs)
            for e in range(num_epochs):
                optimizer.zero_grad()
                loss = 0.0
                for s in t.trainSamples:
                    inFeatures = []
                    for shape in s.inMatrix.dShapes:
                        if shape.color == next(iter(t.unchangedColors)):
                            inFeatures.append(shape.position[0])
                            inFeatures.append(shape.position[1])
                    for c in set(s.colors) - set(colors):
                        for shape in s.inMatrix.dShapes:
                            if shape.color == c:
                                x = inFeatures.copy()
                                x.append(shape.position[0])
                                x.append(shape.position[1])
                                outFeatures = []
                                for outShape in s.outMatrix.dShapes:
                                    if outShape.color == c:
                                        outFeatures.append(outShape.position[0])
                                        outFeatures.append(outShape.position[1])
                                y = torch.tensor(outFeatures).unsqueeze(0).float()
                                x = torch.tensor(x).unsqueeze(0).float()
                                y_pred = model(x).squeeze()
                                loss += criterion(y, y_pred)
                loss.backward()
                optimizer.step()
                losses[e] = loss.item()
            
            with torch.no_grad():
                for s in t.trainSamples+t.testSamples:
                    inFeatures = []
                    for shape in s.inMatrix.dShapes:
                        if shape.color == next(iter(t.unchangedColors)):
                            inFeatures.append(shape.position[0])
                            inFeatures.append(shape.position[1])
                    pred = s.inMatrix.m.copy()
                    for c in set(s.colors) - set(colors):
                        for shape in s.inMatrix.dShapes:
                            if shape.color == c:
                                x = inFeatures.copy()
                                x.append(shape.position[0])
                                x.append(shape.position[1])
                                x = torch.tensor(x).unsqueeze(0).float()
                                y = model(x).squeeze()
                                pred = Utils.deleteShape(pred, shape, t.backgroundColor)
                                cShape = copy.deepcopy(shape)
                                y = torch.tensor(y.round(),dtype=torch.int8)
                                cShape.position = tuple(y.tolist())
                                pred = Utils.insertShape(pred, cShape)
                    plot_sample(s, pred)
                
# %%
        
    #   model = Models.OneConvModel(2, 5, 0)
    if t.sameInShape and t.sameOutShape and all(t.trainSamples[0].inMatrix.shape == s.inMatrix.shape for s in t.testSamples):
        model = Models.LinearModelDummy(t.inShape, t.outShape)
        num_epochs = 100
        criterion = nn.CrossEntropyLoss()
        losses = np.zeros(num_epochs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        for e in range(num_epochs):
            
            optimizer.zero_grad()
            loss = 0.0
            
            for s in t.trainSamples:
                for c in s.colors:
                    if c != 0:
                    #if c != t.backgroundColor:
                        x = Utils.dummifyColor(s.inMatrix.m, c)
                        x = torch.tensor(x).unsqueeze(0).float()
                        y = Utils.deBackgroundizeMatrix(s.outMatrix.m, c)
                        y = torch.tensor(y).unsqueeze(0).long()
                        y = y.view(1, -1)
                                         
                        y_pred = model(x)
                        loss += criterion(y_pred, y)
            
            loss.backward()
            optimizer.step()
            #for p in model.parameters():
            #    p.data.clamp_(min=0, max=1)
            losses[e] = loss.item()
            
        with torch.no_grad():
            for s in t.testSamples:
                pred = np.zeros(s.outMatrix.shape)
                #pred = np.ones(s.inMatrix.shape) * t.backgroundColor
                for c in s.colors:
                    if c != 0:
                    #if c != t.backgroundColor:
                        x = Utils.dummifyColor(s.inMatrix.m, c)
                        x = torch.tensor(x).unsqueeze(0).float()
                        x = model(x).argmax(1).squeeze().view(s.outMatrix.shape).numpy()
                        for i,j in np.ndindex(s.outMatrix.shape):
                            if x[i,j] != 0:
                                pred[i,j] = c
                #plot_sample(s, pred)
                if np.all(s.outMatrix.m == pred):
                    solved.append(t.index)
                    
# %% Dummifying the colors that are common to all samples
solved = [] 
for idx in tqdm(range(0, 800), position=0, leave=True):    
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
        
    if t.sameIOShapes and t.sameNSampleColors:
        nColors = len(t.sampleColors[0])
        model = Models.OneConvModel(nColors, 3, 0)
        num_epochs = 200
        criterion = nn.CrossEntropyLoss()
        losses = np.zeros(num_epochs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0
            for s in t.trainSamples:
                sColors = t.orderedColors.copy()
                for c in s.colors:
                    if c not in sColors:
                        sColors.append(c)
                rel, invRel = Utils.relDicts(sColors)
                x = Utils.dummify(s.inMatrix.m, nColors, rel)
                x = torch.tensor(x).unsqueeze(0).float()
                y = s.outMatrix.m.copy()
                for i,j in np.ndindex(y.shape):
                    y[i,j] = invRel[y[i,j]]
                y = torch.tensor(y).unsqueeze(0).long()
                y_pred = model(x)
                loss += criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            losses[e] = loss.item()
            
        with torch.no_grad():
            for s in t.trainSamples+t.testSamples:
                sColors = t.orderedColors.copy()
                pred = np.zeros(s.outMatrix.shape)
                for c in s.colors:
                    if c not in sColors:
                        sColors.append(c)
                if len(sColors) > nColors:
                    break
                rel, invRel = Utils.relDicts(sColors)
                x = Utils.dummify(s.inMatrix.m, nColors, rel)
                x = torch.tensor(x).unsqueeze(0).float()
                x = model(x).argmax(1).squeeze().numpy()
                for i,j in np.ndindex(s.outMatrix.shape):
                    if x[i,j] in rel.keys():
                        pred[i,j] = rel[x[i,j]]
                    else:
                        pred[i,j] = x[i,j]
                plot_sample(s, pred)
                #if np.all(s.outMatrix.m == pred):
                #   solved.append(t.index)
                
# %% Solving Task 728
# If sameOutShape
                
solved = []
for idx in tqdm(range(0, 800), position=0, leave=True): 
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
    
    if t.sameOutShape:
        

# %% Problem that only requires color changes:
# There are 69 tasks that satisfy the given requirements
# Solves 8/69 for dShapes:
# ['0d3d703e','44d8ac46','7b6016b9','83302e8f','a5313dff','b1948b0a',
#  'c8f0f002','ae58858e']
# Solves 13/69 for shapes:
# ['00d62c1b','0d3d703e','44d8ac46','67385a82','7b6016b9','83302e8f',
#  'a5313dff','aedd82e4','b1948b0a','c8f0f002','e8593010','84db8fc4',
#  'ae58858e']
# Solved for dShapes ---> Solved for shapes

solved = []
for idx in tqdm(range(0, 800), position=0, leave=True): 
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
    
    # All the conditions:
    if t.sameIOShapes:
        isGood = True
        for s in t.trainSamples:
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
        # Learn taking the shapes as training inputs
        if isGood:
            rel, invRel = Utils.relDicts(list(t.colors))
            # Define the features:
            # One per color (1 or 0), nCells, isSquare, isOpen
            nInFeatures = t.nColors + 7
            # Define training parameters and train
            model = Models.SimpleLinearModel(nInFeatures, t.nColors)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            num_epochs = 80
            losses = np.zeros(num_epochs)
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
                    inFeatures[t.nColors+1] = s.isSquare().item()
                    inFeatures[t.nColors+2] = s.isRectangle().item()
                    inFeatures[t.nColors+3] = s.position[0].item()
                    inFeatures[t.nColors+4] = s.position[1].item()
                    inFeatures[t.nColors+5] = s.isBorder
                    inFeatures[t.nColors+6] = s.nHoles
                    y = torch.tensor(invRel[label]).unsqueeze(0).long()
                    x = inFeatures.unsqueeze(0).float()
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                losses[e] = loss.item()
            
            with torch.no_grad():
                for s in t.testSamples:
                    pred = s.inMatrix.m.copy()
                    for shape in s.inMatrix.shapes:
                        if shape.color in t.colors:
                            inFeatures = torch.zeros(nInFeatures)
                            inFeatures[invRel[shape.color]] = 1
                            inFeatures[t.nColors] = shape.nCells
                            inFeatures[t.nColors+1] = shape.isSquare().item()
                            inFeatures[t.nColors+2] = shape.isRectangle().item()
                            inFeatures[t.nColors+3] = shape.position[0].item()
                            inFeatures[t.nColors+4] = shape.position[1].item()
                            inFeatures[t.nColors+5] = shape.isBorder
                            inFeatures[t.nColors+6] = shape.nHoles
                            x = inFeatures.unsqueeze(0).float()
                            y = model(x).squeeze().argmax().item()
                            pred = Utils.changeColorShape(pred, shape, rel[y])
                    #plot_sample(s, pred)
                    if np.all(pred == s.outMatrix.m):
                        solved.append(i)
                

# %% DSL + RL approach
class Candidate():
    def __init__(self, ops, tasks, score=1000):
        self.ops = ops
        self.score = score
        self.tasks = tasks
        self.t = None
    
    def __lt__(self, other):
        return self.score < other.score
    
    def generateTask(self):
        self.t = Task.Task(self.tasks[-1], 'dummyIndex')
    
class Best3Candidates():
    def __init__(self, Candidate1, Candidate2, Candidate3):
        self.candidates = [Candidate1, Candidate2, Candidate3]
        
    def maxCandidate(self):
        x = 0
        if self.candidates[1] > self.candidates[0]:
            x = 1
        if self.candidates[2] > self.candidates[x]:
            x = 2
        return x
        
    def addCandidate(self, c):
        iMaxCand = self.maxCandidate()
        for i in range(3):
            if c < self.candidates[iMaxCand]:
                c.generateTask()
                self.candidates[iMaxCand] = c
                break
 
toBeSolved = set()
for i in index:
    t = Task.Task(allTasks[i], i)
    if t.sameIOShapes and t.sameColorCount:
        toBeSolved.add(i)
        
class Solution():
    def __init__(self, index, ops):
        self.index = index
        self.ops = ops

solved = []

for idx in tqdm(range(50), position=0, leave=True): 
    i = index[idx]
    task = allTasks[i]
    t = Task.Task(task, i)
    
    if t.hasUnchangedGrid and t.gridCellsHaveOneColor:
        cTask = copy.deepcopy(task)
        for s in range(t.nTrain):
            m = np.zeros(t.trainSamples[s].inMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.trainSamples[s].inMatrix.grid.cells[i][j][0].colors))
            cTask["train"][s]["input"] = m.tolist()
            m = np.zeros(t.trainSamples[s].outMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.trainSamples[s].outMatrix.grid.cells[i][j][0].colors))
            cTask["train"][s]["output"] = m.tolist()
        for s in range(t.nTest):
            m = np.zeros(t.testSamples[s].inMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].inMatrix.grid.cells[i][j][0].colors))
            cTask["test"][s]["input"] = m.tolist()
            m = np.zeros(t.testSamples[s].outMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].outMatrix.grid.cells[i][j][0].colors))
            cTask["test"][s]["output"] = m.tolist()
        t2 = Task.Task(cTask, i)
    else:
        #continue
        t2 = t
        cTask = copy.deepcopy(task)
        
    c = Candidate([], [task])
    c.t = t2
    b3c = Best3Candidates(c, c, c)
    
    prevScore = sum([c.score for c in b3c.candidates])
    firstIt = True
    while True:
        copyB3C = copy.deepcopy(b3c)
        for c in copyB3C.candidates:
            if c.score == 0:
                continue
            possibleOps = Utils.getPossibleOperations(t2, c)
            for op in possibleOps:
                cScore = 0
                for s in range(t.nTrain):
                    cTask["train"][s]["input"] = op(c.t.trainSamples[s].inMatrix).tolist()
                    if t2.sameIOShapes and len(t2.unchangedColors) != 0:
                        cTask["train"][s]["input"] = Utils.correctUnchangedColors(\
                             c.t.trainSamples[s].inMatrix.m,\
                             np.array(cTask["train"][s]["input"]),\
                             t2.unchangedColors).tolist()
                for s in range(t.nTest):
                    cTask["test"][s]["input"] = op(c.t.testSamples[s].inMatrix).tolist()
                    if t2.sameIOShapes and len(t2.unchangedColors) != 0:
                        cTask["test"][s]["input"] = Utils.correctUnchangedColors(\
                             c.t.testSamples[s].inMatrix.m,\
                             np.array(cTask["test"][s]["input"]),\
                             t2.unchangedColors).tolist()
                cScore += sum([Utils.correctCells(np.array(cTask["train"][s]["input"]), \
                                                  t2.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
                b3c.addCandidate(Candidate(c.ops+[op], c.tasks+[copy.deepcopy(cTask)], cScore))
            if firstIt:
                firstIt = False
                break
        score = sum([c.score for c in b3c.candidates])
        if score >= prevScore:
            break
        else:
            prevScore = score
            
    for s in range(t.nTest):
        for c in b3c.candidates:
            #print(c.ops)
            x = t2.testSamples[s].inMatrix.m.copy()
            for opI in range(len(c.ops)):
                newX = c.ops[opI](Task.Matrix(x))
                if t2.sameIOShapes and len(t2.unchangedColors) != 0:
                    x = Utils.correctUnchangedColors(x, newX, t.unchangedColors)
                else:
                    x = newX.copy()
            if t.hasUnchangedGrid and t.gridCellsHaveOneColor:
                realX = t.testSamples[s].inMatrix.m.copy()
                cells = t.testSamples[s].inMatrix.grid.cells
                for cellI in range(len(cells)):
                    for cellJ in range(len(cells[0])):
                        cellShape = cells[cellI][cellJ][0].shape
                        position = cells[cellI][cellJ][1]
                        for k,l in np.ndindex(cellShape):
                            realX[position[0]+k, position[1]+l] = x[cellI,cellJ]
                x = realX
            #plot_sample(t.testSamples[s], x)
            #if Utils.correctCells(x, s.outMatrix.m) == 0:
            #    plot_task2(task)
            #    solved.append(Solution(i, c.ops))
            #    break