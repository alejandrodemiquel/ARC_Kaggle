# ARC Kaggle competition

This repository contains part of the code that was used for the second position at the "Abstraction and Reasoning Challenge" Kaggle competition. It's the part of the algorithm that was developed by [Alejandro](https://github.com/alejandrodemiquel) and [Roderic](https://github.com/RodericGuigoCorominas). submissionFile.py, when executed on the private test set, gets a score of 0.882, meaning that it solves 12 tasks of the private test set. The extra 7 tasks that made us have a score of 0.813 were solved as a result of merging teams with [Yuji](https://github.com/yujiariyasu).

## Executing the program
Cloning the repository and executing the file submissionFile.py will run the algorithm on the test data. The bit where everything is executed is at the end of the file ("Main Loop and submission").

## Other files
We've been mainly working using the files main.py, Task.py, Models.py and Utils.py. We only used submissionFile.py for submitting the program to Kaggle. But submissionFile.py already contains everything that is needed to run the algorithm, so one doesn't really need to look into these files.
main.py is the entry point; it contains some basic functions and the main loop. Models.py stores some training models that we tried with very tiny success. Task.py contains the core objects that store all the information about the task to be solved. Utils.py contains all the other functions.

## The algorithm

For every new task that we want to solve, the algorithm follows these steps:
1. Generate an object of the class `Task`. At this step, all the preprocessing is done: checking for matrix shapes, common colors, symmetries... There are 6 core classes that are used to store information about the preprocessing: The classes `Task`, `Sample`, `Matrix`, `Shape`, `Grid` and `Frontier`. They can also be found in the file Task.py. 
2. Based on the information retrieved from the preprocessing, the task might be transformed into something that can be processed more easily. It might make sense to modify some colors, rotate some of the matrices, crop the background or ignore some grid, for example.
3. Once the transformation is done, we generate 3 dummy candidates and enter the loop to get the 3 best candidates. In every iteration of the loop, for each of the current best 3 candidates, we try to execute all sorts of operations that make sense to try, given the properties stored in the object of the class `Task`. Many different sorts of functions are tried, such as "moveShapes", "replicateShapes", "extendColor" or "pixelwiseXorInGridSubmatrices", just to mention a few examples. A complete list can be found below. If any function generates a candidate with a better score than any of the current best 3 candidates, we include this new candidate into our list of best 3 candidates, and remove the worst one. The score is simply computed by executing the functions that led to generate that candidate in the training samples, and checking how many pixels are incorrect. Thus, a score of 0 is the best possible one.
4. Revert the operations executed in step 2 to from the 3 candidates, to obtain the final candidates.
5. If it makes sense, we generate another task either imposing that in every sample there can be either only one shape or only one color. Obviously, whe then have more training and test samples than in the original task. Then, we also compute the three best candidates for this version, and compare them with the previously generated ones to select the final best candidates.

### List of the functions used
This is a list of the functions that might be executed for each candidate in `getPossibleOperations`. Many of them are executed several times, with different parameters. All of them contribute in solving at least one of the given task in the training or the evaluation set.
- fillTheBlank
- downsize
- downsizeMode
- arrangeShapes
- countColors
- countShapes
- colorMap
- symmetrize
- colorSymmetricPixels
- completeRectangles
- changeShapesWithFeatures
- lstm
- changeShapes
- replicateShapes
- colorByPixels
- followPattern
- CNN
- mirror
- rotate
- moveShapes
- pixelRecolor (taken from [this public notebook](https://www.kaggle.com/szabo7zoltan/colorandcountingmoduloq))
- extendColor
- surroundShapes
- paintShapesInHalf
- paintShapeFromBorderColor
- colorLongestLines
- connectAnyPixels
- subMatToLayer
- alignShapes
- symmetrizeSubmatrix
- symmetrizeShapes
- colorByPixels
- overlapSubmatrices
- evolve
- evolvingLine
- paintCrossedCoordinates
- extendMatrix
- multiplyPixels
- multiplyMatrix
- pixelwiseAnd/Or/Xor in grid cells / submatrices
- generateMosaic
- selectSubmatrix
- maxColorFromCell
- cropShape
- moveToPanel
- cropFullFrame
- cropPartialFrame
- paintGridLikeBackground
- cropAllBackground
- switchColors
- minimize
