'''
Evaluates topics/baseline bag of words in in predicting some dependent
variable.  The dependent variable is either a state-level (proportion)
variable per tweet, or the stance of the tweet (hand-coded).  The fold that
the topic model was trained on is encoded in the file name.

Adrian Benton
1/19/2015
'''

from math import log
import re, sys

import linreg
from personal.fmethods import ldList

from sklearn.linear_model import Ridge, LinearRegression
import numpy as np
import scipy as sp

PATH_FOLD_RE = re.compile('\.pred\.(?P<fold>\d+)\.')
MODELS = ['sprite', 'lda', 'unsupervised']
#FOLDS  = [0, 1, 2, 3, 4]
FOLDS  = [0]

def ldWords(inputPath):
  wToI = {}
  iToW = {}
  idToBow = {}
  currIdx = 0
  
  f = open(inputPath)
  for ln in f:
    flds = ln.strip().split()
    id = int(flds[0])
    fold = int(flds[1])
    
    if fold < 0:
      continue
    
    idToBow[id] = {}
    for wStr in flds[5:]:
      w = wStr
      
      if w not in wToI:
        wToI[w] = currIdx
        iToW[currIdx] = w
        currIdx += 1
      
      if wToI[w] not in idToBow[id]:
        idToBow[id][wToI[w]] = 0.0
      idToBow[id][wToI[w]]  += 1.0
  f.close()
  
  return wToI, iToW, idToBow

def toTfIdf(iToW, bow): # Converts a count bag of words to TF-IDF
  tfidfBow = {}
  N_d = dict([(i, 0.) for i in iToW])
  D = float(len(bow))
  idf = dict([(i, D) for i in iToW])
  
  for id, wMap in bow.items():
    for w in wMap:
      N_d[w] += 1.0
  for i in idf:
    idf[i] = log(D/(1.0 + N_d[i]))
  
  for id, wMap in bow.items():
    tfidfBow[id] = dict([(w, v * idf[w]) for w, v in wMap.items()])
  
  return tfidfBow

def getTopicDistributions(assignPath, idToFold):
  '''
  For each tweet in the output file, this guesses the topic distribution
  directly based on the sampled counts, which are then used as features in a
  regression model.
  '''
  
  idToTopicDist = {}
  f = open(assignPath)
  for line in f:
    flds = line.strip().split()
    tid = float(flds[1])
    topicDist = []
    N = 0
    
    if (tid not in idToFold) or (idToFold[tid] < 0):
      continue
    
    assignFlds = flds[5:]
    for wordAssn in assignFlds:
      counts = [int(v) for v in wordAssn.split(':')[1:]
                if re.match('\d+', v)]
      if not topicDist:
        topicDist = [0.]*len(counts)
      else:
        counts = counts[-len(topicDist):]
      
      for idx, c in enumerate(counts):
        topicDist[idx] += c
        N += c
    topicDist = [v/N for v in topicDist]
    idToTopicDist[tid] = topicDist
  f.close()
  
  return idToTopicDist

def ldFolds(foldPath):
  idToFold = {}
  
  f = open(foldPath)
  for line in f:
    flds = line.strip().split()
    tid = int(flds[0])
    fold = int(flds[1])
    idToFold[tid] = fold
  f.close()
  
  return idToFold

def splitTrainDev(idToFeatures, idToFold, devFold):
  trainFold = []
  devFold   = []
  for id, features in idToFeatures.items():
    fold = idToFold[id]
    if fold != devFold:
      trainFold.append(features)
    else:
      devFold.append(features)
  
  return (trainFold, devFold)

def bowToNumpy(trainDevSplit):
  maxIdx = 0
  for split in [0, 1]:
    for bowMap in trainDevSplit[0].values():
      for featIdx, featValue in bowMap.items():
        if featIdx > maxIdx:
          maxIdx = featIdx
  
  trainSorted = sorted(trainDevSplit[0].items())
  devSorted   = sorted(trainDevSplit[1].items())
  
  trainMatrix = np.zeros((len(trainSorted), maxIdx+1))
  devMatrix   = np.zeros((len(devSorted), maxIdx+1))
  
  for rowIdx, (tid, bowMap) in enumerate(trainSorted):
    for colIdx, featValue in bowMap.items():
      trainMatrix[rowIdx,colIdx] = featValue
  for rowIdx, (tid,bowMap) in enumerate(devSorted):
    for colIdx, featValue in bowMap.items():
      devMatrix[rowIdx,colIdx] = featValue
  
  npTrainDevSplit = (trainMatrix, devMatrix)
  
  return npTrainDevSplit

def topicToNumpy(trainDevSplit):
  maxIdx = len(trainDevSplit[0].values()[0])
  
  trainSorted = sorted(trainDevSplit[0].items())
  devSorted   = sorted(trainDevSplit[1].items())
  
  trainMatrix = np.zeros((len(trainDevSplit[0]), maxIdx+1))
  devMatrix   = np.zeros((len(trainDevSplit[1]), maxIdx+1))
  
  for rowIdx, (tid, row) in enumerate(trainSorted):
    for colIdx, featValue in enumerate(row):
      trainMatrix[rowIdx,colIdx] = featValue
  for rowIdx, (tid, row) in enumerate(devSorted):
    for colIdx, featValue in enumerate(row):
      devMatrix[rowIdx,colIdx] = featValue
  npTrainDevSplit = (trainMatrix, devMatrix)
  
  return npTrainDevSplit

def evalBow(bowFeatureFolds=[], yPerFold=[],
            Cs=[0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]):
  
  avgMSE = [float('inf')]*len(Cs)
  avgMAE = [float('inf')]*len(Cs)
  
  for cIdx, c in enumerate(Cs):
    runningMSE = 0.0
    runningMAE = 0.0
    for trainDevSplit, ySplit in zip(bowFeatureFolds, yPerFold):
      trainFeatures = trainDevSplit[0]
      devFeatures   = trainDevSplit[1]
      
      trainY = ySplit[0]
      devY   = ySplit[1]
      
      import pdb; pdb.set_trace()
      
      if c:
        ridgeModel = Ridge(alpha=1.0/(2*c))
      else:
        ridgeModel = LinearRegression()
      
      ridgeModel.fit(trainFeatures, trainY)
      devY_hat   = ridgeModel.predict(devFeatures)
      
      mse = (devY - devY_hat)**2.0/devY.shape[0]
      mae = np.fabs(devY - devY_hat)/devY.shape[0]
      runningMSE += mse
      runningMAE += mae
    
    avgMSE[cIdx] = runningMSE/len(yPerFold)
    avgMAE[cIdx] = runningMAE/len(yPerFold)
    
    print 'BOW Regression No.', cIdx
    
  return avgMSE, avgMAE

def evalTopics(topicFeatureFolds=[], yPerFold=[]):
  avgMSE = 0.0
  avgMAE = 0.0
  
  for trainDevSplit, ySplit in zip(bowFeatureFolds, yPerFold):
    trainFeatures = trainDevSplit[0]
    devFeatures   = trainDevSplit[1]
    
    trainY = ySplit[0]
    devY   = ySplit[1]
    
    model = LinearRegression()
    model.fit(trainFeatures, trainY)
    devY_hat = model.predict(devFeatures)
    
    mse = (devY - devY_hat)**2.0/devY.shape[0]
    mae = np.fabs(devY - devY_hat)/devY.shape[0]
    avgMSE += mse
    avgMAE += mae
  
  avgMSE = avgMSE/len(yPerFold)
  avgMAE = avgMAE/len(yPerFold)
  
  return avgMSE, avgMAE

def ldY(yPath):
  y = dict([(int(ln.strip().split()[0]),
             float(ln.strip().split()[1])) for ln in ldList(yPath)])
  
  return y

def yToNumpy(ySplit):
  trainY = np.zeros((len(ySplit[0]),1))
  devY   = np.zeros((len(ySplit[1]),1))
  
  trainSorted = sorted(ySplit[0].items())
  devSorted   = sorted(ySplit[1].items())
  
  for rowIdx, keyValue in enumerate(trainSorted):
    trainY[rowIdx,0] = keyValue[1]
  
  for rowIdx, keyValue in enumerate(devSorted):
    devY[rowIdx,0] = keyValue[1]
  
  return (trainY, devY)

def ldData(inputPath, assignFmt, yPath):
  '''
  Loads feature vectors and dependent variables for:
  
  - Bag of words counts
  - Bag of words TF-IDF
  - Topic distribution for no scores case
  - Topic distribution for all scores case 
  
  Each of these are a list of train/dev pairs (based on number of folds).
  '''
  
  idToFold = ldFolds(inputPath)
  wtoi, itow, bow = ldWords(inputPath)
  print 'Loaded BOW'
  
  tfidfBow = toTfIdf(itow, bow)
  print 'Converted to TF-IDF'
  
  ys = ldY(yPath)
  
  bowSplits    = []
  tfidfSplits  = []
  spriteSplits = []
  ldaSplits    = []
  unsupSplits  = []
  ySplits      = []
  for devFold in FOLDS:
    bowSplit = (dict([(id, values)
                      for id, values in bow.items()
                      if idToFold[id]!=devFold]),
                dict([(id, values)
                      for id, values in bow.items()
                      if idToFold[id]==devFold]))
    bowSplit = bowToNumpy(bowSplit)
    
    tfidfSplit = (dict([(id, values)
                        for id, values in tfidfBow.items()
                        if idToFold[id]!=devFold]),
                  dict([(id, values)
                        for id, values in tfidfBow.items()
                        if idToFold[id]==devFold]))
    tfidfSplit = bowToNumpy(tfidfSplit)
    
    ySplit = (dict([(id, values) for id, values in ys.items()
                    if idToFold[id]!=devFold]),
              dict([(id, values) for id, values in ys.items()
                    if idToFold[id]==devFold]))
    ySplit = yToNumpy(ySplit)
    
    bowSplits.append(bowSplit)
    tfidfSplits.append(tfidfSplit)
    ySplits.append(ySplit)
    
    for model in MODELS:
      assignPath = assignFmt % (model, devFold)
      
      idToTopicDist = getTopicDistributions(assignPath, idToFold)
      
      topicSplit = (dict([(id, values) for id, values
                          in idToTopicDist.items()
                          if idToFold[id]!=devFold]),
                    dict([(id, values) for id, values
                          in idToTopicDist.items()
                          if idToFold[id]==devFold]))
      topicSplit = topicToNumpy(topicSplit)
      
      if model == 'sprite':
        spriteSplits.append(topicSplit)
      elif model == 'lda':
        ldaSplits.append(topicSplit)
      elif model == 'unsupervised':
        unsupSplits.append(topicSplit)
  
  return bowSplits, tfidfSplits, spriteSplits, ldaSplits, unsupSplits, ySplits

if __name__ == '__main__':
  inputPath = sys.argv[1]
  assignFmt = sys.argv[2]
  yPath     = sys.argv[3]
  
  (bowSplits, tfidfSplits, spriteSplits,
   ldaSplits, unsupSplits, ySplits) = ldData(inputPath, assignFmt, yPath)
  
  for condition, splits, isTopic in [('BOW',    bowSplits, False),
                                     ('TF-IDF', tfidfSplits, False),
                                     ('SPRITE', spriteSplits, True),
                                     ('LDA',    ldaSplits, True),
                                     ('UNSUPERVISED', unsupSplits, True)]:
    if isTopic:
      mse, mae = evalTopics(splits, ySplits)
      print '%s -- MSE: %.3f, MAE: %.3f' % (condition, mse, mae)
    else:
      mses, maes = evalBow(splits, ySplits)
      print '%s -- MSE: %s, MAE: %s' % (condition, mses, maes)

