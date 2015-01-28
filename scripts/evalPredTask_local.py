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

from sklearn.linear_model import Ridge, LinearRegression
import numpy as np

PATH_FOLD_RE = re.compile('\.pred\.(?P<fold>\d+)\.')

def ldWords(textPath):
  wToI = {}
  iToW = {}
  idToBow = {}
  currIdx = 0
  
  f = open(textPath)
  for ln in f:
    flds = ln.strip().split()
    id = int(flds[0])
    idToBow[id] = {}
    for w in flds[1:]:
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
  N_w = dict([(i, 0.) for i in iToW])
  D = float(len(bow))
  idf = dict([(i, D) for i in iToW])
  
  for id, wMap in bow.items():
    for w in wMap:
      N_w[w] += 1.0
  for i in idf:
    idf[i] = log(idf[i]/N_w[i], 2.0)
  
  for id, wMap in bow.items():
    tfidfBow[id] = dict([v*idf for w, v in wMap])
  
  return tfidfBow

def getTopicDistributions(baseName):
  '''
  For each tweet in the output file, this guesses the topic distribution
  directly based on the sampled counts, which are then used as features in a
  model.
  '''
  
  fname = baseName + '.assign'
  
  idToTopicDist = {}
  f = open(fname)
  for line in f:
    flds = line.strip().split()
    tid = float(flds[0])
    topicDist = []
    N = 0
    
    assignFlds = flds[4:]
    for wordAssn in assignFlds:
      counts = [int(v) for v in wordAssn.split(':')[1:]]
      if not topicDist:
        topicDist = [0.]*len(counts)
      
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
  trainDevFeatures = {'train':[], 'dev':[]}
  
  for id, features in idToFeatures.items():
    fold = idToFold[id]
    if fold != devFold:
      foldToFeatures['train'].append(features)
    else:
      foldToFeatures['dev'].append(features)
  
  return foldToFeatures

def bowToNumpy(trainDevSplit):
  maxIdx = 0
  for bowMap in (trainDevSplit['train'] + trainDevSplit['dev']):
    for featIdx, featValue in bowMap.items():
      if featIdx > maxIdx:
        maxIdx = featIdx
  
  trainMatrix = np.zeroes((len(trainDevSplit['train']), maxIdx+1))
  devMatrix   = np.zeroes((len(trainDevSplit['dev']), maxIdx+1))
  
  for rowIdx, bowMap in enumerate(trainDevSplit['train']):
    for colIdx, featValue in bowMap.items():
      trainMatrix[rowIdx,colIdx] = featValue
  for rowIdx, bowMap in enumerate(trainDevSplit['dev']):
    for colIdx, featValue in bowMap.items():
      devMatrix[rowIdx,colIdx] = featValue
  npTrainDevSplit = (trainMatrix, devMatrix)
  
  return npTrainDevSplit

def topicToNumpy(trainDevSplit):
  maxIdx = len(trainDevSplit['train'][0])
  
  trainMatrix = np.zeroes((len(trainDevSplit['train']), maxIdx+1))
  devMatrix   = np.zeroes((len(trainDevSplit['dev']), maxIdx+1))
  
  for rowIdx, row in enumerate(trainDevSplit['train']):
    for colIdx, featValue in enumerate(row):
      trainMatrix[rowIdx,colIdx] = featValue
  for rowIdx, row in enumerate(trainDevSplit['dev']):
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
      
      ridgeModel = Ridge(alpha=1.0/(2*c))
      ridgeModel.fit(trainFeatures, trainY)
      devY_hat   = ridgeModel.predict(devFeatures)
      
      mse = (devY - devY_hat)**2.0/devY.shape[0]
      mae = np.fabs(devY - devY_hat)/devY.shape[0]
      runningMSE += mse
      runningMAE += mae
    
    avgMSE[cIdx] = runningMSE/len(yPerFold)
    avgMAE[cIdx] = runningMAE/len(yPerFold)
  
  return avgMSE, avgMAE

def evalTopics(topicFeatureFolds=[], yPerFold=[]):
  avgMSE = 0.0
  avgMAE = 0.0
  
  runningMSE = 0.0
  runningMAE = 0.0
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
    runningMSE += mse
    runningMAE += mae
  
  avgMSE = runningMSE/len(yPerFold)
  avgMAE = runningMAE/len(yPerFold)
  
  return avgMSE, avgMAE

def ldData(baseNameFmt, textPath, yPath, foldPath):
  '''
  Loads feature vectors and dependent variables for:
  
  - Bag of words counts
  - Bag of words TF-IDF
  - Topic distribution for no scores case
  - Topic distribution for all scores case 
  
  Each of these are a list of train/dev pairs (based on number of folds).
  '''
  
  idToFold = ldFolds(foldPath)
  
  itow, wtoi, bow = ldWords(textPath)
  print 'Loaded BOW'
  
  tfidfBow = toTfIdf(bow)
  print 'Converted to TF-IDF'
  
  for i in range(5):
    baseName = baseNameFmt % (i)
    idToTopicDist = getTopicDistributions(baseName)
    
    
  
  return cntBow, idfBow, 

if __name__ == '__main__':
  baseNameRe = sys.argv[1]
  textPath   = sys.argv[2]
  foldPath   = sys.argv[3]
  
  
