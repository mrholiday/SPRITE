'''
Gradient descent for regularized linear regression implemented in theano.

Adrian Benton
1/21/2015
'''

import theano
import theano.tensor as T
from theano import shared

import numpy as np

EPS = 1.0e-8

class LinReg:
  def __init__(self, F):
    self.F = F
    
    self.randInit()
  
  def randInit(self):
    self.W = shared(np.zeros((self.F, 1)))
    self.B = shared(0.0)
    self.adaW = shared(np.zeros((self.F, 1)))
    self.adaB = shared(0.0)
    self.step = shared(1.0)
    
    self.buildFns()
  
  def buildFns(self):
    self.X = shared(np.zeros((1, self.F)))
    self.Y = shared(np.zeros((1, self.F)))
    self.C = shared(0.0)
    
    Y_hat = T.dot(self.X, self.W) + self.B
    l2Reg = self.C * T.sum(T.pow(self.X, 2))
    
    mseCost   = T.sum(T.pow(Y_hat - self.Y, 2))
    cost      = mseCost + l2Reg
    costGradW = T.grad(cost, self.W)
    costGradB = T.grad(cost, self.B)
    
    maeCost = T.sum(abs(Y_hat - self.Y))
    
    self.costFn = theano.function([], cost)
    print 'Built cost function... ',
    
    self.mseCostFn = theano.function([], mseCost)
    self.maeCostFn = theano.function([], maeCost)
    print 'mae/mse cost... ',
    
    self.costGradWFn = theano.function([], costGradW)
    print 'gradient W... ',
    
    self.costGradBFn = theano.function([], costGradB)
    print 'gradient B... ',
    
    self.getCostAndUpdateFn = theano.function([], cost,
      updates=[(self.W, self.W -
                      (self.step/T.sqrt(self.adaW + EPS) * costGradW)),
               (self.B, self.B -
                      (self.step/T.sqrt(self.adaB + EPS) * costGradB))])
    
    self.updateAdaFn = theano.function([], self.adaB,
      updates=[(self.adaW, self.adaW + T.pow(costGradW, 2.0)),
               (self.adaB, self.adaB + T.pow(costGradB, 2.0))])
    print 'updates... ',
    
    #self.getCostAndUpdateFn = theano.function([], cost,
    #  updates=[(self.adaW, self.adaW + T.pow(costGradW, 2.0)),
    #           (self.adaB, self.adaB + T.pow(costGradB, 2.0)),
    #           (self.W, self.W -
    #                  (self.step/T.sqrt(T.sqrt(self.adaW + EPS)) *
    #                   costGradW)),
    #           (self.B, self.B -
    #                  (self.step/T.sqrt(T.sqrt(self.adaB + EPS)) *
    #                   costGradB))])
    
    #self.getCostAndUpdateFn = theano.function([], cost,
    #  updates=[(self.W, self.W - (self.step * costGradW)),
    #           (self.B, self.B - (self.step * costGradB))])
    
    self.predFn = theano.function([], Y_hat)
    print 'prediction function'
  
  def train(self, X, Y, c, step=1.0, iters=1000, printIter=50):
    self.X.set_value(X)
    self.Y.set_value(Y)
    self.B.set_value(np.mean(Y))
    self.C.set_value(c)
    
    N = Y.shape[0]
    
    # AdaDelta-controlled step size
    self.adaW.set_value(np.zeros((self.F, 1)))
    self.adaB.set_value(0.0)
    self.step.set_value(step)
    
    oldCost = self.costFn()
    cost    = oldCost
    
    import time
    start = time.time()
    
    #print 'Cost:', cost
    
    #import pdb; pdb.set_trace()
    
    for i in range(iters):
      dummy = self.updateAdaFn()
      
      #print self.costGradBFn(), self.adaB.get_value()
      
      oldCost = cost
      cost    = self.getCostAndUpdateFn()
      
      if not (i % printIter):
        print 'Iter: %d, Reg. MSE: %e, Delta: %e, Time: %d' % (
                                                     i, cost/N,
                                                     cost/N - oldCost/N,
                                                     time.time() - start)
    
    #print 'Iter: %d, MSE: %.3f, Time: %d' % (i, cost/N,
    #                                          time.time() - start)
  
  def predict(self, X):
    self.X.set_value(X)
    return self.predFn()
  
  def maeCost(self, X, Y):
    self.X.set_value(X)
    self.Y.set_value(Y)
    return self.maeCostFn()
  
  def mseCost(self, X, Y):
    self.X.set_value(X)
    self.Y.set_value(Y)
    return self.mseCostFn()

def test(inputPath, assignFmt, yPath):
  from evalPredTask import ldData
  
  (bowSplits, tfidfSplits, spriteSplits,
   ldaSplits, unsupSplits, ySplits) = ldData(inputPath, assignFmt, yPath)
  
  for condition, splits, isTopic in [('BOW',    bowSplits, False),
                                     ('SPRITE', spriteSplits, True)]:
    trainX, devX = splits[0]
    trainY, devY = ySplits[0]
    trainN, devN = trainY.shape[0], devY.shape[0]
    
    lr = LinReg(trainX.shape[1])
    lr.train(lr.train(trainX, trainY, 0.0, 1.0, 1000, 1))
    trainMse = lr.mseCost(trainX, trainY)/trainN
    trainMae = lr.maeCost(trainX, trainY)/trainN
    devMse   = lr.mseCost(devX, devY)/devN
    devMae   = lr.maeCost(devX, devY)/devN
    
    print '(Train) MSE,MAE: %e %e | (Dev) MSE,MAE: %e %e' % (trainMse,
                                                             trainMae,
                                                             devMse,
                                                             devMae)

if __name__ == '__main__':
  import sys
  
  inputPath = sys.argv[1]
  assignFmt = sys.argv[2]
  yPath     = sys.argv[3]
  test(inputPath, assignFmt, yPath)
