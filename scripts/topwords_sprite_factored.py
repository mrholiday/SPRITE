#!/usr/bin/python

'''
Get the top words given the output from training a SpriteFactoredModel.

Adrian Benton
2/4/2015
'''

import argparse

import os, re
from operator import itemgetter
from math import exp

#OMEGA_PATH_RE_BASE = '^%s\.(?P<factor>.+)\.(?P<view>\d+)\.omega$'
OMEGA_PATH_RE_BASE = '^%s\.(?P<factor>.+)\.omega$'
ALPHA_PATH_RE_BASE = '^%s\.(?P<factor>.+)\.alpha$'
DELTA_PATH_RE_BASE = '^%s\.(?P<factor>.+)\.delta$'
BETA_PATH_RE_BASE  = '^%s\.(?P<factor>.+)\.beta$'
BETAB_PATH_RE_BASE = '^%s\.(?P<factor>.+)\.betaB$'

NUM_TOPWORDS = 20 # How many words to print out.

def main(basename, numObserved, includePrior):
  '''
  Arguments
  basename:  Basename for training output files
  numObserved: Number of observed factors.
  includePrior: Includes pseudocounts from prior when  generating topics.
  
  Prints out top words per topics as well as learned parameters for each
  factor/topic.
  '''
  
  Cph = {} # Factor to # phi components
  Cth = {} # Factor to # theta components
  Z   = {} # View to # topics
  NumSamples = 0 # Number of kept samples
  NumViews   = 0 # Number of views
  FactorNames = []
  
  filename = '%s.assign' % basename
  
  f = open(filename, 'r')
  line = f.next()
  tokens = line.split('\t')
  tokens.pop(0) # Tweet ID
  
  for view in range(len(tokens) - numObserved):
    samples = tokens[numObserved + view].split(' ')[0].split(':')
    
    samples.pop(0) # The token
    
    Z[view]    = len(samples)
    NumSamples = sum([int(count) for count in samples])
    NumViews   = view + 1
  f.close()
  
  print 'Z = {%s}' % (', '.join([str(v) for k, v in sorted(Z.items())]))
  print 'Num Samples: %d' % (NumSamples)
  
  totalNumTokens = 0
  totalNumCounts = 0
  
  count = {}
  count = dict([(view, dict([(i, {}) for i in range(Z_view)]))
                 for view, Z_view in Z.items()])
  
  f = open(filename, 'r')
  for line in f:
    viewSamples = line.strip().split('\t')[numObserved+1:]
    
    for view, tokenSamples in enumerate(viewSamples):
      for tokenSample in tokenSamples.split():
        parts = tokenSample.split(":")
        samples = [int(s) for s in parts[-Z[view]:]]
        word = ":".join(parts[0:(len(parts) - Z[view])])
        if not word:
          continue
        
        totalNumTokens += 1
        for i in range(Z[view]):
          if word not in count[view][i]:
            count[view][i][word] = 0
          count[view][i][word] += samples[i] / float(NumSamples)
          totalNumCounts += samples[i]
  f.close()
  
  print 'Total # tokens: %d, # samples: %d\n' % (totalNumTokens,
                                                 totalNumCounts)
  
  base, out_dir = os.path.basename(basename), os.path.dirname(basename)
  
  ''' ======== Loading Omega ======== '''
  
  '''
  This was when omega was split across multiple views.
  Now it is collapsed.
  
  omegaRe    = re.compile(OMEGA_PATH_RE_BASE % (base))
  omegaPaths = os.listdir(out_dir)
  omegaPaths = [(omegaRe.match(p).group('factor'),
                 int(omegaRe.match(p).group('view')),
                 os.path.join(out_dir, p)) for p in omegaPaths
                if omegaRe.match(p)]
  
  FactorNames = sorted(list(set([fName for fName, v, p in omegaPaths])))
  
  # Factor name -> View -> Component -> Word -> Weight
  omega = {}
  for factorName, view, p in omegaPaths:
    if factorName not in omega:
      omega[factorName] = {}
    omega[factorName][view] = {}
  
  for factorName, view, omegaPath in omegaPaths:
    f = open(omegaPath, 'r')
    Cph[factorName] = len(f.next().split()) - 1
    f.close()
    
    for c in range(Cph[factorName]):
      omega[factorName][view][c] = {}
    
    f = open(omegaPath, 'r')
    for line in f:
      tokens = line.rstrip().split(' ')
      word   = tokens.pop(0)
      
      if not word:
        continue
      
      for c in range(Cph[factorName]):
        omega[factorName][view][c][word] = float(tokens[c])
    f.close()
  '''
  
  omegaRe    = re.compile(OMEGA_PATH_RE_BASE % (base))
  omegaPaths = os.listdir(out_dir)
  omegaPaths = [(omegaRe.match(p).group('factor'),
                 os.path.join(out_dir, p)) for p in omegaPaths
                if omegaRe.match(p)]
  
  FactorNames = sorted(list(set([fName for fName, p in omegaPaths])))
  
  # Factor name -> View -> Component -> Word -> Weight
  omega = {}
  for factorName, p in omegaPaths:
    omega[factorName] = {}
  
  for factorName, omegaPath in omegaPaths:
    f = open(omegaPath, 'r')
    Cph[factorName] = len(f.next().split()) - 1
    f.close()
    
    for c in range(Cph[factorName]):
      omega[factorName][c] = {}
    
    f = open(omegaPath, 'r')
    for line in f:
      tokens = line.rstrip().split(' ')
      word   = tokens.pop(0)
      
      if not word:
        continue
      
      for c in range(Cph[factorName]):
        omega[factorName][c][word] = float(tokens[c])
    f.close()
  
  ''' ======== Loading OmegaBias ======== '''
  
  # View -> Word -> Weight
  omegaBias = dict([(view, {}) for view in range(NumViews)])
  for view in range(NumViews):
    f = open('%s.v%d.omegabias' % (basename, view), 'r')
    for line in f:
      tokens = line.rstrip().split()
      if len(tokens) < 2:
        continue
      
      word = tokens[0]
      wt   = float(tokens[1])
      
      omegaBias[view][word] = wt
    f.close()
  
  ''' ======== Loading Beta ======== '''
  
  betaRe    = re.compile(BETA_PATH_RE_BASE % (base))
  betaPaths = os.listdir(out_dir)
  betaPaths = [(betaRe.match(p).group('factor'),
                 os.path.join(out_dir, p)) for p in betaPaths
                if betaRe.match(p)]
  
  # Factor name -> View -> Topic -> Component -> Weight
  beta = {}
  for factorName, betaPath in betaPaths:
    beta[factorName] = {}
    
    f = open(betaPath, 'r')
    for line in f:
      tokens = line.strip().split()
      viewComp = tokens.pop(0)
      view = int(viewComp.split('_')[0])
      component = int(viewComp.split('_')[1])
      
      if view not in beta[factorName]:
        beta[factorName][view] = {}
      
      for z, wt in enumerate(tokens):
        if z not in beta[factorName][view]:
          beta[factorName][view][z] = {}
        
        beta[factorName][view][z][component] = float(wt)
    f.close()
  
  ''' ======== Loading BetaB (Sparsity Term) ======== '''
  
  betaBRe    = re.compile(BETAB_PATH_RE_BASE % (base))
  betaBPaths = os.listdir(out_dir)
  betaBPaths = [(betaBRe.match(p).group('factor'),
                 os.path.join(out_dir, p)) for p in betaBPaths
                if betaBRe.match(p)]
  
  # Factor name -> View -> Topic -> Component -> Weight
  betaB = {}
  for factorName, betaBPath in betaBPaths:
    betaB[factorName] = {}
    
    f = open(betaBPath, 'r')
    for line in f:
      tokens = line.strip().split()
      viewComp = tokens.pop(0)
      view = int(viewComp.split('_')[0])
      component = int(viewComp.split('_')[1])
      
      if view not in betaB[factorName]:
        betaB[factorName][view] = {}
      
      for z, wt in enumerate(tokens):
        if z not in betaB[factorName][view]:
          betaB[factorName][view][z] = {}
        
        betaB[factorName][view][z][component] = float(wt)
    f.close()
  
  ''' ======== Loading Delta ======== '''
  
  deltaRe    = re.compile(DELTA_PATH_RE_BASE % (base))
  deltaPaths = os.listdir(out_dir)
  deltaPaths = [(deltaRe.match(p).group('factor'),
                 os.path.join(out_dir, p)) for p in deltaPaths
                if deltaRe.match(p)]
  
  delta = {}
  for factorName, deltaPath in deltaPaths:
    delta[factorName] = {}
    Cth[factorName]   = 1 # Track of number of components for this factor
    
    f = open(deltaPath, 'r')
    for line in f:
      tokens = line.strip().split()
      viewAndComp = tokens.pop(0)
      
      view = int(viewAndComp.split('_')[0])
      component = int(viewAndComp.split('_')[1])
      
      if view not in delta[factorName]:
        delta[factorName][view] = {}
      
      if component >= Cth[factorName]:
        Cth[factorName] = component + 1
      
      if component not in delta[factorName][view]:
        delta[factorName][view][component] = {}
      
      for z, wt in enumerate(tokens):
        delta[factorName][view][component][z] = float(wt)
    
    f.close()
  
  for factorName in FactorNames:
    print '============ Factor "%s" ============\n' % (factorName)
    
    #if factorName in polarFactors:
    if len(omega[factorName]) == 1: # If we have only one component, then print out top and bottom words
      print '-'*12, 'Omega Positive', '-'*12
      
      words = sorted(omega[factorName][0].items(),
                     key=itemgetter(1),
                     reverse=True)
      for word, v in words[:NUM_TOPWORDS]:
        print word, v
      print '\n'
      
      print '-'*12, 'Omega Negative', '-'*12
      words = sorted(words, key=itemgetter(1), reverse=False)
      for word, v in words[:NUM_TOPWORDS]:
        print word, -1.0*v
      
      print '\n'
    else:
      for c in range(Cph[factorName]):
        print '-'*12, 'Omega', c, '-'*12
      
        words = sorted(omega[factorName][c].items(),
                       key=itemgetter(1), reverse=True)
        
        for word, v in words[:NUM_TOPWORDS]:
          print word
        
        print '\n'
    
    for view in sorted(delta[factorName].keys()):
      print '+'*12, 'View', view, '+'*12, '\n'
      
      '''
      if factorName in polarFactors:
        print '-'*12, 'Omega Positive', '-'*12
        
        words = sorted(omega[factorName][view][0].items(),
                       key=itemgetter(1),
                       reverse=True)
        for word, v in words[:NUM_TOPWORDS]:
          print word, v
        print '\n'
        
        print '-'*12, 'Omega Negative', '-'*12
        words = sorted(words, key=itemgetter(1), reverse=False)
        for word, v in words[:NUM_TOPWORDS]:
          print word, -1.0*v
        print '\n'
      else:
        for c in range(Cph[factorName]):
          print '-'*12, 'Omega', c, '-'*12
          
          words = sorted(omega[factorName][view][c].items(),
                         key=itemgetter(1), reverse=True)
          
          for word, v in words[:NUM_TOPWORDS]:
            print word
          
          print '\n'
      '''
      
      for c in range(Cth[factorName]):
        print '-'*12, 'Delta', c, '-'*12
        
        for z in range(Z[view]):
          print '%d: %0.3f*%0.3f = %f' % (z, betaB[factorName][view][z][c],
                                          delta[factorName][view][c][z],
                                          betaB[factorName][view][z][c]*
                                          delta[factorName][view][c][z])
        print '\n'
      
    print '\n'
  
  # Adjust counts based on prior
  if includePrior:
    for view in range(NumViews):
      for z in range(Z[view]):
        for w in count[view][z]:
          prior = 0.
          
          for factorName in FactorNames:
            for c in range(Cph[factorName]):
              prior += betaB[factorName][view][z][c] * beta[factorName][view][z][c] * omega[factorName][c][w]
          prior += omegaBias[view][w]
          prior = exp(prior)
          
          count[view][z][w] += prior
  
  print '='*12, 'TOPICS', '='*12, '\n'
  for view in range(NumViews):
    print '-'*12, 'View', view, '-'*12, '\n'
    
    for z in range(Z[view]):
      print 'Topic %d' % z
      
      if FactorNames: print 'Beta:'
      for factorName in FactorNames:
        for c in range(Cph[factorName]):
          print '%s %d: %0.3f*%0.3f = %f' % (factorName, c,
                                          betaB[factorName][view][z][c],
                                          beta[factorName][view][z][c],
                                          betaB[factorName][view][z][c]*
                                          beta[factorName][view][z][c])
      
      print '\n'
      
      w = 0
      words = sorted(count[view][z].items(),
                     key=itemgetter(1),
                     reverse=True)
      for word, v in words[:NUM_TOPWORDS]:
        print '%s: %0.3f' % (word, v)
      print "\n"

if __name__ ==  "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('basename', help='points to the .assign file, without the .assign extension')
  parser.add_argument('--numscores', help='how many supervised scores our data has', type=int, default=0)
  parser.add_argument('--includeprior', help='include pseudocounts from prior when generating topics', action='store_true', default=False)
  parser.add_argument('--numtopwords', help='number of top words to print out', type=int, default=20)
  
  args = parser.parse_args()
  
  numScores = args.numscores
  baseName = args.basename
  includePrior = args.includeprior
  NUM_TOPWORDS = args.numtopwords
  
  main(baseName, numScores, includePrior)
