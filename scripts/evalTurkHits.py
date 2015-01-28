from csv import DictReader

import os

IN_PATH = '/home/adrianb/Desktop/Batch_1794840_batch_results_abApproved.csv'

def getAcc():
  correctPerCondition = {}
  
  f = open(IN_PATH)
  reader= DictReader(f)
  for row in reader:
    gold   = row['Input.correct_answer']
    answer = row['Answer.answer']
    data   = row['Input.issue'].lower()
    model  = row['Input.model'].lower()
    hitId  = row['HITId']
    is_rejected = row['AssignmentStatus'] == 'Rejected'
    is_correct  = 1 if (((gold == '0') and (answer == 'a')) or
                        ((gold == '1') and (answer == 'b')) or
                        ((gold == '2') and (answer == 'c'))) else 0
    
    if is_rejected:
      continue
    
    key = (data, model)
    if key not in correctPerCondition:
      correctPerCondition[key] = {}
    if hitId not in correctPerCondition[key]:
      correctPerCondition[key][hitId] = []
    
    correctPerCondition[key][hitId].append(is_correct)
  f.close()
  
  accOverall  = {}
  nOverall    = {}
  accMajority = {}
  
  
  for key in correctPerCondition:
    accOverall[key] = 0.0
    accMajority[key] = 0.0
    nOverall  = 0.0
    nMajority = 0.0
    
    for hitId, correctVals in correctPerCondition[key].items():
      nMajority += 1.0
      nOverall  += len(correctVals)
      
      accOverall[key]  += sum(correctVals)
      
      majorityVote = 1 if sum(correctVals) >= len(correctVals)/2.0 else 0
      accMajority[key] += majorityVote
    
    accOverall[key]  = (accOverall[key]/nOverall, accOverall[key], nOverall)
    accMajority[key] = (accMajority[key]/nMajority,
                        accMajority[key], nMajority)
  
  datas  = ['guns', 'smoking', 'vaccination']
  models = ['lda0', 'lda1', 'sprite', 'unsupervised', 'random']
  
  print '---- Overall Accuracy ----'
  for m in models:
    print ''
    for d in datas:
      print '%s\t%s\t%s' % (m, d, '%.3f\t%d\t%d' % accOverall[(d, m)])
  print ''
  
  print '---- Majority Accuracy ----'
  for m in models:
    print ''
    for d in datas:
      print '%s\t%s\t%s' % (m, d, '%.3f\t%d\t%d' % accMajority[(d, m)])

if __name__ == '__main__':
  getAcc()
