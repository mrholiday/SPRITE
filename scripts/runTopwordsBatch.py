import sys, time
import os

#topDir = '../../data/gunControl_oversampleBiased100K50Percent_withState_withSent/'
topDir = sys.argv[1]
dataSetting = sys.argv[2]

ds = [os.path.join(topDir, d) for d in os.listdir(topDir)
      if os.path.isdir(os.path.join(topDir, d))]
#settings = ['all', 'noscores']
#settings = ['noscores', 'stanceAndSent', 'ownershipAndSent', 'onlyownership', 'onlystance']

bFmt = 'input.%s.all.onlyfreq.1.txt%s'
aFmt = 'input.%s.all.onlyfreq.1.txt%s.assign'
tFmt = '%s-%s-%s.topwords'

idx = 0

settings = ['', '.lda', '.unsupervised']

for d, setting in [(d, setting) for d in ds for setting in settings]:
  basePath    = os.path.join(d, bFmt % (dataSetting, setting))
  assignPath  = os.path.join(d, aFmt % (dataSetting, setting))
  topWordPath = tFmt % (os.path.basename(d).replace('/', ''),
                        dataSetting, setting)
  if not os.path.exists(assignPath):
    print basePath + " not finished... skipping"
    continue
  
  print 'python topwords_sprite.py %s %d > %s &' % (basePath, 3, topWordPath)
  dummy = os.system('python topwords_sprite.py %s %d > %s &' % (basePath,
                                                                3,
                                                              topWordPath))
  #print d, setting
  idx += 1
  if not idx % 20:
    time.sleep(210)

time.sleep(210)
