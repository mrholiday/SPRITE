'''
Template for submitting a batch of training SPRITE grid jobs.

Adrian Benton
2/5/2014
'''

import os

variants = ['Directly, Linked Topics across Views',
            'One supertopic, many children',
            'Many supertopics, many children',
            'LDA single view, one supertopic with many child',
            'LDA single view, many supertopics with many children']
models = ['Sprite2ViewLinkedTopics', 'Sprite', '']
inPaths = ['input.test1.txt', 'input.test2.txt']
outDirs = ['/path/to/test1/output/dir', '/path/to/test2/output/dir']

Zs = [10, 20, 40]
SigAlphas = [0.5, 1.0, 2.0]
Steps = [0.01, 0.02]

for model in models:
  for inPath, outDir in zip(inPaths, outDir):
    for Z in Zs:
      for sigmaAlpha in SigAlphas:
        for step in Steps:
          os.system(
            'qsub -q himem.q runSprite.sh %s %s %s %d %f %f' % (model,
                                                                inPath,
                                                                outDir,
                                                                Z,
                                                                sigmaAlpha,
                                                                step)
                   )
