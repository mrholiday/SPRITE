import glob, os, sys

ds = sys.argv[1:]
cmds = []

for inDir in ds:
  if not os.path.isdir(inDir): 
    print 'Not directory:', inDir
  
  walk  = os.walk(inDir)
  
  for baseDir, ds, ps in walk:
    for p in ps:
      if p.endswith('assign'):
        basePath = os.path.join(baseDir, p.replace('.assign', ''))
        
        # Guess how many values were observed in the data
        f = open(os.path.join(baseDir, p))
        vs = f.next().strip().split('\t')
        numObserved = 0
        for fld in vs[1:]:
          try:
            for v in fld.split(' '):
              value = float(v)
            numObserved += 1
          except:
            break
        f.close()
        
        topicPath = os.path.join(baseDir, p.replace('.assign', '.topics'))
        
        cmds.append('python topwords_sprite_factored.py %s %d | tee %s' % (basePath, numObserved, topicPath))

for i, cmd in enumerate(cmds):
  print i, len(cmds), cmd
  os.system(cmd)
