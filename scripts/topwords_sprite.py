
#!/usr/bin/python

import sys
from operator import itemgetter
from math import exp

def main(basename, numScores):
	filename = '%s.assign' % basename
        
        f = open(filename, 'r')
	line = f.next()
        tokens = line.split()
        tokens.pop(0) # Tweet ID
        
        nums = tokens[numScores].split(':')
        
        nums.pop(0)
        
        Z = len(nums)
        N = 0
        for c in nums:
          N += int(c)
	f.close()
        
        print 'Z = %d, N = %d' % (Z, N)
        
	numtokens = 0
	numcounts = 0
        
	count = {}
	for i in range(0, Z): count[i] = {}
        
        f = open(filename, 'r')
	for line in f:
		tokens = line.split()
		tokens.pop(0) # Tweet ID
                
                if len(tokens) < (1+numScores): continue
                
                for i in range(numScores):
                  id = tokens.pop(0)
                
		for token in tokens:
			parts = token.split(":")
			z = parts[-Z:]
			word = ":".join(parts[0:len(parts)-Z])
                        
			numtokens += 1
			for i in range(Z):
				if word not in count[i]:
					count[i][word] = 0
				count[i][word] += int(z[i]) / float(N)
				numcounts += int(z[i])
        
	print numtokens
	print numcounts
        
	omega = {}
	for line in open(basename+'.omega', 'r'):
		Cph = len(line.split()) - 1
		break
        for c in range(Cph): omega[c] = {}
        
	for line in open(basename+'.omega', 'r'):
		tokens = line.strip().split()
		word = tokens.pop(0)
		for c in range(Cph):
			omega[c][word] = float(tokens[c])
        
	omegaBias = {}
	for line in open(basename+'.omegaBias', 'r'):
		tokens = line.strip().split()
		word = tokens[0]
		omegaBias[word] = float(tokens[1])
                
		#for c in range(1, Cph):
		#	omega[c][word] += omegaBias[word]
        
	beta = {}
	for z in range(Z): beta[z] = {}
	z = 0
	for line in open(basename+'.beta', 'r'):
		tokens = line.strip().split()
		for c in range(Cph):
			beta[z][c] = float(tokens[c])
		z += 1
        
	betaB = {}
	for z in range(Z): betaB[z] = {}
	z = 0
	for line in open(basename+'.betaB', 'r'):
		tokens = line.strip().split()
		for c in range(Cph):
			betaB[z][c] = float(tokens[c])
		z += 1
        
	for line in open(basename+'.delta', 'r'):
		Cth = len(line.split()) - 1
		break
	delta = {}
	for c in range(Cth): delta[c] = {}
	z = 0
	for line in open(basename+'.delta', 'r'):
		tokens = line.strip().split()
		tokens.pop(0)
		for c in range(Cth):
			delta[c][z] = float(tokens[c])
		z += 1
        
	for c in range(Cth):
		print 'Delta %d' % c
		for z in range(Z):
			print '%d: %0.3f*%0.3f = %f' % (z, betaB[z][c],
                                                        delta[c][z],
                                                        betaB[z][c]*
                                                        delta[c][z])
		print '\n'
	print '\n'
        
        for scoreFactor in range(numScores):
          omegaPos = {}
          
          if scoreFactor not in omega:
            continue
          
          for word in omega[scoreFactor]:
            omegaPos[word] = 1.0*omega[scoreFactor][word] #+ omegaBias[word]
          print 'Omega_%d Pos' % (scoreFactor)
          w = 0
          words = sorted(omegaPos.items(), key=itemgetter(1), reverse=True)
          for word, v in words:
            print word, v
            w += 1
            if w >= 25: break
          print '\n'
          omegaNeg = {}
          for word in omega[scoreFactor]:
            omegaNeg[word] = -1.0*omega[scoreFactor][word] #+ omegaBias[word]
          print 'Omega_%d Neg' % (scoreFactor)
          w = 0
          words = sorted(omegaNeg.items(), key=itemgetter(1), reverse=True)
          topwords = []
          for word, v in words:
            topwords.append('%s %f' % (word, v))
            w += 1
            if w >= 25: break
          print '\n'.join(reversed(topwords))
          print '\n'
        
	for c in range(numScores, Cph):
		print 'Omega %d' % c
                
		w = 0
		words = sorted(omega[c].items(), key=itemgetter(1), reverse=True)
		for word, v in words:
			print word#, v
			w += 1
			if w >= 25: break
		print "\n"
        
	# adjust counts based on prior
	for z in range(Z):
		for w in count[z]:
			prior = 0.
                        
                        if c in omega:
                          prior += beta[z][c] * omega[c][w]
                          for c in range(1, Cph):
                            prior += betaB[z][c] * beta[z][c] * omega[c][w]
			  prior += omegaBias[w]
			  prior = exp(prior)
			  #count[z][w] += prior
        
	print ''
	for z in range(Z):
		print 'Topic %d' % z
                
		print ''
		print 'Beta:'
		for c in range(Cph):
			print '%d: %0.3f*%0.3f = %f' % (c, betaB[z][c], beta[z][c], betaB[z][c]*beta[z][c])
		print ''
                
		w = 0
		words = sorted(count[z].items(), key=itemgetter(1), reverse=True)
		for word, v in words:
			print word#, v
			w += 1
			if w >= 25: break
		print "\n"

if __name__ ==  "__main__":
  import sys
  numScores = 3
  #numScores = 1
  
  baseName = sys.argv[1]
  if len(sys.argv) > 2:
    numScores = int(sys.argv[2])
  
  main(baseName, numScores)
