package models.factored;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.math.BigInteger;
import java.util.Map;


import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;

import utils.IO;
import utils.Log;
import utils.MathUtils;
import utils.Tup2;
import utils.Tup5;

/**
 * Should be equivalent to SpriteJoint model but allows for multiple
 * factors.  Factors may be observed, or latent this is an attempt to
 * make SpriteJoint more generic.  At the moment, not hopeful though.
 * 
 * @author adrianb
 *
 */
public class SpriteFactoredTopicModel extends ParallelTopicModel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7666805827230028611L;
	
	//public Map<String, Integer>[] wordMaps;
	//public Map<Integer,String>[] wordMapInvs;
    
	public Map<String, Integer> wordMap;
	public Map<Integer,String> wordMapInv;
    
	// All of the factors used across views.  Ordering should be the same as
	// the observed scores in the input files.
	protected Factor[] factors;
	
	protected int[] Z;   // Possibly different number of topics for each view.
	protected int D;
	protected int W;   
	//protected int[] W;   // Each view has its own alphabet.
	protected int numViews = 0; // Number of different views for the document
	
	// To index thread locks
	protected int[] runningZSums;
	protected int[] runningDSums;
	protected int[] runningWSums;
	
	// Current sampled topic assignments
	protected int[][][] docsZ;    // Burned-in samples of Document -> View -> Word Index 
	protected int[][][][] docsZZ; // Burned-in samples of Document -> View -> Word Index -> Topic
	
	// Samples
	protected int[][][] nDZ; // Document -> View -> Topic samples
	protected int[][]    nD; // Document -> View samples (number of tokens in each, set at initialization)
	protected int[][][] nZW; // View -> Topic -> Word samples
	protected int[][]    nZ; // View -> Topic samples
	
	protected int numFactorsObserved = 0;
	protected int[] observedFactorSizes;
	
	// NOTE: Removed this since I think it complicates things too much.  Just
	// assume a separate \widetilde{\theta} and \widetilde{\phi} for each view.
	//protected int[] priorToView; // Maps each theta/phi prior to the view it is responsible for.
	
	// Should be the same size, one per view
	protected SpriteThetaPrior[] thetaPriors;
	protected SpritePhiPrior[]   phiPriors;
	
	private double stepSize; // Master step size
	
	public String outputDir = null; // Where model parameters get written
	
	/**
	 * Gets the ranges over which we want to split our threads up for each
	 * view.  The length of \theta and \phi priors are assumed equal.
	 */
	private int[][] getDataRanges(SpriteThetaPrior[] tpriors, SpritePhiPrior[] ppriors) {
		int[][] ranges = new int[tpriors.length][3];
		for (int i = 0; i < tpriors.length; i++) {
			ranges[i][0] = tpriors[i].Z;
			ranges[i][1] = tpriors[i].D;
			ranges[i][2] = ppriors[i].W;
		}
		
		return ranges;
	}
	
	public SpriteFactoredTopicModel(SpriteThetaPrior[] thetaPriors0, SpritePhiPrior[] phiPriors0,
									Factor[] factors0, int numThreads0, double stepSize0) {
		super.setParallelParams(numThreads0, getDataRanges(thetaPriors0, phiPriors0));
		
		thetaPriors = thetaPriors0;
		phiPriors   = phiPriors0;
		
		factors     = factors0;
		
		numFactorsObserved = 0;
		for (Factor f : factors) {
			if (f.isObserved()) {
				numFactorsObserved++;
			}
		}
		
		// Initialize observed factor sizes (number of observed components) for loading documents
		observedFactorSizes = new int[numFactorsObserved];
		int factorIdx = 0;
		for (Factor f : factors) {
			if (f.isObserved()) {
				observedFactorSizes[factorIdx] = f.C;
				factorIdx++;
			}
		}
		
		// TODO: Number of theta/phi priors same as number of views.  May want to share a prior over multiple views *shrug*
		numViews = thetaPriors.length;
		
		/*
		for (int i = 0; i < priorToView.length; i++) {
			if ((priorToView[i]+1) > numViews+1) {
				numViews = priorToView[i] + 1;
			}
		}
		*/
		
		// Each view has its own set of topics.  May be joined by supertopics represented by factors.
		Z = new int[numViews];
		for (int i = 0; i < thetaPriors.length; i++) {
			Z[i] = thetaPriors[i].Z;
		}
		
		stepSize = stepSize0;
	}
	
	private Integer getLock(int v, int[] runningSum, int index) {
		if (v > 0) {
			return runningSum[v-1] + index;
		}
		else {
			return index;
		}
	}
	
	@Override
	public void initTrain() {
		docsZ = new int[D][numViews][];
		docsZZ = new int[D][numViews][][];
		
		// Initialize samples.  Since each view has its own set of topics, we need to sample for
		// each one separately.
		nDZ = new int[D][numViews][];
		nD  = new int[D][numViews];
		nZW = new int[numViews][][];
		nZ  = new int[numViews][];
		
		for (int v = 0; v < numViews; v++) {
			for (int d = 0; d < D; d++) {
				nDZ[d][v] = new int[Z[v]];
			}
			
			nZW[v] = new int[Z[v]][W];
			//nZW[v] = new int[Z[v]][W[v]];
			nZ[v]  = new int[Z[v]];
		}
		
		for (int d = 0; d < D; d++) { 
			for (int v = 0; v < numViews; v++) {
				int numDocTokens = docs[d][v].length;
				
				docsZ[d][v]  = new int[numDocTokens];
				docsZZ[d][v] = new int[numDocTokens][Z[v]];
				
				for (int n = 0; n < numDocTokens; n++) {
					int w = docs[d][v][n];
					
					int z = MathUtils.r.nextInt(Z[v]); // sample uniformly
					docsZ[d][v][n] = z;
					
					// update counts
					
					nZW[v][z][w] += 1;	
					nZ[v][z] += 1;
					nDZ[d][v][z] += 1;
					nD[d][v] += 1;
				}
			}
		}
		
		runningDSums = new int[numViews]; runningDSums[0] = D;
		runningZSums = new int[numViews]; runningZSums[0] = Z[0];
		runningWSums = new int[numViews]; runningWSums[0] = W;
		
		// Compute running sums of topics/words to initialize thread locks
		for (int v = 1; v < numViews; v++) {
			runningDSums[v] = D;
			runningZSums[v] = runningZSums[v-1] + Z[v];
			runningWSums[v] = W;
			//runningWSums[v] = runningWSums[v-1] + W[v];
		}
		
		wordLocks  = new Integer[runningWSums[numViews - 1]];
		topicLocks = new Integer[runningZSums[numViews - 1]];
		docLocks   = new Integer[runningDSums[numViews - 1]];
		
		// The min/max number of topics/documents/words for each view.  We partition threads accordingly.
		varDims = new int[numViews][3];
		for (int v = 0; v < numViews; v++) {
			varDims[v][0] = Z[v];
			varDims[v][1] = D;
			varDims[v][2] = W;
			//varDims[v][2] = W[v];
			
			//for (int w = 0; w < W[v]; w++) {
			for (int w = 0; w < W; w++) {
				int wLock = getLock(0, runningWSums, w);
				wordLocks[wLock] = (Integer)wLock;
			}
			for (int d = 0; d < D; d++) {
				int dLock = getLock(0, runningDSums, d);
				docLocks[dLock] = (Integer)dLock;
			}
			for (int z = 0; z < Z[v]; z++) {
				int zLock = getLock(v, runningZSums, z);
				topicLocks[zLock] = (Integer)zLock;
			}
		}
		
		super.initTrain(); // Spin up threads
	}
	
	@Override
	public void logIteration() {
		// Print out delta bias vectors
		for (SpriteThetaPrior prior : thetaPriors) {
			prior.logState();
		}
		// Print out omega bias vectors
		for (SpritePhiPrior prior : phiPriors) {
			prior.logState();
		}
		
		// Print out per-component weights
		for (Factor f : factors) {
			f.logState();
		}
	}
	
	@Override
	public void collectSamples() {
		// Only collect samples for the last couple hundred iterations
		
		if (burnedIn) {
			for (int v = 0; v < numViews; v++) {
				for (int d = 0; d < D; d++) {
					for (int n = 0; n < docs[d][v].length; n++) { 
						int x = docsZ[d][v][n];
						docsZZ[d][v][n][x] += 1; // Cache the sampled topic for this single word
					}
				}
			}
		}
	}
	
	@Override
	public double computeLL(int[][][] corpus) {
		// TODO: For now assumes that we've already aggregated samples for
		// this corpus.  Should probably pull out a separate function that
		// will take samples and then compute log-likelihood.  Ultimately
		// I should remove global references to the corpus and sample counts
		// and pass these in between functions.
		
		double LL = 0;
		
		for (int d = 0; d < D; d++) {
			for (int v = 0; v < numViews; v++) {
				int numDocTokens = corpus[d][v].length;
				
				for (int n = 0; n < numDocTokens; n++) { 
					int w = corpus[d][v][n];
					
					double tokenLL = 0;
					
					// marginalize over z
					for (int z = 0; z < Z[v]; z++) {
						//tokenLL += (nDZ[d][v][z] + priorDZ[d][v][z]) / (nD[d][v] + thetaNormPerView[v][d])*
						//		(nZW[v][z][w] + priorZW[v][z][w]) / (nZ[v][z] + phiNormPerView[v][z]);
						
						tokenLL += (nDZ[d][v][z] + thetaPriors[v].thetaTilde[d][z]) / (nD[d][v] + thetaPriors[v].thetaNorm[d])*
								(nZW[v][z][w] + phiPriors[v].phiTilde[z][w]) / (nZ[v][z] + phiPriors[v].phiNorm[z]);
					}
					
					LL += Math.log(tokenLL);
				}
			}
		}
		
		return LL;
	}
	
	@Override
	public void updatePriors(Tup2<Integer, Integer>[][] parameterRanges) {
		// Only update the priors for topics minZ to maxZ for each view
		
		for (int v = 0; v < numViews; v++) {
			int minZ = parameterRanges[v][0]._1();
			int maxZ = parameterRanges[v][0]._2();
			
			int minD = parameterRanges[v][1]._1();
			int maxD = parameterRanges[v][1]._2();
			
			thetaPriors[v].updatePrior(minD, maxD); // We split computation of theta across documents
			phiPriors[v].updatePrior(minZ, maxZ);   // Split computation of phi across topics
		}
	}
	
	@Override
	public void sampleBatch(Tup2<Integer, Integer>[][] parameterRanges) {
		for (int v = 0; v < numViews; v++) {
			int minD = parameterRanges[v][1]._1();
			int maxD = parameterRanges[v][1]._2();
			
			for (int d = minD; d < maxD; d++) {
				for (int n = 0; n < docs[d][v].length; n++) {
					sample(d, v, n);
				}
				
				if (d % 10000 == 0) {
					Log.info("SpriteFactoredTopicModel", String.format("Done view %d, document %d", v, d));
				}
			}
		}
	}
	
	private void sample(int d, int v, int n) {
		int w = docs[d][v][n];
		int topic = docsZ[d][v][n];
		
		// decrement counts
		
		synchronized(topicLocks[getLock(v, runningZSums, topic)]) {
			nZW[v][topic][w] -= 1;
			nZ[v][topic] -= 1;
			nDZ[d][v][topic] -= 1;
		}
		
		// sample new topic value 
		
		double[] p = new double[Z[v]];
		double pTotal = 0;
		
		for (int z = 0; z < Z[v]; z++) {
			//p[z] = (nDZ[d][z] + priorDZ[d][v][z]) *
			//		(nZW[z][w] + priorZW[v][z][w]) / (nZ[z] + phiNorm[v][z]);
			
			p[z] = (nDZ[d][v][z] + thetaPriors[v].thetaTilde[d][z]) *
					(nZW[v][z][w] + phiPriors[v].phiTilde[z][w]) / (nZ[v][z] + phiPriors[v].phiNorm[z]);
			
			pTotal += p[z];
		}
		
		double u = MathUtils.r.nextDouble() * pTotal;
		
		double probVal = 0.0;
		for (int z = 0; z < Z[v]; z++) {
			probVal += p[z];
			
			if (probVal > u) {
				topic = z;
				break;
			}
		}
		
		// increment counts
		
		synchronized(topicLocks[getLock(v, runningZSums, topic)]) {
			nZW[v][topic][w] += 1;	
			nZ[v][topic] += 1;
			nDZ[d][v][topic] += 1;
		}
		
		// set new assignments
		docsZ[d][v][n] = topic;
	}
	
	@Override
	public void updateGradient(Tup2<Integer, Integer>[][] parameterRanges) {
		// Compute gradients
		
		for (int v = 0; v < numViews; v++) {
			int minZ = parameterRanges[v][0]._1();
			int maxZ = parameterRanges[v][0]._2();
			
			for (int z = minZ; z < maxZ; z++) {
				//for (int w = 0; w < W[v]; w++) {
				for (int w = 0; w < W; w++) {
					phiPriors[v].updateGradient(z, w, nZ[v][z], nZW[v][z][w], getLock(v, this.runningWSums, w));
				}
			}
		}
		
		for (int v = 0; v < numViews; v++) {
			int minZ = parameterRanges[v][0]._1();
			int maxZ = parameterRanges[v][0]._2();
			for (int z = minZ; z < maxZ; z++) {
				for (int d = 0; d < D; d++) {
					int docCount = nD[d][v];
					int docTopicCount = nDZ[d][v][z];
					thetaPriors[v].updateGradient(z, d, docCount, docTopicCount, getLock(v, runningDSums, d));
				}
			}
		}
	}
	
	@Override
	public void doGradientStep(Tup2<Integer, Integer>[][] parameterRanges) {
		for (Factor f: factors) {
			for (int v : f.revViewIndices.keySet()) {
				int minZ = parameterRanges[v][0]._1();
				int maxZ = parameterRanges[v][0]._2();
				int minD = parameterRanges[v][1]._1();
				int maxD = parameterRanges[v][1]._2();
				int minW = parameterRanges[v][2]._1();
				int maxW = parameterRanges[v][2]._2();
				
				f.doGradientStep(v, minZ, maxZ, minD, maxD, minW, maxW, stepSize);
			}
		}
		
		for (int v = 0; v < numViews; v++) {
			int minZ = parameterRanges[v][0]._1();
			int maxZ = parameterRanges[v][0]._2();
			int minW = parameterRanges[v][2]._1();
			int maxW = parameterRanges[v][2]._2();
			
			thetaPriors[v].doGradientStep(minZ, maxZ, stepSize); // Update delta bias
			phiPriors[v].doGradientStep(minW, maxW, stepSize); // Update omega bias
		}
	}
	
	@Override
	public void readDocs(String filename) throws Exception {
		Tup5<BigInteger[], Map<String, Integer>, Map<Integer, String>,
		     double[][][], int[][][]> loadedValues = IO.readTrainInput(filename, observedFactorSizes, numViews);
		
		docIds     = loadedValues._1();
		wordMap    = loadedValues._2();
		wordMapInv = loadedValues._3();
		double[][][] observedValues = loadedValues._4(); // Factor -> Doc -> Component
		docs = loadedValues._5();
		
		D = docs.length;
		W = wordMap.size();
//		W = new int[numViews];
//		for (int v = 0; v < numViews; v++) {
//			W[v] = wordMaps[v].size();
//		}
		
		// Assign component weights to observed factors
		int factorIdx = 0;
		for (Factor f : factors) {
//			int[] W_subset = new int[f.viewIndices.length];
//			for (int i = 0; i < f.viewIndices.length; i++) {
//				W_subset[i] = W[f.viewIndices[i]];
//			}
			
			// Initialize factors -- observed and latent.
			if (f.isObserved()) {
				f.initialize(observedValues[factorIdx], W, D);
				//f.initialize(observedValues[factorIdx], W_subset, D);
				factorIdx++;
			}
			else {
				f.initialize(W, D);
				//f.initialize(W_subset, D);
			}
		}
		
		for (SpriteThetaPrior p : thetaPriors) {
			p.initialize(D);
		}
		
		for (int v = 0; v < numViews; v++) {
			phiPriors[v].initialize(W);
//			phiPriors[v].initialize(W[v]);
		}
		
	}
	
	public void writeOutput(String filename) throws Exception {
		if (outputDir != null) {
			writeOutput(filename, outputDir);
		}
		else {
			writeOutput(filename, new File(filename).getParent()); // Write to the same directory as the input file.
		}
	}
	
	@Override
	public void writeOutput(String filename, String outputDir) throws Exception {
		File outDirFile = new File(outputDir);
		outDirFile.mkdir();
		
		String baseName = new File(filename).getName();
		
		//Write topic assignments file
		
		FileWriter fw = new FileWriter(new File(outputDir, baseName + ".assign"));
		BufferedWriter bw = new BufferedWriter(fw);
		
		for (int d = 0; d < D; d++) {
			bw.write(docIds[d].toString());
			for (Factor f: factors) {
				if (f.isObserved()) {
					bw.write("\t" + f.getAlphaString(d));
				}
			}
			
			for (int v = 0; v < numViews; v++) {
				bw.write("\t");
				for (int n = 0; n < docs[d][v].length; n++) {
					String word = wordMapInv.get(docs[d][v][n]);
//					String word = this.wordMapInvs[v].get(docs[d][v][n]);
					
					bw.write(word);  // for multiple samples
					for (int zz = 0; zz < Z[v]; zz++) {
						bw.write(":" + docsZZ[d][v][n][zz]);
					}
					
					if (n < (docs[d][v].length - 1))
						bw.write(" ");
				}
			}
			
			bw.newLine();
		}
		
		bw.close();
		fw.close();
		
		// Write omega/delta bias terms
		for (int v = 0; v < numViews; v++) {
			fw = new FileWriter(new File(outputDir, String.format("%s.v%d.deltabias", baseName, v)));
			bw = new BufferedWriter(fw);
			
			SpriteThetaPrior tprior = thetaPriors[v];
			tprior.writeDeltaBias(bw);
			
			bw.close();
			fw.close();
			
			fw = new FileWriter(new File(outputDir, String.format("%s.v%d.omegabias", baseName, v)));
			bw = new BufferedWriter(fw);
			
			SpritePhiPrior pprior = phiPriors[v];
			pprior.writeOmegaBias(bw, wordMapInv);
//			pprior.writeOmegaBias(bw, wordMapInvs[v]);
			
			bw.close();
			fw.close();
		}
		
		// Write factor parameters
		
		for (Factor f : factors) {
			fw = new FileWriter(new File(outputDir, String.format("%s.%s.beta", baseName, f.factorName)));
			bw = new BufferedWriter(fw);
			
			f.writeBeta(bw, false);
			
			bw.close();
			fw.close();
			
			fw = new FileWriter(new File(outputDir, String.format("%s.%s.betaB", baseName, f.factorName)));
			bw = new BufferedWriter(fw);
			
			f.writeBeta(bw, true);
			
			bw.close();
			fw.close();
			
			fw = new FileWriter(new File(outputDir, String.format("%s.%s.omega", baseName, f.factorName)));
			bw = new BufferedWriter(fw);
			
			f.writeOmega(bw, wordMapInv);
			
			bw.close();
			fw.close();
			
//			for (int v : f.revViewIndices.keySet()) {
//				fw = new FileWriter(new File(outputDir, String.format("%s.%s.%d.omega", baseName, f.factorName, v)));
//				bw = new BufferedWriter(fw);
//				
//				f.writeOmega(bw, v, wordMapInvs[f.revViewIndices.get(v)]);
//				
//				bw.close();
//				fw.close();
//			}
			
			fw = new FileWriter(new File(outputDir, String.format("%s.%s.alpha", baseName, f.factorName)));
			bw = new BufferedWriter(fw);
			
			f.writeAlpha(bw);
			
			bw.close();
			fw.close();
			
			fw = new FileWriter(new File(outputDir, String.format("%s.%s.delta", baseName, f.factorName)));
			bw = new BufferedWriter(fw);
			
			f.writeDelta(bw);
			
			bw.close();
			fw.close();
		}
		
		// Serialize the model so we can use it/continue training later if necessary
		try {
			FileOutputStream fileOut = new FileOutputStream(new File(outputDir, baseName + ".ser"));
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(this);
			out.close();
			fileOut.close();
		} catch(IOException e) {
		   	e.printStackTrace();
		}
	}

	@Override
	public void updateGradientTest(Tup2<Integer, Integer>[][] parameterRanges) {
		// TODO Auto-generated method stub
		
	}

}
