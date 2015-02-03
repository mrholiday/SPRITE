package main;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Map;

import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;

import utils.IO;
import utils.MathUtils;
import utils.Tup2;
import utils.Tup4;

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
	public Map<String, Integer>[] wordMaps;
	public Map<Integer,String>[] wordMapInvs;
    
	// All of the factors used across views.  Ordering should be the same as
	// the observed scores in the input files.
	protected Factor[] factors;
	
	protected int[] Z;   // Possibly different number of topics for each view.
	protected int D;
	protected int[] W;   // Each view has its own alphabet.
	protected int numViews = 0; // Number of different views for the document
	
	protected int[][][] docs; // Loaded documents.  Document -> View -> Words
	
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
	
	// TODO: Removed this since I think it complicates things too much.  Just
	// assume a separate \widetilde{\theta} and \widetilde{\phi} for each view.
	//protected int[] priorToView; // Maps each theta/phi prior to the view it is responsible for.
	
	// Should be the same size, one per view
	protected SpriteThetaPrior[] thetaPriors;
	protected SpritePhiPrior[]   phiPriors;
	
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
									Factor[] factors0, int numThreads0) {
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
		
		// TODO: Number of theta/phi priors same as number of views
		numViews = thetaPriors.length;
		
		/*
		// Kind of annoying... maybe just pass numViews as a parameter
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
	}
	
	private int getLock(int v, int index) {
		return v*10000000 + index;
	}
	
	@Override
	public void initialize() {
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
			
			nZW[v] = new int[Z[v]][W[v]];
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
		
		// The min/max number of topics/documents/words for each view.  We partition threads accordingly.
		varDims = new int[numViews][3];
		for (int v = 0; v < numViews; v++) {
			varDims[v][0] = Z[v];
			varDims[v][1] = D;
			varDims[v][2] = W[v];
			
			// Kluge to store one integer per lock.  Should be fine so long as we have less than 10M words/topics/documents
			for (int w = 0; w < W[v]; w++) {
				int wLock = getLock(v, w);
				wordLocks[wLock] = wLock;
			}
			for (int d = 0; d < D; d++) {
				int dLock = getLock(0, d);
				docLocks[dLock] = dLock;
			}
			for (int z = 0; z < Z[v]; z++) {
				int zLock = getLock(v, z);
				wordLocks[zLock] = zLock;
			}
		}
		
		super.initialize(); // Spin up threads
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
		// TODO Auto-generated method stub
		
	}
	
	private void sample(int d, int v, int n) {
		int w = docs[d][v][n];
		int topic = docsZ[d][v][n];
		
		// decrement counts
		
		synchronized(topicLocks[topic]) {
			nZW[v][topic][w] -= 1;
			nZ[v][topic] -= 1;
			nDZ[d][v][topic] -= 1;
		}
		
		// sample new topic value 
		
		double[] p = new double[Z[v]];
		double pTotal = 0;
		
		for (int z = 0; z < Z[v]; z++) {
			p[z] = (nDZ[d][z] + priorDZ[d][v][z]) *
					(nZW[z][w] + priorZW[v][z][w]) / (nZ[z] + phiNorm[v][z]);
			
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
		
		synchronized(topicLocks[topic]) {
			nZW[v][topic][w] += 1;	
			nZ[v][topic] += 1;
			nDZ[d][v][topic] += 1;
		}
		
		// set new assignments
		docsZ[d][v][n] = topic;
	}
	
	
	@Override
	public void updateGradient(Tup2<Integer, Integer>[][] parameterRanges) {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public void doGradientStep(Tup2<Integer, Integer>[][] parameterRanges) {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public void readDocs(String filename) throws Exception {
		Tup4<Map<String, Integer>[], Map<Integer, String>[],
		     double[][][], int[][][]> loadedValues = IO.readTrainInput(filename, observedFactorSizes, numViews);
		
		wordMaps    = loadedValues._1();
		wordMapInvs = loadedValues._2();
		double[][][] observedValues = loadedValues._3(); // Factor -> Doc -> Component
		docs = loadedValues._4();
		
		D = docs.length;
		W = new int[numViews];
		for (int v = 0; v < numViews; v++) {
			W[v] = wordMaps[v].size();
		}
		
		// Assign component weights to observed factors
		int factorIdx = 0;
		for (Factor f : factors) {
			int[] W_subset = new int[f.viewIndices.length];
			for (int i = 0; i < f.viewIndices.length; i++) {
				W_subset[i] = W[f.viewIndices[i]];
			}
			
			// Initialize factors -- observed and latent.
			if (f.isObserved()) {
				f.initialize(observedValues[factorIdx], W_subset);
				factorIdx++;
			}
			else {
				f.initialize(W_subset);
			}
		}
		
	}
	
	public void writeOutput(String filename) throws Exception {
		writeOutput(filename, new File(filename).getParent()); // Write to the same directory as the input file.
	}
	
	@Override
	public void writeOutput(String filename, String outputDir) throws Exception {
		// TODO Auto-generated method stub
		String baseName = new File(filename).getName();
		
		// Write topic assignments file
		
		
		// Write bias terms
		
		// Write factor parameters
		
		// Serialize the model so we can use it/continue training later if necessary
		// Serialize this model so we can load it later
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

}
