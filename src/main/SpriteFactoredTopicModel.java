package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;

import utils.Tup2;
import utils.Tup4;

/**
 * Should be equivalent to SpriteJoint model but allows for multiple,
 * possibly categorical, factors.  Factors may be observed, or latent
 * This is an attempt to make SpriteJoint more generic.  Not hopeful though.
 * 
 * @author adrianb
 *
 */
public class SpriteFactoredTopicModel extends ParallelTopicModel {
	public Map<String, Integer>[] wordMaps;
	public Map<Integer,String>[] wordMapInvs;
    
	// All of the factors used across views.  Ordering should be the same as
	// the observed scores in the input files.
	protected Factor[] factors;
	
	protected int[] Z;   // Possibly different number of topics for each view.
	protected int D;
	protected int[] W;   // Each view has its own alphabet.
	protected int numViews = 0; // Number of different views for the document
	
	protected int[][][] docs;
	
	// Current sampled topic assignments
	protected int[][] docsZ;
	protected int[][][] docsZZ;
	
	// Samples
	protected int[][][] nDZ; // Document -> View -> Topic samples
	protected int[][]    nD; // Document -> View samples (number of tokens in each, set at initialization)
	protected int[][][] nZW; // View -> Topic -> Word samples
	protected int[][]    nZ; // View -> Topic samples
	
	protected int numFactorsObserved = 0;
	protected int[] observedFactorSizes;
	
	protected int[] priorToView; // Maps each theta/phi prior to the view it is responsible for.
	
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
			ranges[i][0] = tpriors[i].D;
			ranges[i][0] = ppriors[i].W;
		}
		
		return ranges;
	}
	
	public SpriteFactoredTopicModel(SpriteThetaPrior[] thetaPriors0, SpritePhiPrior[] phiPriors0,
									Factor[] factors0, int[] priorToView0, int numThreads0) {
		super.setParallelParams(numThreads0, getDataRanges(thetaPriors0, phiPriors0));
		
		thetaPriors = thetaPriors0;
		phiPriors   = phiPriors0;
		
		priorToView = priorToView0;
		factors     = factors0;
		
		numFactorsObserved = 0;
		for (Factor f : factors) {
			if (f.isObserved()) {
				numFactorsObserved++;
			}
		}
		
		// Initialize observed factor sizes for loading documents
		observedFactorSizes = new int[numFactorsObserved];
		int factorIdx = 0;
		for (Factor f : factors) {
			if (f.isObserved()) {
				observedFactorSizes[factorIdx] = f.C;
				factorIdx++;
			}
		}
		
		// Kind of annoying... maybe just pass numViews as a parameter
		for (int i = 0; i < priorToView.length; i++) {
			if ((priorToView[i]+1) > numViews+1) {
				numViews = priorToView[i] + 1;
			}
		}
		
		// Each view has its own set of topics.  May be joined by supertopics.
		Z = new int[numViews];
		for (int i = 0; i < priorToView.length; i++) {
			Z[priorToView[i]] = thetaPriors[i].Z;
		}
		
		numFactorsObserved = observedFactorSizes.length;
	}
	
	@Override
	public void initialize() {
		docsZ = new int[D][];
		docsZZ = new int[D][][];
		
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
		
		for (int z = 0; z < Z; z++) {
			deltaBias[z] = initDeltaB;
		}
		for (int w = 0; w < W; w++) {
			omegaBias[w] = initOmegaB;
		}
		
		varDims = new int[numViews][3];
		for (int v = 0; v < numViews; v++) {
			varDims[v][0] = Z[v];
			varDims[v][1] = D;
			varDims[v][2] = W[v];
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
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public double computeLL(int[][][] corpus) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void updatePriors(Tup2<Integer, Integer>[][] parameterRanges) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void sampleBatch(Tup2<Integer, Integer>[][] parameterRanges) {
		// TODO Auto-generated method stub
		
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
	
	@Override
	public void writeOutput(String filename) throws Exception {
		// TODO Auto-generated method stub
		
		
		
	}

}
