package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;

import utils.Tup2;

/**
 * Should be equivalent to SpriteJoint model but allows for multiple,
 * possibly categorical, factors.  Factors may be observed, or latent
 * This is an attempt to make SpriteJoint more generic.  Not hopeful though.
 * 
 * @author adrianb
 *
 */
public class SpriteFactoredTopicModel extends ParallelTopicModel {
	public HashMap<String,Integer> wordMap;
	public HashMap<Integer,String> wordMapInv;
    
	protected int Z;
	protected int D;
	protected int W;
	protected int VIEWS; // Number of different views for the document
	
	// Current sampled topic assignments
	protected int[][] docsZ;
	protected int[][][] docsZZ;
	
	// Samples
	protected int[][] nDZ;
	protected int[]    nD;
	protected int[][] nZW;
	protected int[]    nZ;
	
	private int numFactorsObserved = 0;
	
	/**
	 * Gets the ranges over which we want to split our threads up for each
	 * view.  The length of \theta and \phi priors are assumed equal.
	 */
	private int[][] getDataRanges(SpriteThetaPrior[] tpriors, SpritePhiPrior[] ppriors) {
		int[][] ranges = new int[tpriors.length][3];
		for (int i = 0; i < tpriors.length; i++) {
			ranges[i][0] = tpriors[i].Z;
			ranges[i][0] = tpriors[i].D;
			ranges[i][0] = ppriors[i].V;
		}
		
		return ranges;
	}
	
	public SpriteFactoredTopicModel(SpriteThetaPrior[] thetaPriors0, SpritePhiPrior[] phiPriors0,
									int[] priorToView0, int numThreads0) {
		super(numThreads0, getDataRanges(thetaPriors0, phiPriors0));
		
		Z = Z0;
		initOmegaB = initOmegaB0;
		initDeltaB = initDeltaB0;
		factorSizes    = factorSizes0;
		factorObserved = factorObserved0;
		for (int i = 0; i < factorObserved.length; i++) {
			if ( factorObserved[i] ) numFactorsObserved++;
		}
		
		// Stores document labels
		docsC = new double[numFactorsObserved][D][];
		int j = 0;
		for (int i = 0; i < factorObserved.length; i++) {
			if ( factorObserved[i] ) {
				for (int k = 0; k < D; k++) {
					docsC[j][k] = new double[factorSizes[i]];
				}
				j++;
			}
		}
		
		factorRhos = factorRhos0;
		factorNames = factorNames0;
	}
	
	@Override
	public void initialize() {
		deltaBias = new double[Z];
		thetaNorm = new double[D];
		priorDZ = new double[D][Z];
		
		omegaBias = new double[W];
		phiNorm = new double[Z];
		priorZW = new double[Z][W];
		
		// Initialize gradient
		adaOmegaBias = new double[W];
		adaDeltaBias = new double[Z];
		gradientOmegaBias = new double[W];
		gradientDeltaBias = new double[Z];
		
		docsZ = new int[D][];
		docsZZ = new int[D][][];
		
		nDZ = new int[D][Z];
		nD = new int[D];
		nZW = new int[Z][W];
		nZ = new int[Z];
		
		// Init factors
		factors = new Factor[factorSizes.length];
		int j = 0;
		for (int i = 0 ; i < factorSizes.length; i++) {
			if ( factorObserved[i] ) {
				factors[i] = new Factor(docsC[j], Z, W, factorRhos[i], factorNames[i]);
				j++;
			}
			else {
				factors[i] = new Factor(factorSizes[i], Z, W, D, factorRhos[i], factorNames[i]);
			}
		}
		
		for (int z = 0; z < Z; z++) {
			deltaBias[z] = initDeltaB;
		}
		for (int w = 0; w < W; w++) {
			omegaBias[w] = initOmegaB;
		}
		
		varDims = new int[] {Z, D, W};
		
		super.initialize(); // Spin up threads
	}
	
	@Override
	public void logIteration() {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public void collectSamples() {
		// TODO Auto-generated method stub

	}

	@Override
	public double computeLL(int[][] corpus) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void updatePriors(List<Tup2<Integer, Integer>> parameterRanges) {
		// TODO Auto-generated method stub

	}

	@Override
	public void sampleBatch(List<Tup2<Integer, Integer>> parameterRanges) {
		// TODO Auto-generated method stub

	}

	@Override
	public void updateGradient(List<Tup2<Integer, Integer>> parameterRanges) {
		// TODO Auto-generated method stub

	}

	@Override
	public void doGradientStep(List<Tup2<Integer, Integer>> parameterRanges) {
		// TODO Auto-generated method stub

	}

	@Override
	public void readDocs(String filename) throws Exception {
		System.out.println("Reading input...");
		
		wordMap = new HashMap<String,Integer>();
		wordMapInv = new HashMap<Integer,String>();
		
		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr); 
		
		String s;
		
		D = 0;
		while((s = br.readLine()) != null) {
			D++;
		}
		
		docs = new int[D][];
		fr.close();
		
		fr = new FileReader(filename);
		br = new BufferedReader(fr); 
		
		int d = 0;
		while ((s = br.readLine()) != null) {
			String[] tokens = s.split("\\s+");
			
			int N = tokens.length;
			
			docs[d] = new int[N-2];
			
			for (int i = 0; i < numFactorsObserved; i++) {
				String[] factorValues = tokens[1+i].split(",");
				for (int j = 0; j < docsC[i][d].length; j++) {
					docsC[i][d][j] = Double.parseDouble(factorValues[j]);
				}
			}
			
			for (int n = (numFactorsObserved+1); n < N; n++) {
				String word = tokens[n];
				
				int key = wordMap.size();
				if (!wordMap.containsKey(word)) {
					wordMap.put(word, new Integer(key));
					wordMapInv.put(new Integer(key), word);
				}
				else {
					key = ((Integer) wordMap.get(word)).intValue();
				}
				
				docs[d][n-2] = key;
			}
			
			d++;
		}
		
		br.close();
		fr.close();
		
		W = wordMap.size();
		
		System.out.println(D + " documents");
		System.out.println(W + " word types");
	}
	
	@Override
	public void writeOutput(String filename) throws Exception {
		// TODO Auto-generated method stub
		
	}

}
