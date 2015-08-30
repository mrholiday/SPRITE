package models.original;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;

import utils.MathUtils;
import main.TopicModel;


// TODO: Broken.  Need to correctly index to docs.  Same goes for all the other old models where docs was changed from 2-d array to 3-d.

/**
 * SPRITE model used for ICWSM paper with only a single component.
 * The component coefficient \alpha is estimated for each document
 * but has a Gaussian prior whose mean is a function of input values
 * (hashtag-based gun control stance and state-level gun ownership rates) 
 */
public class SpriteICWSMNoSupervisedOmegaFixedMean extends TopicModel implements Serializable {
    
	/**
	 * 
	 */
	private static final long serialVersionUID = 6274172089209888876L;
	
	double eps = 1.0e-6; // small epsilon to stabilize digamma
	
	public HashMap<String,Integer> wordMap;
	public HashMap<Integer,String> wordMapInv;
	
	private Random r;
	
	public String priorPrefix;
	
	public BigInteger[] docIds; // Only written to output, tweet summarization easier
	public     double[] docsC0; // stance
	public     double[] docsC1; // ownership
	public     double[] docsC2; // sentiment
	
	public int[][] docsZ;
	public int[][][] docsZZ;
	
	public int[][] nDZ;
	public int[] nD;
	public int[][] nZW;
	public int[] nZ;
    
	public int D;
	public int W;
	public int Z;
	public int Cth;
	public int Cph;
	
	public double deltaB;
	public double[][] delta;	
	public double[] deltaBias;
	public double[][] priorDZ;
	public double[] thetaNorm;
	public double[][] alpha;
    
	public double omegaB;
	//public double[][] omega;
	public double[] omegaBias;
	public double[][] priorZW;
	public double[] phiNorm;
	public double[][] beta;
	public double[][] betaB;
	
	public double lambda0; // Controls weight of document score 0
	public double lambda1; // Controls weight of document score 1
	
	public double stepA;
	public double stepSizeADZ;
	public double stepSizeAZ;
	public double stepSizeAB;
	public double stepSizeW;
	public double stepSizeWB;
	public double stepSizeB;
	
	protected double sigmaAlpha; // How far away alpha can deviate from the mean interpolated score
	public double sigmaA;
	public double sigmaAB;
	public double sigmaW;
	public double sigmaWB;
	
	public double sigmaDelta;
	public double sigmaDeltaBias;
	public double sigmaOmega;
	public double sigmaOmegaBias;
	
	public int likelihoodFreq;
	
	// Pulled out to compute gradient in parallel.
	//double[][] gradientOmega;
	double[] gradientOmegaBias;
	double[][] gradientBeta;
	
	double[][] gradientDelta;
	double[] gradientDeltaBias;
	double[][] gradientAlpha;
	
	double gradientLambda0;
	double gradientLambda1;
	
	//double[][] adaOmega;
	double[] adaOmegaBias;
	double[][] adaBeta;
	double[][] adaDelta;
	double[] adaDeltaBias;
	double[][] adaAlpha;
	double adaLambda0;
	double adaLambda1;
	
	public int seed;
	
	// Parallel nonsense
	private int numThreads;
	
	private enum ThreadComm {UPDATE, KILL, SAMPLE, CALC_GRADIENT, GRADIENT_STEP};
	
	private transient ArrayBlockingQueue<ThreadComm> THREAD_COMM_QUEUE = null; // Signals workers have a job to do
    private transient ArrayBlockingQueue<String>     THREAD_DONE_QUEUE = null; // Signals workers are done
    
	private transient Worker[] THREADS = null;
	
	private Integer[] wordLocks;
	private Integer[] topicLocks;
	private Integer[] docLocks;
	
	private boolean computePerplexity; // If true, will train on half of tokens, and print out held-out perplexity.
	
	public SpriteICWSMNoSupervisedOmegaFixedMean(int z, double sigmaAlpha0, double sigmaA0, double sigmaAB0, double sigmaW0, double sigmaWB0,
			double deltaB0, double omegaB0, int likelihoodFreq0,
			String prefix, double stepA0, int seed0, int numThreads0, boolean computePerplexity0) {
		Z = z;
		
		topicLocks = new Integer[Z];
		for (int i = 0; i < Z; i++) {
			topicLocks[i] = (Integer)i;
		}
		
		sigmaAlpha = sigmaAlpha0;
		sigmaA = sigmaA0;
		sigmaAB = sigmaAB0;
		sigmaW = sigmaW0;
		sigmaWB = sigmaWB0;
		deltaB = deltaB0;
		omegaB = omegaB0;
		
		sigmaDelta = sigmaA0;
		sigmaDeltaBias = sigmaAB0;
		sigmaOmega = sigmaW0;
		sigmaOmegaBias = sigmaWB0;
		
		stepA = stepA0;
		
		// No supertopics in this model
		Cth = 1;
		Cph = 1;
		
		likelihoodFreq = likelihoodFreq0;
		priorPrefix = prefix;
		
		seed = seed0;
		
		numThreads = numThreads0;
		computePerplexity = computePerplexity0;
	}
	
	public void initTrain() {
		System.out.println("Initializing...");
		
		System.out.println("seed = "+seed);
		if (seed == -1) r = new Random();
		else r = new Random(seed);
		
		System.out.println("sigmaA = "+sigmaA);
		System.out.println("sigmaW = "+sigmaW);
		System.out.println("stepSizeADZ = "+stepSizeADZ);
		System.out.println("stepSizeAZ = "+stepSizeAZ);
		System.out.println("stepSizeAB = "+stepSizeAB);
		System.out.println("stepSizeW = "+stepSizeW);
		System.out.println("stepSizeWB = "+stepSizeWB);
		System.out.println("stepSizeB = "+stepSizeB);
		System.out.println("deltaB = "+deltaB);
		System.out.println("omegaB = "+omegaB);
		
		alpha = new double[D][Cth];
		delta = new double[Cth][Z];
		deltaBias = new double[Z];
		thetaNorm = new double[D];
		priorDZ = new double[D][Z];
		
		beta = new double[Z][Cph];
		betaB = new double[Z][Cph];
		//omega = new double[Cph][W];
		omegaBias = new double[W];
		phiNorm = new double[Z];
		priorZW = new double[Z][W];
		
		lambda0 = 1.0;
		lambda1 = 1.0;
		
		//adaOmega = new double[Cph][W];
		adaOmegaBias = new double[W];
		adaBeta = new double[Z][Cph];
		//adaBetaB = new double[Z][Cph];
		adaDelta = new double[Cth][Z];
		adaDeltaBias = new double[Z];
		adaAlpha = new double[D][Cth];
		
		// Initialize gradient
		//gradientOmega = new double[Cph][W];
		gradientOmegaBias = new double[W];
		gradientBeta = new double[Z][Cph];
		//gradientBetaB = new double[Z][Cph];
		
		gradientDelta   = new double[Cth][Z];
		gradientDeltaBias = new double[Z];
		gradientAlpha   = new double[D][Cth];
		
		docsZ = new int[D][];
		docsZZ = new int[D][][];
		
		nDZ = new int[D][Z];
		nD = new int[D];
		nZW = new int[Z][W];
		nZ = new int[Z];
		
		for (int d = 0; d < D; d++) {
			alpha[d][0] = alphaMean(d);
		}
		
		for (int z = 0; z < Z; z++) {
			deltaBias[z] = deltaB;
		}
		
		for (int w = 0; w < W; w++) {
			omegaBias[w] = omegaB;
		}
		
		for (int c = 1; c < Cth; c++) { 
				for (int z = 0; z < Z; z++) {
				delta[c][z] = (r.nextDouble() - 0.5) / 100.0;
				delta[c][z] += -2.0;
			}
		}
		
		/*
		for (int c = 0; c < Cph; c++) {
			for (int w = 0; w < W; w++) {
				omega[c][w] = (r.nextDouble() - 0.5) / 100.0;
				//omega[c][w] += -2.0;
			}
		}
		*/
		
		for (int z = 0; z < Z; z++) {
			for (int c = 1; c < Cph; c++) {
				betaB[z][c] = 1.0;
				//beta[z][c] = -2.0;
				//beta[z][c] = 0.0;
				beta[z][c] = delta[c][z];
			}
		}
		
		for (int z = 0; z < Z; z++) {
			phiNorm[z] = 0;
			for (int w = 0; w < W; w++) {
				priorZW[z][w] = priorW(z, w);
				phiNorm[z] += priorZW[z][w];
			}
		}
		
		for (int d = 0; d < D; d++) {
			thetaNorm[d] = 0;
			for (int z = 0; z < Z; z++) {
				priorDZ[d][z] = priorA(d, z); 
				thetaNorm[d] += priorDZ[d][z];
			}
		}
		
		for (int d = 0; d < D; d++) { 
			docsZ[d] = new int[docs[0][d].length];
			docsZZ[d] = new int[docs[0][d].length][Z];
			
			if (!computePerplexity) {
				for (int n = 0; n < docs[0][d].length; n++) {
					int w = docs[0][d][n];
					
					int z = r.nextInt(Z); // sample uniformly
					docsZ[d][n] = z;
					
					// update counts
					
					nZW[z][w] += 1;	
					nZ[z] += 1;
					nDZ[d][z] += 1;
					nD[d] += 1;
				}
			}
			else { // Train on every other token.  Ignore the other tokens.
				for (int n = 0; n < docs[0][d].length; n += 2) {
					int w = docs[0][d][n];
					
					int z = r.nextInt(Z); // sample uniformly
					docsZ[d][n] = z;
					
					// update counts
					
					nZW[z][w] += 1;	
					nZ[z] += 1;
					nDZ[d][z] += 1;
					nD[d] += 1;
				}
			}
		}
		
		// Spin up worker threads.  Will only perform work when ThreadComm message is received.
		//if (numThreads > 1) {
		THREAD_COMM_QUEUE = new ArrayBlockingQueue<ThreadComm>(numThreads);
		THREAD_DONE_QUEUE = new ArrayBlockingQueue<String>(numThreads);
		THREADS     = new Worker[numThreads];
		
		int dStep = D/numThreads;
		int zStep = Z/numThreads;
		int wStep = W/numThreads;
		for (int i = 0; i < numThreads; i++) {
			int minW = wStep*i;
			int maxW = i < (numThreads-1) ? wStep*(i+1) : W;
			int minD = dStep*i;
			int maxD = i < (numThreads-1) ? dStep*(i+1) : D;
			int minZ = zStep*i;
			int maxZ = i < (numThreads-1) ? zStep*(i+1) : Z;
			THREADS[i] = new Worker(minW, maxW, minD, maxD, minZ, maxZ, i);
			THREADS[i].start();
		}
		//}
	}

	// returns the mean of the Gaussian prior for the document's alpha value
	public double alphaMean(int d) {
		return (lambda1 * docsC1[d]) + (lambda0 * docsC0[d]);
	}
	
	// returns the delta_dz prior given all the parameters
	public double priorA(int d, int z) {
		double weight = deltaBias[z];
		
		for (int c = 0; c < 1; c++) {
			weight += alpha[d][c] * delta[c][z];
		}
		/*//for (int c = 1; c < Cth; c++) {
		for (int c = 3; c < Cth; c++) {
			weight += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]);
		}*/
		
		return Math.exp(weight);
	}
	
	// returns the omega_zw prior given all the parameters
	public double priorW(int z, int w) {
		double weight = omegaBias[w];
		
		/*
		for (int c = 0; c < 1; c++) {
			weight += beta[z][c] * omega[c][w];
		}
		*/
		
		/*//for (int c = 1; c < Cph; c++) {
		for (int c = 3; c < Cph; c++) {
			weight += betaB[z][c] * Math.exp(beta[z][c]) * omega[c][w];
		}*/
		
		return Math.exp(weight);
	}
	
	/*
	 * Updates the gradients for a subset of topics and words.
	 */
	public void updateGradient(int iter, int minZ, int maxZ) {
		// compute gradients
		
		for (int z = minZ; z < maxZ; z++) {
			for (int w = 0; w < W; w++) {
				double dg1  = MathUtils.digamma(phiNorm[z] + eps);
				double dg2  = MathUtils.digamma(phiNorm[z] + nZ[z] + eps);
				double dgW1 = MathUtils.digamma(priorZW[z][w] + nZW[z][w] + eps);
				double dgW2 = MathUtils.digamma(priorZW[z][w] + eps);
				
				double gradientTerm = priorZW[z][w] * (dg1-dg2+dgW1-dgW2);
				
//				for (int c = 0; c < 1; c++) {
//					gradientBeta[z][c] += omega[c][w] * gradientTerm;
//					
//					synchronized(wordLocks[w]) {
//					  gradientOmega[c][w] += beta[z][c] * gradientTerm;
//					}
//				}
				/*//for (int c = 1; c < Cph; c++) { // hierarchy
				for (int c = 3; c < Cph; c++) { // hierarchy
					gradientBeta[z][c]  += betaB[z][c] * Math.exp(beta[z][c]) * omega[c][w] * gradientTerm;
					gradientBetaB[z][c] += Math.exp(beta[z][c]) * omega[c][w] * gradientTerm;
					gradientOmega[c][w] += betaB[z][c] * Math.exp(beta[z][c]) * gradientTerm;
				}*/
				
				//gradientBeta[z][0] += omega[0][w] * gradientTerm;
				
				synchronized(wordLocks[w]) {
				  //gradientOmega[0][w] += beta[z][0] * gradientTerm;
				  gradientOmegaBias[w] += gradientTerm;
				}
			}
		}
		
		for (int z = minZ; z < maxZ; z++) {
			for (int d = 0; d < D; d++) {
				double dg1  = MathUtils.digamma(thetaNorm[z] + eps);
				double dg2  = MathUtils.digamma(thetaNorm[z] + nD[d] + eps);
				double dgW1 = MathUtils.digamma(priorDZ[d][z] + nDZ[d][z] + eps);
				double dgW2 = MathUtils.digamma(priorDZ[d][z] + eps);
				
				double gradientTerm = priorDZ[d][z] * (dg1-dg2+dgW1-dgW2);
				
//				for (int c = 0; c < 1; c++) { // factor
//					gradientAlpha[d][c] += delta[c][z] * gradientTerm;
//					gradientBeta[z][c] += alpha[d][c] * gradientTerm;
//				}
				/*for (int c = 3; c < Cth; c++) { // hierarchy
					gradientAlpha[d][c] += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					gradientDelta[c][z] += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					gradientBetaB[z][c] += Math.exp(alpha[d][c]) * Math.exp(delta[c][z]) * gradientTerm;
				}*/
//				synchronized(docLocks[d]) {
//					gradientAlpha[d][0] += delta[0][z] * Math.exp(delta[0][z]) * gradientTerm;
//				}
				gradientBeta[z][0] += alpha[d][0] * gradientTerm;
				gradientDeltaBias[z] += gradientTerm;
			}
		}
	}
	
	public void doGradientStep(int iter, int minZ, int maxZ, int minW, int maxW, int minD, int maxD) {
		/*
		 * Take a gradient step for a subset of topics and words.  Assumes gradient is current.
		 */
		// gradient ascent
		
		double step = stepA;
		double stepB = 1.0;
		
		/*double sigma0 = 0.5;
		  double sigmaBeta = 0.5;
		  double sigmaOmega = 0.5;
		  double sigmaOmegaBias = 1.0;
		  double sigmaAlpha = 0.5;
		  double sigmaDelta = 0.5;
		  double sigmaDeltaBias = 0.5;*/
		
		double sigma0 = 10.0;
//		double sigmaBeta = 10.0;
//		double sigmaOmega = 10.0;
		double sigmaOmegaBias = 10.0;
//		double sigmaDelta = 10.0;
//		double sigmaDeltaBias = 10.0;
		
		for (int z = minZ; z < maxZ; z++) {
			for (int c = 0; c < 1; c++) {
				gradientBeta[z][c] += -(beta[z][c]) / Math.pow(sigma0, 2);
				adaBeta[z][c] += Math.pow(gradientBeta[z][c], 2);
				beta[z][c] += (step / (Math.sqrt(adaBeta[z][c])+eps)) * gradientBeta[z][c];
				gradientBeta[z][c] = 0.; // Clear gradient for the next iteration
			}
			/*//for (int c = 1; c < Cph; c++) {
			for (int c = 3; c < Cph; c++) {
				gradientBeta[z][c] += -(beta[z][c]) / Math.pow(sigmaBeta, 2);
				adaBeta[z][c] += Math.pow(gradientBeta[z][c], 2);
				beta[z][c] += (step / (Math.sqrt(adaBeta[z][c])+eps)) * gradientBeta[z][c];
				//beta[z][c] = 0.0; // unweighted
				gradientBeta[z][c] = 0.;
			}*/
		}
		
		// this block below was for BetaB, which isn't used in this model
		/*double dirp = 0.01; // Dirichlet hyperparameter
		double rho = 0.95; // AdaDelta weighting
		//double priorTemp = 0.0; // no prior
		double priorTemp = Math.pow(1.00, iter-199);
		System.out.println("priorTemp = "+priorTemp);
		double[][] prevBetaB = new double[Z][Cph];
		for (int z = minZ; z < maxZ; z++) {
			double norm = 0.0;
			//for (int c = 1; c < Cph; c++) {
			for (int c = 3; c < Cph; c++) {
				//System.out.println("gradient "+gradientBeta[z][c]);
				double prior = (dirp - 1.0) / betaB[z][c];
				//System.out.println("prior "+prior);
				
				//adaBetaB[z][c] += Math.pow(gradientBetaB[z][c], 2); // traditional update
				if (adaBetaB[z][c] == 0) adaBetaB[z][c] = 1.0;
				adaBetaB[z][c] = (rho * adaBetaB[z][c]) + ((1.0-rho) * Math.pow(gradientBetaB[z][c], 2)); // average
				gradientBetaB[z][c] += priorTemp * prior; // exclude from adaBetaB
				prevBetaB[z][c] = betaB[z][c]; // store in case update goes wrong
				betaB[z][c] *= Math.exp((step / (Math.sqrt(adaBetaB[z][c])+eps)) * gradientBetaB[z][c]);
				
				norm += betaB[z][c];
				gradientBetaB[z][c] = 0.;
			}
			//for (int c = 0; c < 1; c++) {
			for (int c = 0; c < 3; c++) {
				betaB[z][c] = 1.0;
			}
			//System.out.println("normalizer "+z+" "+norm);
			//for (int c = 1; c < Cph; c++) {
			for (int c = 3; c < Cph; c++) {
				if (norm == 0.0 || norm == Double.POSITIVE_INFINITY) {
					System.out.println("BAD betaB");
					//betaB[z][c] = 1.0 / (Cph - 1);
					betaB[z][c] = prevBetaB[z][c]; // undo update if not well defined
				}
				else {
					betaB[z][c] /= norm;
				}
			}
		}*/
		
		/*
		for (int c = 0; c < Cph; c++) {
			for (int w = minW; w < maxW; w++) {
				gradientOmega[c][w] += -(omega[c][w]) / Math.pow(sigmaOmega, 2);
				adaOmega[c][w] += Math.pow(gradientOmega[c][w], 2);
				omega[c][w] += (step / (Math.sqrt(adaOmega[c][w])+eps)) * gradientOmega[c][w];
				gradientOmega[c][w] = 0.; // Clear gradient for the next iteration
			}
		}
		*/
		
		for (int w = minW; w < maxW; w++) {
			gradientOmegaBias[w] += -(omegaBias[w]) / Math.pow(sigmaOmegaBias, 2);
			adaOmegaBias[w] += Math.pow(gradientOmegaBias[w], 2);
			omegaBias[w] += (step / (Math.sqrt(adaOmegaBias[w])+eps)) * gradientOmegaBias[w];
			gradientOmegaBias[w] = 0.; // Clear gradient for the next iteration
		}
		
		/*
		for (int d = minD; d < maxD; d++) {
			//for (int c = 1; c < Cth; c++) {
			for (int c = 0; c < Cth; c++) {
				gradientAlpha[d][c] += -(alpha[d][c] - alphaMean(d)) / Math.pow(sigmaAlpha, 2);
				adaAlpha[d][c] += Math.pow(gradientAlpha[d][c], 2);
				alpha[d][c] += (step / (Math.sqrt(adaAlpha[d][c])+eps)) * gradientAlpha[d][c];
				//if (alpha[d][c] < 1.0e-6) alpha[d][c] = 1.0e-6;
				gradientAlpha[d][c] = 0.; // Clear gradient for the next iteration
			}
		}
		*/
		
		/*
		 for (int c = 0; c < Cth; c++) {
		 	for (int z = minZ; z < maxZ; z++) {
				gradientDelta[c][z] += -(delta[c][z]) / Math.pow(sigmaDelta, 2);
				adaDelta[c][z] += Math.pow(gradientDelta[c][z], 2);
				delta[c][z] += (step / (Math.sqrt(adaDelta[c][z])+eps)) * gradientDelta[c][z];
				//delta[c][z] = 0.0; // unweighted
				gradientDelta[c][z] = 0.;
			}
		}
		*/
		
		for (int c = 0; c < 1; c++) {
			for (int z = minZ; z < maxZ; z++) {
				 delta[c][z] = beta[z][c];
			}
		}
		
		for (int z = minZ; z < maxZ; z++) {
			gradientDeltaBias[z] += -(deltaBias[z]) / Math.pow(sigmaDeltaBias, 2);
			adaDeltaBias[z] += Math.pow(gradientDeltaBias[z], 2);
			deltaBias[z] += (step / (Math.sqrt(adaDeltaBias[z])+eps)) * gradientDeltaBias[z];
			gradientDeltaBias[z] = 0.;
		}
	}
	
	// update lambda gradient and then do gradient step
	public void updateLambda() { }

	private class Worker extends Thread {
		/**
		 * Spins till message received.  Updates priors when there is a GO message in the queue.
		 */
		private int minZ;
		private int maxZ;
		private int minD;
		private int maxD;
		private int minW;
		private int maxW;
		public String threadName;
		private int threadNum;
		private int iter = 0;
		
		public Worker(int minWArg, int maxWArg, int minDArg, int maxDArg, int minZArg, int maxZArg, int threadNumArg) {
//			minC = minCArg;
//			maxC = maxCArg;
			minW = minWArg;
			maxW = maxWArg;
			minD = minDArg;
			maxD = maxDArg;
			minZ = minZArg;
			maxZ = maxZArg;
			threadName = String.format("Worker_%d-%d", minZ, maxZ);
			threadNum  = threadNumArg;
		}
		
		@Override
		public void run() {
			
			while (true) {
				try {
					ThreadComm msg = THREAD_COMM_QUEUE.take();
					if (msg.equals(ThreadComm.KILL)) {
						break;
					}
					else if (msg.equals(ThreadComm.UPDATE)) {
						updatePriors(minZ, maxZ);
					}
					else if (msg.equals(ThreadComm.SAMPLE)) {
						sampleBatch(minD, maxD);
						iter += 1;
					}
					else if (msg.equals(ThreadComm.CALC_GRADIENT)) {
//						updateGradient(iter, minZ, maxZ, minW, maxW);
						updateGradient(iter, minZ, maxZ);
					}
					else if (msg.equals(ThreadComm.GRADIENT_STEP)) {
						doGradientStep(iter, minZ, maxZ, minW, maxW, minD, maxD);
					}
					THREAD_DONE_QUEUE.put(threadName + " - " + true);
				} catch (InterruptedException e) {
					e.printStackTrace();
					try {
						THREAD_DONE_QUEUE.put(threadName + " - " + false);
					} catch (InterruptedException e1) {
						e1.printStackTrace();
					}
				}
			}
		}
		
		public void start() {
		      System.out.println("Starting " +  threadName );
		      Thread t = new Thread (this, threadName);
		      t.start();
		}
	}
	
	public void updatePriors(int minZ, int maxZ) {
		// Only update the priors for topics minZ to maxZ
		
		// compute the priors with the new params and update the cached prior variables 
		for (int z = minZ; z < maxZ; z++) {
			for (int d = 0; d < D; d++) {
				priorDZ[d][z] = priorA(d, z);
				thetaNorm[d] += priorDZ[d][z];
			}
			
			for (int w = 0; w < W; w++) {
				priorZW[z][w] = priorW(z, w);
				//if (w % 1000 == 0) System.out.println("prior "+z+" "+w+" "+priorZW[z][w]);
				phiNorm[z] += priorZW[z][w];
			}
			System.out.println("phiNorm"+z+" "+phiNorm[z]);
		}
	
	}
	
	// the E and M steps, for one iteration
	public void doSamplingIteration(int iter) {
		long startTime = System.currentTimeMillis();
		
		// sample z values for all the tokens
		try {
			for (int i = 0; i < numThreads; i++) {
				THREAD_COMM_QUEUE.put(ThreadComm.SAMPLE);
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		for (int i = 0; i < numThreads; i++) {
			try {
				String msg = THREAD_DONE_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		// hyperparameter updates	
		if (iter >= 200) {
			try {
				for (int i = 0; i < numThreads; i++) {
					THREAD_COMM_QUEUE.put(ThreadComm.CALC_GRADIENT);
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
			for (int i = 0; i < numThreads; i++) {
				try {
					String msg = THREAD_DONE_QUEUE.take();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			try {
				for (int i = 0; i < numThreads; i++) {
					THREAD_COMM_QUEUE.put(ThreadComm.GRADIENT_STEP);
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
			for (int i = 0; i < numThreads; i++) {
				try {
					String msg = THREAD_DONE_QUEUE.take();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			updateLambda();
		}
		
		for (int z = 0; z < Z; z++) {
			System.out.println("deltaBias_"+z+" "+deltaBias[z]);
		}
		for (int z = 0; z < Z; z++) {
			System.out.print("delta_"+z);
			for (int c = 0; c < Cth; c++) {
				System.out.print(" "+Math.exp(delta[c][z]));
			}
			System.out.println();
		}
		for (int z = 0; z < Z; z++) {
			System.out.print("beta_"+z);
			for (int c = 0; c < Cph; c++) {
				System.out.print(" "+beta[z][c]);
			}
			System.out.println();
		}
		
		/*
		for (int z = 0; z < Z; z++) {
			System.out.print("betaB_"+z);
			for (int c = 0; c < Cph; c++) {
				System.out.print(" "+betaB[z][c]);
			}
			System.out.println();
		}
		*/
		
		for (int w = 0; w < W; w += 10000) {
			System.out.println("omegaBias_"+w+" "+omegaBias[w]);
		}
		
		/*
		for (int w = 0; w < W; w += 10000) {
			System.out.print("omega_"+w);
			for (int c = 0; c < Cph; c++) {
				System.out.print(" "+omega[c][w]);
			}
			System.out.println();
		}
		*/
		
		System.out.println("lambda0 = "+lambda0);
		System.out.println("lambda1 = "+lambda1);
		
		thetaNorm = new double[D];
		phiNorm   = new double[Z];
		
		// compute the priors with the new params and update the cached prior variables 
		// Parallel update
		for (int i = 0; i < numThreads; i++) {
			try {
				THREAD_COMM_QUEUE.put(ThreadComm.UPDATE);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < numThreads; i++) {
			try {
				String msg = THREAD_DONE_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int d = 0; d < D; d++) {
			if (d % 10000 == 0) {
				System.out.print("alpha_"+d);
				for (int c = 0; c < Cth; c++) {
					System.out.print(" "+alpha[d][c]);
				}
				System.out.println();
			}
		}
		
		//for (int d = 0; d < D; d++) {
		//	if (d % 100 == 0) System.out.println("thetaNorm"+d+" "+thetaNorm[d]);
		//}
		
		if (iter % likelihoodFreq == 0) {
			System.out.println("Train Log-likelihood: " + computeLL(false));
			if (computePerplexity) {
				System.out.println("Held-out Log-likelihood: " + computeLL(true));
				System.out.println("Train Perplexity: " + computePerplexity(false));
				System.out.println("Held-out Perplexity: " + computePerplexity(true));
			}
		}
		
		// collect samples (docsZZ) 
		if (burnedIn) {
			for (int d = 0; d < D; d++) {
				if (!computePerplexity) {
					for (int n = 0; n < docs[0][d].length; n++) { 
						int x = docsZ[d][n];
						
						docsZZ[d][n][x] += 1;
					}
				}
				else { // Only collect samples for half of the tokens
					for (int n = 0; n < docs[0][d].length; n += 2) { 
						int x = docsZ[d][n];
						
						docsZZ[d][n][x] += 1;
					}
				}
			}
		}
		long endTime = System.currentTimeMillis();
		
		System.out.println(String.format("Iteration time:\t%d\t%d", iter, endTime - startTime));
	}
    
	public void sampleBatch(int minD, int maxD) {
		for (int d = minD; d < maxD; d++) {
			if (!computePerplexity) {
				for (int n = 0; n < docs[0][d].length; n++) {
					sample(d, n);
				}
			}
			else {
				for (int n = 0; n < docs[0][d].length; n += 2) {
					sample(d, n);
				}
			}
			if (d % 10000 == 0) {
				System.out.println("Done document " + d);
			}
		}
	}
	
	public void sample(int d, int n) {
		int w = docs[0][d][n];
		int topic = docsZ[d][n];
		
		// decrement counts
		
		synchronized(topicLocks[topic]) {
			nZW[topic][w] -= 1;
			nZ[topic] -= 1;
			nDZ[d][topic] -= 1;
		}
		
		// sample new topic value 
		
		double[] p = new double[Z];
		double pTotal = 0;
		
		for (int z = 0; z < Z; z++) {
			p[z] = (nDZ[d][z] + priorDZ[d][z]) *
					(nZW[z][w] + priorZW[z][w]) / (nZ[z] + phiNorm[z]);
			
			pTotal += p[z];
		}
		
		double u = r.nextDouble() * pTotal;
		
		double v = 0.0;
		for (int z = 0; z < Z; z++) {
			v += p[z];
			
			if (v > u) {
				topic = z;
				break;
			}
		}
		
		// increment counts
		
		synchronized(topicLocks[topic]) {
			nZW[topic][w] += 1;	
			nZ[topic] += 1;
			nDZ[d][topic] += 1;
		}
		
		// set new assignments
		docsZ[d][n] = topic;
	}
	
	public double computePerplexity(boolean isHeldOut) {
		double perplexity = 0;
		double denom = 0.0;
		double logProbSum = 0.0;
		
		for (int d = 0; d < D; d++) {
			int startN = isHeldOut ? 1 : 0;
			for (int n = startN; n < docs[0][d].length; n += 2) {
				int w = docs[0][d][n];
				
				double tokenLL = 0;
				
				// marginalize over z
				
				for (int z = 0; z < Z; z++) {
					tokenLL += (nDZ[d][z] + priorDZ[d][z]) / (nD[d] + thetaNorm[d])*
							(nZW[z][w] + priorZW[z][w]) / (nZ[z] + phiNorm[z]);
				}
				
				logProbSum += MathUtils.log(tokenLL, 2.0);
				denom++;
			}
		}
		
		perplexity = Math.pow(2.0, -logProbSum/denom);
		return perplexity;
	}
	
	public double computeLL(boolean isHeldOut) {
		double LL = 0;
		
		for (int d = 0; d < D; d++) {
			
			if (isHeldOut) { // Compute LL on the held-out set
				for (int n = 1; n < docs[0][d].length; n += 2) {
					int w = docs[0][d][n];
					
					double tokenLL = 0;
					
					// marginalize over z
					
					for (int z = 0; z < Z; z++) {
						tokenLL += (nDZ[d][z] + priorDZ[d][z]) / (nD[d] + thetaNorm[d])*
								(nZW[z][w] + priorZW[z][w]) / (nZ[z] + phiNorm[z]);
					}
					
					LL += Math.log(tokenLL);
				}
			}
			else { // Compute LL over the training data
				if (!computePerplexity) { // LL over all examples
					for (int n = 0; n < docs[0][d].length; n++) { 
						int w = docs[0][d][n];
						
						double tokenLL = 0;
						
						// marginalize over z
						
						for (int z = 0; z < Z; z++) {
							tokenLL += (nDZ[d][z] + priorDZ[d][z]) / (nD[d] + thetaNorm[d])*
									(nZW[z][w] + priorZW[z][w]) / (nZ[z] + phiNorm[z]);
						}
						
						LL += Math.log(tokenLL);
					}
				}
				else {
					for (int n = 0; n < docs[0][d].length; n += 2) { 
						int w = docs[0][d][n];
						
						double tokenLL = 0;
						
						// marginalize over z
						
						for (int z = 0; z < Z; z++) {
							tokenLL += (nDZ[d][z] + priorDZ[d][z]) / (nD[d] + thetaNorm[d])*
									(nZW[z][w] + priorZW[z][w]) / (nZ[z] + phiNorm[z]);
						}
						
						LL += Math.log(tokenLL);
					}
				}
			}
		}

		return LL;
	}
	
	// computes the log-likelihood of the corpus
	// this marginalizes over the hidden variables but not the parameters
	// (i.e. we condition on the current estimates of \theta and \phi)
	public double computeLL(int[][] docs) {
		double LL = 0;
		
		for (int d = 0; d < D; d++) {
			for (int n = 0; n < docs[d].length; n++) { 
				int w = docs[d][n];
				
				double tokenLL = 0;
				
				// marginalize over z
				
				for (int z = 0; z < Z; z++) {
					tokenLL += (nDZ[d][z] + priorDZ[d][z]) / (nD[d] + thetaNorm[d])*
				  		(nZW[z][w] + priorZW[z][w]) / (nZ[z] + phiNorm[z]);
				}
				
				LL += Math.log(tokenLL);
			}
		}
		
		return LL;
	}
	
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
		
		docLocks = new Integer[D];
		for (int i = 0; i < D; i++) {
			docLocks[i] = (Integer)i;
		}
		
		docIds = new BigInteger[D];
		docs = new int[1][D][];
		docsC0 = new double[D];
		docsC1 = new double[D];
		docsC2 = new double[D];
		
		fr.close();
		
		fr = new FileReader(filename);
		br = new BufferedReader(fr); 
		
		int d = 0;
		while ((s = br.readLine()) != null) {
			String[] tokens = s.split("\\s+");
			
			int N = tokens.length;
			
			docs[0][d] = new int[N-4];
			docIds[d]  = new BigInteger(tokens[0]);
			docsC0[d]  = Double.parseDouble(tokens[1]); // So gun stance points in the same direction as ownership
			docsC1[d]  = Double.parseDouble(tokens[2]); 
			docsC2[d]  = Double.parseDouble(tokens[3]);
			
			for (int n = 4; n < N; n++) {
				String word = tokens[n];
				
				int key = wordMap.size();
				if (!wordMap.containsKey(word)) {
					wordMap.put(word, new Integer(key));
					wordMapInv.put(new Integer(key), word);
				}
				else {
					key = ((Integer) wordMap.get(word)).intValue();
				}
				
				docs[0][d][n-4] = key;
			}
			
			d++;
		}
		
		br.close();
		fr.close();
		
		W = wordMap.size();
		
		wordLocks = new Integer[W];
		for (int i = 0; i < W; i++) {
			wordLocks[i] = (Integer)i;
		}
		
		System.out.println(D + " documents");
		System.out.println(W + " word types");
	}
	
	public void writeOutput(String filename, String outputDir) throws Exception {
		String baseName = new File(filename).getName();
		
		FileWriter fw = new FileWriter(new File(outputDir, baseName + ".assign"));
		BufferedWriter bw = new BufferedWriter(fw);
		
		for (int d = 0; d < D; d++) {
			bw.write(docIds[d] + " ");
			bw.write(docsC0[d] + " ");
			bw.write(docsC1[d] + " ");
			bw.write(docsC2[d] + " ");
			
			for (int n = 0; n < docs[0][d].length; n++) {
				String word = wordMapInv.get(docs[0][d][n]);
				
				//bw.write(word+":"+docsZ[d][n]+" "); // only current sample
				bw.write(word);  // for multiple samples
				for (int zz = 0; zz < Z; zz++) {
					bw.write(":" + docsZZ[d][n][zz]);
				}
				bw.write(" ");
			}
			bw.newLine();
		}
		
		bw.close();
		fw.close();
		
		fw = new FileWriter(new File(outputDir, baseName + ".beta"));
		bw = new BufferedWriter(fw);
		for (int z = 0; z < Z; z++) { 
			//for (int c = 0; c < 1; c++) {
			//for (int c = 0; c < 3; c++) {
			//	bw.write(beta[z][c]+" ");
			//}
			//for (int c = 1; c < Cph; c++) {
			//for (int c = 0; c < Cph; c++) {
			//	//bw.write(beta[z][c]+" ");
			//	bw.write(Math.exp(beta[z][c])+" ");
			//}
			
			for (int c = 0; c < Cph; c++) {
				//bw.write(beta[z][c]+" ");
				bw.write(beta[z][c] + " ");
			}
			
			bw.newLine();
		}
		bw.close();
		fw.close();
		
		fw = new FileWriter(new File(outputDir, baseName+".betaB"));
		bw = new BufferedWriter(fw);
		for (int z = 0; z < Z; z++) { 
			for (int c = 0; c < Cph; c++) {
				bw.write(betaB[z][c]+" ");
			}
			bw.newLine();
		}
		bw.close();
		fw.close();
		
		
		fw = new FileWriter(new File(outputDir, baseName+".omega"));
		bw = new BufferedWriter(fw);
		for (int w = 0; w < W; w++) {
			String word = wordMapInv.get(w);
			bw.write(word);
			
			for (int c = 0; c < Cph; c++) { 
				//bw.write(" " + omega[c][w]);
				bw.write(" 0.0");
			}
			bw.newLine();
		}
		bw.close();
		fw.close();
		
		fw = new FileWriter(new File(outputDir, baseName+".omegaBias"));
		bw = new BufferedWriter(fw);
		for (int w = 0; w < W; w++) {
			String word = wordMapInv.get(w);
			bw.write(word);
			bw.write(" "+omegaBias[w]);
			bw.newLine();
		}
		bw.close();
		fw.close();

		fw = new FileWriter(new File(outputDir, baseName+".alpha"));
		bw = new BufferedWriter(fw);
		for (int d = 0; d < D; d++) {
			//for (int c = 0; c < 1; c++) {
			//bw.write(docsC0[d] + " ");
			//bw.write(docsC1[d] + " ");
			//bw.write(docsC2[d] + " ");
			
			for (int c = 0; c < Cth; c++) { 
				bw.write(alpha[d][c]+" ");
			}
			
			//for (int c = 1; c < Cth; c++) { 
			//for (int c = 0; c < 3; c++) {
			//	bw.write(alpha[d][c]+" ");
			//}
			//for (int c = 1; c < alpha[d].length; c++) { 
			//	bw.write(Math.exp(alpha[d][c])+" ");
			//}
			bw.newLine();
		}
		bw.close();
		fw.close();
		
		fw = new FileWriter(new File(outputDir, baseName+".delta"));
		bw = new BufferedWriter(fw);
		for (int z = 0; z < Z; z++) {
			bw.write(""+z);
			
			//for (int c = 0; c < 1; c++) { 
			//for (int c = 0; c < 3; c++) { 
			//	bw.write(" "+delta[c][z]);
			//}
			
			//for (int c = 1; c < Cth; c++) { 
			for (int c = 0; c < Cth; c++) { 
				bw.write(" "+Math.exp(delta[c][z]));
			}
			bw.newLine();
		}
		bw.close();
		fw.close();
		
		fw = new FileWriter(new File(outputDir, baseName+".deltaBias"));
		bw = new BufferedWriter(fw);
		for (int z = 0; z < Z; z++) {
			bw.write(""+z);
			bw.write(" "+deltaBias[z]);
			bw.newLine();
		}
		bw.close();
		fw.close();
		
		fw = new FileWriter(new File(outputDir, baseName+".lambda"));
		bw = new BufferedWriter(fw);
		bw.write(lambda0 + " " + lambda1);
		bw.close();
		fw.close();
		
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
	
	public void cleanUp() {
		for (int i = 0; i < numThreads; i++) {
			try {
				THREAD_COMM_QUEUE.put(ThreadComm.KILL);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0 ; i < numThreads; i++) {
			try {
				THREADS[i].join(10000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		System.out.println("Killed prior update threads");
	}
	
	@Override
	public void logIteration() { }

	@Override
	public void collectSamples() { }

	@Override
	public double computeLL(int[][][] docs) {
		return computeLL(docs[0]);
	}

	@Override
	protected void initTest() {
		// TODO Auto-generated method stub
		
	}
	
}