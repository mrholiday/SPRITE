package models.original.threefields;

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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;

import main.TopicModel;

import utils.MathUtils;
import utils.Tup2;

/**
 * SPRITE model used for ICWSM paper with only a single component.
 * The component coefficient \alpha is estimated for each document
 * but has a Gaussian prior whose mean is a function of input values
 * (hashtag-based gun control stance and state-level gun ownership rates).
 * Includes a third feature as well.
 */
public class SpriteICWSMThreeFields extends TopicModel implements Serializable {
    
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
	
	public double initialDeltaBias;
	public double[][] delta;	
	public double[] deltaBias;
	public double[][] priorDZ;
	public double[] thetaNorm;
	public double[][] alpha;
    
	public double initialOmegaBias;
	public double[][] omega;
	public double[] omegaBias;
	public double[][] priorZW;
	public double[] phiNorm;
	public double[][] beta;
	public double[][] betaB;
	
	public double lambda0; // Controls weight of document score 0
	public double lambda1;
	public double lambda2;
	
	public double step;
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
	double[][] gradientOmega;
	double[] gradientOmegaBias;
	double[][] gradientBeta;
	
	double[][] gradientDelta;
	double[] gradientDeltaBias;
	double[][] gradientAlpha;
	
	double gradientLambda0;
	double gradientLambda1;
	double gradientLambda2;
	
	double[][] adaOmega;
	double[] adaOmegaBias;
	double[][] adaBeta;
	double[][] adaDelta;
	double[] adaDeltaBias;
	double[][] adaAlpha;
	double adaLambda0;
	double adaLambda1;
	double adaLambda2;
	
	// For computing topic coherence
	int[] uniCounts;
	int[][] biCounts;
	
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
	
	public SpriteICWSMThreeFields(int z, double sigmaAlpha0, double sigmaA0, double sigmaAB0, double sigmaW0, double sigmaWB0,
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
		sigmaDelta = sigmaA0;
		sigmaDeltaBias = sigmaAB0;
		sigmaOmega = sigmaW0;
		sigmaOmegaBias = sigmaWB0;
		
		initialDeltaBias = deltaB0;
		initialOmegaBias = omegaB0;
		
		step = stepA0;
		
		// No supertopics in this model, just single distant perspective component
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
		System.out.println("deltaB = "+initialDeltaBias);
		System.out.println("omegaB = "+initialOmegaBias);
		
		alpha = new double[D][Cth];
		delta = new double[Cth][Z];
		deltaBias = new double[Z];
		thetaNorm = new double[D];
		priorDZ = new double[D][Z];
		
		beta = new double[Z][Cph];
		betaB = new double[Z][Cph];
		omega = new double[Cph][W];
		omegaBias = new double[W];
		phiNorm = new double[Z];
		priorZW = new double[Z][W];
		
		lambda0 = 1.0;
		lambda1 = 1.0;
		lambda2 = 1.0;
		
//		lambda0 = 0.0;
//		lambda1 = 0.0;
//		lambda2 = 0.0;
		
		adaOmega = new double[Cph][W];
		adaOmegaBias = new double[W];
		adaBeta = new double[Z][Cph];
		//adaBetaB = new double[Z][Cph];
		adaDelta = new double[Cth][Z];
		adaDeltaBias = new double[Z];
		adaAlpha = new double[D][Cth];
		
		// Initialize gradient
		gradientOmega = new double[Cph][W];
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
			deltaBias[z] = initialDeltaBias;
		}
		
		for (int w = 0; w < W; w++) {
			omegaBias[w] = initialOmegaBias;
		}
		
		for (int c = 0; c < Cth; c++) {
			for (int z = 0; z < Z; z++) {
				delta[c][z] = (r.nextDouble() - 0.5) / 100.0;
				//delta[c][z] += -2.0;
			}
		}
		
		for (int c = 0; c < Cph; c++) {
			for (int w = 0; w < W; w++) {
				omega[c][w] = (r.nextDouble() - 0.5) / 100.0;
				//omega[c][w] += -2.0;
			}
		}
		
		for (int z = 0; z < Z; z++) {
			for (int c = 0; c < Cph; c++) {
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
		
		// For topic coherence
		uniCounts = new int[W];
		biCounts  = new int[W][W];
				
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
			
			// For topic coherence
			Set<Integer> uniqueWords = new HashSet<Integer>();
			for (int n = 0; n < docs[0][d].length; n++) {
				uniqueWords.add(docs[0][d][n]);
			}
			for (int w : uniqueWords) {
				uniCounts[w] += 1;
				for (int w2 : uniqueWords) {
					biCounts[w][w2] += 1;
				}
			}
		}
		
		// Spin up worker threads.  Will only perform work when ThreadComm message is received.
		//if (numThreads > 1) {
		THREAD_COMM_QUEUE = new ArrayBlockingQueue<ThreadComm>(numThreads);
		THREAD_DONE_QUEUE = new ArrayBlockingQueue<String>(numThreads);
		THREADS     = new Worker[numThreads];
		
		float dStep = D/((float)numThreads);
		float zStep = Z/((float)numThreads);
		float wStep = W/((float)numThreads);
		
		for (int i = 0; i < numThreads; i++) {
			int minW = (int)(wStep*i);
			int maxW = i < (numThreads-1) ? (int)(wStep*(i+1)) : W;
			int minD = (int)(dStep*i);
			int maxD = i < (numThreads-1) ? (int)(dStep*(i+1)) : D;
			int minZ = (int)(zStep*i);
			int maxZ = i < (numThreads-1) ? (int)(zStep*(i+1)) : Z;
			THREADS[i] = new Worker(minW, maxW, minD, maxD, minZ, maxZ, i);
			THREADS[i].start();
		}
		//}
	}

	// returns the mean of the Gaussian prior for the document's alpha value
	public double alphaMean(int d) {
		return (lambda1 * docsC1[d]) + (lambda0 * docsC0[d]) + (lambda2 * docsC2[d]);
	}
	
	// returns the delta_dz prior given all the parameters
	public double priorA(int d, int z) {
		double weight = deltaBias[z];
		
		for (int c = 0; c < 1; c++) {
			weight += alpha[d][c] * delta[c][z];
		}
		
		return Math.exp(weight);
	}
	
	// returns the omega_zw prior given all the parameters
	public double priorW(int z, int w) {
		double weight = omegaBias[w];
		
		for (int c = 0; c < 1; c++) {
			weight += beta[z][c] * omega[c][w];
		}
		
		return Math.exp(weight);
	}
	
	/*
	 * Updates the gradients for a subset of topics and words.
	 */
	public void updateGradient(int iter, int minZ, int maxZ) {
		// compute gradients
		
		// For debugging with a single thread.  Want to see if regularization dominates the gradient updates
//		double gradientOmegaBNorm = 0.0;
//		double gradientDeltaBNorm = 0.0;
//		double gradientOmegaNorm  = 0.0;
//		double gradientBetaNorm   = 0.0;
		
		for (int z = minZ; z < maxZ; z++) {
			for (int w = 0; w < W; w++) {
				double dg1  = MathUtils.digamma(phiNorm[z] + eps);
				double dg2  = MathUtils.digamma(phiNorm[z] + nZ[z] + eps);
				double dgW1 = MathUtils.digamma(priorZW[z][w] + nZW[z][w] + eps);
				double dgW2 = MathUtils.digamma(priorZW[z][w] + eps);
				
				double gradientTerm = priorZW[z][w] * (dg1-dg2+dgW1-dgW2);
				
				gradientBeta[z][0] += omega[0][w] * gradientTerm;
//				gradientBetaNorm += (omega[0][w] * gradientTerm) * (omega[0][w] * gradientTerm);
				
				synchronized(wordLocks[w]) {
				  gradientOmega[0][w]  += beta[z][0] * gradientTerm;
				  gradientOmegaBias[w] += gradientTerm;
//				  gradientOmegaNorm    += (beta[z][0] * gradientTerm) * (beta[z][0] * gradientTerm);
//				  gradientOmegaBNorm   += gradientTerm * gradientTerm;
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
				
				synchronized(docLocks[d]) {
					gradientAlpha[d][0] += delta[0][z] * gradientTerm;
				}
				gradientBeta[z][0] += alpha[d][0] * gradientTerm;
				gradientDeltaBias[z] += gradientTerm;
//				gradientDeltaBNorm += gradientTerm * gradientTerm;
			}
		}
		
//		for (int z = minZ; z < maxZ; z++) {
//			gradientBetaNorm += gradientBeta[z][0] * gradientBeta[z][0];
//		}
		
		// gradient for lambda
        for (int d = 0; d < D; d++) {
                double gradientTerm = (alpha[d][0] - alphaMean(d)) / Math.pow(sigmaAlpha, 2); // not negated
                
                gradientLambda0 += docsC0[d] * gradientTerm;
                gradientLambda1 += docsC1[d] * gradientTerm;
                gradientLambda2 += docsC2[d] * gradientTerm;
        }
    }
	
	public void doGradientStep(int iter, int minZ, int maxZ, int minW, int maxW, int minD, int maxD) {
		/*
		 * Take a gradient step for a subset of topics and words.  Assumes gradient is current.
		 */
		// gradient ascent
		
//		double step = step;
		
		double sigmaBeta = 10.0;
		
		for (int z = minZ; z < maxZ; z++) {
			for (int c = 0; c < 1; c++) {
				gradientBeta[z][c] += -(beta[z][c]) / Math.pow(sigmaBeta, 2);
				adaBeta[z][c] += Math.pow(gradientBeta[z][c], 2);
				beta[z][c] += (step / (Math.sqrt(adaBeta[z][c])+eps)) * gradientBeta[z][c];
				gradientBeta[z][c] = 0.; // Clear gradient for the next iteration
			}
		}
		
		for (int c = 0; c < Cph; c++) {
			for (int w = minW; w < maxW; w++) {
				gradientOmega[c][w] += -(omega[c][w]) / Math.pow(sigmaOmega, 2);
				adaOmega[c][w] += Math.pow(gradientOmega[c][w], 2);
				omega[c][w] += (step / (Math.sqrt(adaOmega[c][w])+eps)) * gradientOmega[c][w];
				gradientOmega[c][w] = 0.; // Clear gradient for the next iteration
			}
		}
		
		for (int w = minW; w < maxW; w++) {
			gradientOmegaBias[w] += -(omegaBias[w]) / Math.pow(sigmaOmegaBias, 2);
			adaOmegaBias[w] += Math.pow(gradientOmegaBias[w], 2);
			omegaBias[w] += (step / (Math.sqrt(adaOmegaBias[w])+eps)) * gradientOmegaBias[w];
			gradientOmegaBias[w] = 0.; // Clear gradient for the next iteration
		}
		
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
		
		//for (int c = 0; c < 1; c++) {
		for (int c = 0; c < 1; c++) {
			for (int z = minZ; z < maxZ; z++) {
				delta[c][z] = beta[z][c];
			}
		}
		/*//for (int c = 1; c < Cth; c++) {
		for (int c = 3; c < Cth; c++) {
			for (int z = minZ; z < maxZ; z++) {
				gradientDelta[c][z] += -(delta[c][z]) / Math.pow(sigmaDelta, 2);
				adaDelta[c][z] += Math.pow(gradientDelta[c][z], 2);
				delta[c][z] += (step / (Math.sqrt(adaDelta[c][z])+eps)) * gradientDelta[c][z];
				//delta[c][z] = 0.0; // unweighted
				gradientDelta[c][z] = 0.;
			}
		}
		*/
		for (int z = minZ; z < maxZ; z++) {
			gradientDeltaBias[z] += -(deltaBias[z]) / Math.pow(sigmaDeltaBias, 2);
			adaDeltaBias[z] += Math.pow(gradientDeltaBias[z], 2);
			deltaBias[z] += (step / (Math.sqrt(adaDeltaBias[z])+eps)) * gradientDeltaBias[z];
			gradientDeltaBias[z] = 0.;
		}
		
		//gradientLambda += -(lambda) / Math.pow(sigmaLambda, 2); // do we need regularization for lambda? probably not
		adaLambda0 += Math.pow(gradientLambda0, 2);
		lambda0 += (step / (Math.sqrt(adaLambda0)+eps)) * gradientLambda0;
		gradientLambda0 = 0.;
		
		adaLambda1 += Math.pow(gradientLambda1, 2);
		lambda1 += (step / (Math.sqrt(adaLambda1)+eps)) * gradientLambda1;
		gradientLambda1 = 0.;
		
		adaLambda2 += Math.pow(gradientLambda2, 2);
		lambda2 += (step / (Math.sqrt(adaLambda2)+eps)) * gradientLambda2;
		gradientLambda2 = 0.;
	}
	
	/*
	// update lambda gradient and then do gradient step
	public void updateLambda() {
		double step = stepA;
		
		// gradient for lambda
        for (int d = 0; d < D; d++) {
                double gradientTerm = (alpha[d][0] - alphaMean(d)) / Math.pow(sigmaAlpha, 2); // not negated
                
                gradientLambda0 += docsC0[d] * gradientTerm;
                gradientLambda1 += docsC1[d] * gradientTerm;
                gradientLambda2 += docsC2[d] * gradientTerm;
        }
        
	}
	*/
	
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
			
//			updateLambda();
		}
		
		for (int z = 0; z < Z; z++) {
			System.out.println("deltaBias_"+z+" "+deltaBias[z]);
		}
		for (int z = 0; z < Z; z++) {
			System.out.print("delta_"+z);
			for (int c = 0; c < Cth; c++) {
				System.out.print(" "+delta[c][z]);
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
		for (int w = 0; w < W; w += 10000) {
			System.out.print("omega_"+w);
			for (int c = 0; c < Cph; c++) {
				System.out.print(" "+omega[c][w]);
			}
			System.out.println();
		}

		System.out.println("lambda0 = "+lambda0);
		System.out.println("lambda1 = "+lambda1);
		System.out.println("lambda2 = "+lambda2);
		
		for (int d = 0; d < D; d++) {
			thetaNorm[d] = 0.;
		}
		for (int z = 0; z < Z; z++) {
			phiNorm[z] = 0.;
		}
		
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
			
			Tup2<Double, Double> coherence = computeCoher();
			System.out.println("Coherence: " + coherence._1() + " " + coherence._2());
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
		
		/*
		// For debugging: full posterior likelihood
		
		// P(\theta | \alpha, \delta)
		double[][] theta = new double[D][Z];
		double llTheta   = 0.0;
		
		for (int d = 0; d < D; d++) {
			double pTotal = 0.0;
			for (int z = 0; z < Z; z++) {
				theta[d][z] = (nDZ[d][z] + priorDZ[d][z]) / (nD[d] + thetaNorm[d]);
				pTotal += theta[d][z];
			}
			//for (int z = 0; z < Z; z++) {
			//	theta[d][z] /= pTotal;
			//}
			
			System.out.print("Theta[" + d + "]: ");
			StringBuilder b = new StringBuilder();
			for (int z = 0; z < Z; z++) {
				System.out.print(" " + theta[d][z]);
			}
			System.out.println();
			
			System.out.print("PriorDZ[" + d + "]: ");
			b = new StringBuilder();
			for (int z = 0; z < Z; z++) {
				System.out.print(" " + priorDZ[d][z]);
			}
			System.out.println();
			
			llTheta += MathUtils.dirichletLogProb(theta[d], priorDZ[d])/Math.log(2.0);
			
			System.out.println("Theta weight: " + pTotal);
			System.out.println("logProbTheta: " + MathUtils.dirichletLogProb(theta[d], priorDZ[d])/Math.log(2.0));
		}
		System.out.println("LL theta:" + llTheta);
		
		// P(\phi | \beta, \omega)
		double[][] phi = new double[Z][W];
		double llPhi   = 0.0;
		
		for (int z = 0; z < Z; z++) {
			double pTotal = 0.0;
			for (int w = 0; w < W; w++) {
				phi[z][w] = (nZW[z][w] + priorZW[z][w]) / (nZ[z] + phiNorm[z]);
			    pTotal += phi[z][w];
			}
			//for (int w = 0; w < W; w++) {
			//	phi[z][w] /= pTotal;
			//}
			
			System.out.print("Phi[" + z + "]: ");
			StringBuilder b = new StringBuilder();
			for (int w = 0; w < W; w++) {
				System.out.print(" " + phi[z][w]);
			}
			System.out.println();
			
			System.out.print("PriorZW[" + z + "]: ");
			b = new StringBuilder();
			for (int w = 0; w < W; w++) {
				System.out.print(" " + priorZW[z][w]);
			}
			System.out.println();
			llPhi += MathUtils.dirichletLogProb(phi[z], priorZW[z])/Math.log(2.0);
			
			System.out.println("Phi weight: " + pTotal);
			System.out.println("logProbPhi: " + MathUtils.dirichletLogProb(phi[z], priorZW[z])/Math.log(2.0));
		}
		System.out.println("LL phi: " + llPhi);
		
		// P(alpha, beta, delta, omega | regularization)
		double llAlpha    = 0.0;
		for (int d = 0; d < D; d++) {
			double mean = this.alphaMean(d);
			for (int c = 0; c < Cth; c++) {
				double probAlpha = MathUtils.normalProb(alpha[d][c], mean, sigmaAlpha);
				llAlpha += MathUtils.log(probAlpha, 2.0);
				
				System.out.println("alpha[" + d + "]: " + alpha[d][c] + " alphaMean: " + mean);
				System.out.println("probAlpha[" + d + "]: " + probAlpha);
			}
		}
		System.out.println("LL alpha: " + llAlpha);
		
		double llDelta = 0.0;
		for (int z = 0; z < Z; z++) {
			for (int c = 0; c < Cph; c++) {
				llDelta += MathUtils.log(MathUtils.normalProb(delta[c][z], 0.0,
						this.sigmaDelta), 2.0);
			}
			llDelta += MathUtils.log(MathUtils.normalProb(this.deltaBias[z],
					0.0, this.sigmaDeltaBias), 2.0);
		}
		System.out.println("LL delta: " + llDelta);
		
		double llOmega = 0.0;
		for (int w = 0; w < W; w++) {
			for (int c = 0; c < Cph; c++) {
				llOmega += MathUtils.log(MathUtils.normalProb(omega[c][w], 0.0,
						this.sigmaOmega), 2.0);
			}
			llOmega += MathUtils.log(MathUtils.normalProb(this.omegaBias[w],
					0.0, this.sigmaOmegaBias), 2.0);
		}
		System.out.println("LL omega: " + llOmega);
		
		System.out.println("Perplexity denominator: " + denom);
		double posteriorLL = logProbSum + llTheta + llPhi + llAlpha + llDelta + llOmega;
		System.out.println("Posterior LL for heldout=" + isHeldOut + ": " + posteriorLL);
		*/
		
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
	
	public Tup2<Double, Double> computeCoher() {
		double coher = 0.0;
		double coherWithPrior = 0.0;
		
		int M = 20;
		for (int z = 0; z < Z; z++) {
			double coherZ = 0.0;
			
			List<Tup2<Integer, Integer>> topwords  = new ArrayList<Tup2<Integer, Integer>>();
			for (int w = 0; w < W; w++) {
				topwords.add(new Tup2<Integer, Integer>(nZW[z][w], w));
			}
			
			Collections.sort(topwords, new Comparator<Tup2<Integer, Integer>>() {
				@Override
				public int compare(Tup2<Integer, Integer> o1,
						Tup2<Integer, Integer> o2) {
					return - o1._1().compareTo(o2._1());
				}
			});
			
			System.out.println(String.format("Topwords[%d]: %s:%d %s:%d %s:%d %s:%d %s:%d",
					z, wordMapInv.get(topwords.get(0)._2()), topwords.get(0)._1(),
					wordMapInv.get(topwords.get(1)._2()), topwords.get(1)._1(),
					wordMapInv.get(topwords.get(2)._2()), topwords.get(2)._1(),
					wordMapInv.get(topwords.get(3)._2()), topwords.get(3)._1(),
					wordMapInv.get(topwords.get(4)._2()), topwords.get(4)._1()));
			
			for (int m = 1; m < M; m++) {
				int word2 = topwords.get(m)._2();
				
				for (int i = 0; i < m; i++) {
					int word1 = topwords.get(i)._2();
					
					//System.out.println(wordMapInv.get(vm)+" "+wordMapInv.get(vi)+" "+DCC[vm][vi]+" "+DC[vi]);
					coherZ += Math.log((biCounts[word2][word1] + 1.0) / uniCounts[word1]);
				}
			}
			
			int debugWord1 = topwords.get(0)._2();
			int debugWord2 = topwords.get(1)._2();
			
//			System.out.println(String.format("Counts[%d]: %s,%s:%d,%d,%d => %f",
//					z, wordMapInv.get(debugWord1), wordMapInv.get(debugWord2),
//					uniCounts[debugWord1], uniCounts[debugWord2], biCounts[debugWord2][debugWord1],
//					Math.log((biCounts[debugWord2][debugWord1] + 1.0) / uniCounts[debugWord1])));
			coher += coherZ;
//			System.out.println(String.format("Coher[%d]: %f %f", z, coherZ, coher));
			
			double coherZWithPrior = 0.0;
			
			ArrayList<Tup2<Double, Integer>> topwordsWithPrior  = new ArrayList<Tup2<Double, Integer>>();
			for (int w = 0; w < W; w++) {
				topwordsWithPrior.add(new Tup2<Double, Integer>((double)(nZW[z][w] + priorZW[z][w])/((double)nZ[z] + phiNorm[z]), w));
			}
			Collections.sort(topwordsWithPrior, new Comparator<Tup2<Double, Integer>>() {
				@Override
				public int compare(Tup2<Double, Integer> o1,
						Tup2<Double, Integer> o2) {
					return - o1._1().compareTo(o2._1());
				}
			});
			
			for (int m = 1; m < M; m++) {
				int word2 = topwordsWithPrior.get(m)._2();
				
				for (int i = 0; i < m; i++) {
					int word1 = topwordsWithPrior.get(i)._2();
					
					//System.out.println(wordMapInv.get(vm)+" "+wordMapInv.get(vi)+" "+DCC[vm][vi]+" "+DC[vi]);
					coherZWithPrior += Math.log((biCounts[word2][word1] + 1.0) / uniCounts[word1]);
				}
			}
			coherWithPrior += coherZWithPrior;
		}

		coher /= Z;
		coherWithPrior /= Z;
		
		return new Tup2<Double, Double>(coher, coherWithPrior);
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
				bw.write(" " + omega[c][w]);
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
			
			//for (int c = 0; c < Cth; c++) { 
			//	bw.write(Math.exp(alpha[d][c])+" ");
			//}
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
				bw.write(" "+delta[c][z]);
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
		bw.write(lambda0 + " " + lambda1 + " " + lambda2);
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