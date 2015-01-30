package main;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Random;
//import org.apache.commons.math.special.Gamma;
import java.util.concurrent.ArrayBlockingQueue;

import utils.MathUtils;


public class SpriteJoint extends TopicModel {
	double eps = 1.0e-6; // small epsilon to stabilize digamma
	
	public HashMap<String,Integer> wordMap;
	public HashMap<Integer,String> wordMapInv;

	private Random r;

	public String priorPrefix;

	public int[][] docs;
	public double[] docsC;
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
	public double[][] omega;
	public double[] omegaBias;
	public double[][] priorZW;
	public double[] phiNorm;
	public double[][] beta;
	public double[][] betaB;
	
	public double stepA;
	public double stepSizeADZ;
	public double stepSizeAZ;
	public double stepSizeAB;
	public double stepSizeW;
	public double stepSizeWB;
	public double stepSizeB;

	public double sigmaA;
	public double sigmaAB;
	public double sigmaW;
	public double sigmaWB;
	
	public int likelihoodFreq;

	// Pulled out to compute gradient in parallel.
	double[][] gradientOmega;
	double[] gradientOmegaBias;
	double[][] gradientBeta;
	double[][] gradientBetaB;
	
	double[][] gradientDelta;
	double[] gradientDeltaBias;
	double[][] gradientAlpha;
	
	double[][] adaOmega;
	double[] adaOmegaBias;
	double[][] adaBeta;
	double[][] adaBetaB;
	double[][] adaDelta;
	double[] adaDeltaBias;
	double[][] adaAlpha;
	
	public int seed;
	
	// Parallel nonsense
	private int numThreads;
	
	private enum ThreadComm {UPDATE, KILL, SAMPLE, CALC_GRADIENT, GRADIENT_STEP};
	
	private ArrayBlockingQueue<ThreadComm> THREAD_COMM_QUEUE = null; // Signals workers have a job to do
    private ArrayBlockingQueue<String>     THREAD_DONE_QUEUE = null; // Signals workers are done
    
	private Worker[] THREADS = null;
	
	public SpriteJoint(int z, double sigmaA0, double sigmaAB0, double sigmaW0, double sigmaWB0, double stepSizeADZ0, double stepSizeAZ0, double stepSizeAB0, double stepSizeW0, double stepSizeWB0, double stepSizeB0, double delta00, double delta10, double deltaB0, double omegaB0, int likelihoodFreq0, String prefix, double stepA0, int Cth0, int Cph0, int seed0, int numThreads0) {
		Z = z;
		
		sigmaA = sigmaA0;
		sigmaAB = sigmaAB0;
		sigmaW = sigmaW0;
		sigmaWB = sigmaWB0;
		stepSizeADZ = stepSizeADZ0;
		stepSizeAZ = stepSizeAZ0;
		stepSizeAB = stepSizeAB0;
		stepSizeW = stepSizeW0;
		stepSizeWB = stepSizeWB0;
		stepSizeB = stepSizeB0;
		deltaB = deltaB0;
		omegaB = omegaB0;
		
		stepA = stepA0;
		Cth = Cth0;
		Cph = Cph0;
		
		likelihoodFreq = likelihoodFreq0;
		priorPrefix = prefix;
		
		seed = seed0;
		
		numThreads = numThreads0;
	}
	
	public void initialize() {
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
		omega = new double[Cph][W];
		omegaBias = new double[W];
		phiNorm = new double[Z];
		priorZW = new double[Z][W];
		
		adaOmega = new double[Cph][W];
		adaOmegaBias = new double[W];
		adaBeta = new double[Z][Cph];
		adaBetaB = new double[Z][Cph];
		adaDelta = new double[Cth][Z];
		adaDeltaBias = new double[Z];
		adaAlpha = new double[D][Cth];
		
		// Initialize gradient
		gradientOmega = new double[Cph][W];
		gradientOmegaBias = new double[W];
		gradientBeta = new double[Z][Cph];
		gradientBetaB = new double[Z][Cph];
		
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
			alpha[d][0] = docsC[d];
			for (int c = 1; c < Cth; c++) {
				alpha[d][c] = -2.0;
				//alpha[d][c] += (r.nextDouble() - 0.5) / 100.0;
			}
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
		
		for (int c = 1; c < Cph; c++) {
			for (int w = 0; w < W; w++) {
				omega[c][w] = (r.nextDouble() - 0.5) / 100.0;
				//omega[c][w] += -2.0;
			}
		}
		
		for (int z = 0; z < Z; z++) {
			betaB[z][0] = 1.0;
			for (int c = 1; c < Cph; c++) {
				betaB[z][c] = 1.0 / (Cph-1);
				beta[z][c] = -2.0;
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
			docsZ[d] = new int[docs[d].length];
			docsZZ[d] = new int[docs[d].length][Z];
			
			for (int n = 0; n < docs[d].length; n++) {
				int w = docs[d][n];
				
				int z = r.nextInt(Z); // sample uniformly
				docsZ[d][n] = z;
				
				// update counts
				
				nZW[z][w] += 1;	
				nZ[z] += 1;
				nDZ[d][z] += 1;
				nD[d] += 1;
			}
		}
		
		// Spin up worker threads.  Will only work when ThreadComm message is received.
		if (numThreads > 1) {
			THREAD_COMM_QUEUE = new ArrayBlockingQueue<ThreadComm>(numThreads);
		    THREAD_DONE_QUEUE = new ArrayBlockingQueue<String>(numThreads);
			//THREAD_BITS = new Boolean[numThreads];
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
		}
	}

	// initialize bias variables by using moment-matching approx to Dirichlet params, then take the log
	public void initializeBias() {
		// first delta
		{
		double[] mean = new double[Z];
		double[] var = new double[Z];
		double[] m = new double[Z];
		double norm = 0.0;
		for (int z = 0; z < Z; z++) {
			for (int d = 0; d < D; d++) {
				double frac = (double) (nDZ[d][z] + 0.001) / (nD[d] + 0.001*Z);
				mean[z] += frac;
			}
			mean[z] /= (double) D;
			
			for (int d = 0; d < D; d++) {
				double frac = (double) (nDZ[d][z] + 0.001) / (nD[d] + 0.001*Z);
				var[z] += Math.pow(frac - mean[z], 2);
			}
			var[z] /= (double) D;
			
			m[z] = (mean[z] * (1.0 - mean[z])) / var[z];
			m[z] -= 1.0;
			
			deltaBias[z] = mean[z];
			norm += deltaBias[z];
		}
		
		for (int z = 0; z < Z; z++) {
//System.out.println("mean "+mean[c]);
//System.out.println("var "+var[c]);
//System.out.println("m "+m[c]);
//System.out.println("alphaS "+alphaS[c]);
			deltaBias[z] /= norm;
//System.out.println("alphaS "+alphaS[c]);
		}

		double logSum = 0.0;
		for (int z = 0; z < Z; z++) {
			logSum += Math.log(m[z]);
		}
//System.out.println("logSum "+logSum);
		norm = Math.exp(logSum / (Z-1));

		System.out.print("init deltaBias");
		for (int z = 0; z < Z; z++) {
			deltaBias[z] *= norm;
			System.out.print(" "+deltaBias[z]);
			deltaBias[z] = Math.log(deltaBias[z]);
		}
		System.out.println();
		}

		// next omega
		{
		double[] mean = new double[W];
		double[] var = new double[W];
		double[] m = new double[W];
		double norm = 0.0;
		for (int w = 0; w < W; w++) {
			for (int z = 0; z < Z; z++) {
				double frac = (double) (nZW[z][w] + 0.001) / (nZ[z] + 0.001*W);
				mean[w] += frac;
			}
			mean[w] /= (double) Z;

			for (int z = 0; z < Z; z++) {
				double frac = (double) (nZW[z][w] + 0.001) / (nZ[z] + 0.001*W);
				var[w] += Math.pow(frac - mean[w], 2);
			}
			var[w] /= (double) Z;

			m[w] = (mean[w] * (1.0 - mean[w])) / var[w];
			m[w] -= 1.0;

			omegaBias[w] = mean[w];
			norm += omegaBias[w];
		}
		
		for (int w = 0; w < W; w++) {
//System.out.println("mean "+mean[c]);
//System.out.println("var "+var[c]);
//System.out.println("m "+m[c]);
//System.out.println("alphaS "+alphaS[c]);
			omegaBias[w] /= norm;
//System.out.println("alphaS "+alphaS[c]);
		}

		double logSum = 0.0;
		for (int w = 0; w < W; w++) {
			logSum += Math.log(m[w]);
		}
//System.out.println("logSum "+logSum);
		norm = Math.exp(logSum / (W-1));

		System.out.print("init omegaBias");
		for (int w = 0; w < W; w++) {
			omegaBias[w] *= norm;
			if (w % 1000 == 0) System.out.print(" "+omegaBias[w]);
			omegaBias[w] = Math.log(omegaBias[w]);
		}
		System.out.println();
		}
	}
	
	// returns the delta_dz prior given all the parameters
	public double priorA(int d, int z) {
		double weight = deltaBias[z];

		for (int c = 0; c < 1; c++) {
			weight += alpha[d][c] * delta[c][z];
		}
		for (int c = 1; c < Cth; c++) {
			//weight += alpha[d][c] * betaB[z][c] * delta[c][z];
			weight += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]);
		}
		
		return Math.exp(weight);
	}
	
	// returns the omega_zw prior given all the parameters
	public double priorW(int z, int w) {
		double weight = omegaBias[w];
		
		for (int c = 0; c < 1; c++) {
			weight += beta[z][c] * omega[c][w];
		}
		for (int c = 1; c < Cph; c++) {
			//weight += betaB[z][c] * beta[z][c] * omega[c][w];
			weight += betaB[z][c] * Math.exp(beta[z][c]) * omega[c][w];
			//weight += betaB[z][c] * Math.exp(beta[z][c]) * Math.exp(omega[c][w]);
		}
		
		return Math.exp(weight);
	}
	
	public void clearGradient() {
		for (int c = 0; c < Cph; c++) {
			for (int z = 0; z < Z; z++) {
				gradientBeta[z][c] = 0.;
				gradientBetaB[z][c] = 0.;
			}
			for (int w = 0; w < W; w++) {
				gradientOmega[c][w] = 0.;
			}
		}
		
		for (int w = 0; w < W; w++) {
			gradientOmegaBias[w] = 0.;
		}
		
		for (int c = 0; c < Cth; c++) {
			for (int z = 0; z < Z; z++) {
				gradientDelta[c][z] = 0.;
			}
			for (int d = 0; d < D; d++) {
				gradientAlpha[d][c] = 0.;
			}
		}
		
		for (int z = 0; z < Z; z++) {
			gradientDeltaBias[z] = 0.;
		}
	}
	
	/*
	 * Updates the gradients for a subset of topics and words.
	 */
	public void updateGradient(int iter, int minZ, int maxZ, int minW, int maxW) {
		// compute gradients
		
		for (int z = minZ; z < maxZ; z++) {
			for (int w = minW; w < maxW; w++) {
				double dg1  = MathUtils.digamma0(phiNorm[z] + eps);
				double dg2  = MathUtils.digamma0(phiNorm[z] + nZ[z] + eps);
				double dgW1 = MathUtils.digamma0(priorZW[z][w] + nZW[z][w] + eps);
				double dgW2 = MathUtils.digamma0(priorZW[z][w] + eps);
				
				double gradientTerm = priorZW[z][w] * (dg1-dg2+dgW1-dgW2);
				
				for (int c = 0; c < 1; c++) { // factor
					gradientBeta[z][c] += omega[c][w] * gradientTerm;
					gradientOmega[c][w] += beta[z][c] * gradientTerm;
				}
				for (int c = 1; c < Cph; c++) { // hierarchy
					gradientBeta[z][c]  += betaB[z][c] * Math.exp(beta[z][c]) * omega[c][w] * gradientTerm;
					gradientBetaB[z][c] += Math.exp(beta[z][c]) * omega[c][w] * gradientTerm;
					gradientOmega[c][w] += betaB[z][c] * Math.exp(beta[z][c]) * gradientTerm;
				}
				gradientOmegaBias[w] += gradientTerm;
			}
		}
		
		for (int z = minZ; z < maxZ; z++) {
			for (int d = 0; d < D; d++) {
				double dg1  = MathUtils.digamma0(thetaNorm[z] + eps);
				double dg2  = MathUtils.digamma0(thetaNorm[z] + nD[d] + eps);
				double dgW1 = MathUtils.digamma0(priorDZ[d][z] + nDZ[d][z] + eps);
				double dgW2 = MathUtils.digamma0(priorDZ[d][z] + eps);
				
				double gradientTerm = priorDZ[d][z] * (dg1-dg2+dgW1-dgW2);

				for (int c = 0; c < 1; c++) { // factor
					gradientBeta[z][c] += alpha[d][c] * gradientTerm;
				}
				for (int c = 1; c < Cth; c++) { // hierarchy
					gradientAlpha[d][c] += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					gradientDelta[c][z] += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					gradientBetaB[z][c] += Math.exp(alpha[d][c]) * Math.exp(delta[c][z]) * gradientTerm;
				}
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
		double sigmaBeta = 10.0;
		double sigmaOmega = 10.0;
		double sigmaOmegaBias = 10.0;
		double sigmaAlpha = 10.0;
		double sigmaDelta = 10.0;
		double sigmaDeltaBias = 10.0;
		
		for (int z = minZ; z < maxZ; z++) {
			for (int c = 0; c < 1; c++) {
				gradientBeta[z][c] += -(beta[z][c]) / Math.pow(sigma0, 2);
				adaBeta[z][c] += Math.pow(gradientBeta[z][c], 2);
				beta[z][c] += (step / (Math.sqrt(adaBeta[z][c])+eps)) * gradientBeta[z][c];
				gradientBeta[z][c] = 0.;
			}
			for (int c = 1; c < Cph; c++) {
				gradientBeta[z][c] += -(beta[z][c]) / Math.pow(sigmaBeta, 2);
				adaBeta[z][c] += Math.pow(gradientBeta[z][c], 2);
				beta[z][c] += (step / (Math.sqrt(adaBeta[z][c])+eps)) * gradientBeta[z][c];
				//beta[z][c] = 0.0; // unweighted
				gradientBeta[z][c] = 0.;
			}
		}
		
		double dirp = 0.01; // Dirichlet hyperparameter
		double rho = 0.95; // AdaDelta weighting
		//double priorTemp = 0.0; // no prior
		double priorTemp = Math.pow(1.00, iter-199);
		System.out.println("priorTemp = "+priorTemp);
		double[][] prevBetaB = new double[Z][Cph];
		for (int z = minZ; z < maxZ; z++) {
			double norm = 0.0;
			for (int c = 1; c < Cph; c++) {
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
			for (int c = 0; c < 1; c++) {
				betaB[z][c] = 1.0;
			}
			//System.out.println("normalizer "+z+" "+norm);
			for (int c = 1; c < Cph; c++) {
				if (norm == 0.0 || norm == Double.POSITIVE_INFINITY) {
					System.out.println("BAD betaB");
					//betaB[z][c] = 1.0 / (Cph - 1);
					betaB[z][c] = prevBetaB[z][c]; // undo update if not well defined
				}
				else {
					betaB[z][c] /= norm;
				}
			}
		}
		
		for (int c = 0; c < Cph; c++) {
			for (int w = minW; w < maxW; w++) {
				gradientOmega[c][w] += -(omega[c][w]) / Math.pow(sigmaOmega, 2);
				adaOmega[c][w] += Math.pow(gradientOmega[c][w], 2);
				omega[c][w] += (step / (Math.sqrt(adaOmega[c][w])+eps)) * gradientOmega[c][w];
				gradientOmega[c][w] = 0.;
			}
		}
		
		for (int w = minW; w < maxW; w++) {
			gradientOmegaBias[w] += -(omegaBias[w]) / Math.pow(sigmaOmegaBias, 2);
			adaOmegaBias[w] += Math.pow(gradientOmegaBias[w], 2);
			omegaBias[w] += (step / (Math.sqrt(adaOmegaBias[w])+eps)) * gradientOmegaBias[w];
			gradientOmegaBias[w] = 0.;
		}
		
		for (int d = minD; d < maxD; d++) {
			for (int c = 1; c < Cth; c++) {
				gradientAlpha[d][c] += -(alpha[d][c]) / Math.pow(sigmaAlpha, 2);
				adaAlpha[d][c] += Math.pow(gradientAlpha[d][c], 2);
				alpha[d][c] += (step / (Math.sqrt(adaAlpha[d][c])+eps)) * gradientAlpha[d][c];
				//if (alpha[d][c] < 1.0e-6) alpha[d][c] = 1.0e-6;
				gradientAlpha[d][c] = 0.;
			}
		}
		
		for (int c = 0; c < 1; c++) {
			for (int z = minZ; z < maxZ; z++) {
				delta[c][z] = beta[z][c];
			}
		}
		for (int c = 1; c < Cth; c++) {
			for (int z = minZ; z < maxZ; z++) {
				gradientDelta[c][z] += -(delta[c][z]) / Math.pow(sigmaDelta, 2);
				adaDelta[c][z] += Math.pow(gradientDelta[c][z], 2);
				delta[c][z] += (step / (Math.sqrt(adaDelta[c][z])+eps)) * gradientDelta[c][z];
				//delta[c][z] = 0.0; // unweighted
				gradientDelta[c][z] = 0.;
			}
		}
		for (int z = minZ; z < maxZ; z++) {
			gradientDeltaBias[z] += -(deltaBias[z]) / Math.pow(sigmaDeltaBias, 2);
			adaDeltaBias[z] += Math.pow(gradientDeltaBias[z], 2);
			deltaBias[z] += (step / (Math.sqrt(adaDeltaBias[z])+eps)) * gradientDeltaBias[z];
			gradientDeltaBias[z] = 0.;
		}
	}
	
	// gradient-based parameter update
	public void updateWeights(int iter) {
		// compute gradients
		
		double[][] gradientOmega = new double[Cph][W];
		double[] gradientOmegaBias = new double[W];
		double[][] gradientBeta = new double[Z][Cph];
		double[][] gradientBetaB = new double[Z][Cph];
		
		for (int z = 0; z < Z; z++) {
			for (int w = 0; w < W; w++) {
				double dg1  = MathUtils.digamma0(phiNorm[z] + eps);
				double dg2  = MathUtils.digamma0(phiNorm[z] + nZ[z] + eps);
				double dgW1 = MathUtils.digamma0(priorZW[z][w] + nZW[z][w] + eps);
				double dgW2 = MathUtils.digamma0(priorZW[z][w] + eps);
				
				double gradientTerm = priorZW[z][w] * (dg1-dg2+dgW1-dgW2);
				
				for (int c = 0; c < 1; c++) { // factor
					gradientBeta[z][c] += omega[c][w] * gradientTerm;
					gradientOmega[c][w] += beta[z][c] * gradientTerm;
				}
				for (int c = 1; c < Cph; c++) { // hierarchy
					gradientBeta[z][c]  += betaB[z][c] * Math.exp(beta[z][c]) * omega[c][w] * gradientTerm;
					gradientBetaB[z][c] += Math.exp(beta[z][c]) * omega[c][w] * gradientTerm;
					gradientOmega[c][w] += betaB[z][c] * Math.exp(beta[z][c]) * gradientTerm;
				}
				gradientOmegaBias[w] += gradientTerm;
			}
		}
		
		double[][] gradientDelta   = new double[Cth][Z];
		double[] gradientDeltaBias = new double[Z];
		double[][] gradientAlpha   = new double[D][Cth];
		
		for (int z = 0; z < Z; z++) {
			for (int d = 0; d < D; d++) {
				double dg1  = MathUtils.digamma0(thetaNorm[z] + eps);
				double dg2  = MathUtils.digamma0(thetaNorm[z] + nD[d] + eps);
				double dgW1 = MathUtils.digamma0(priorDZ[d][z] + nDZ[d][z] + eps);
				double dgW2 = MathUtils.digamma0(priorDZ[d][z] + eps);
				
				double gradientTerm = priorDZ[d][z] * (dg1-dg2+dgW1-dgW2);

				for (int c = 0; c < 1; c++) { // factor
					gradientBeta[z][c] += alpha[d][c] * gradientTerm;
				}
				for (int c = 1; c < Cth; c++) { // hierarchy
					gradientAlpha[d][c] += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					gradientDelta[c][z] += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					gradientBetaB[z][c] += Math.exp(alpha[d][c]) * Math.exp(delta[c][z]) * gradientTerm;
				}
				gradientDeltaBias[z] += gradientTerm;
			}
		}
		
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
		double sigmaBeta = 10.0;
		double sigmaOmega = 10.0;
		double sigmaOmegaBias = 10.0;
		double sigmaAlpha = 10.0;
		double sigmaDelta = 10.0;
		double sigmaDeltaBias = 10.0;
		
		for (int z = 0; z < Z; z++) {
			for (int c = 0; c < 1; c++) {
				gradientBeta[z][c] += -(beta[z][c]) / Math.pow(sigma0, 2);
				adaBeta[z][c] += Math.pow(gradientBeta[z][c], 2);
				beta[z][c] += (step / (Math.sqrt(adaBeta[z][c])+eps)) * gradientBeta[z][c];
			}
			for (int c = 1; c < Cph; c++) {
				gradientBeta[z][c] += -(beta[z][c]) / Math.pow(sigmaBeta, 2);
				adaBeta[z][c] += Math.pow(gradientBeta[z][c], 2);
				beta[z][c] += (step / (Math.sqrt(adaBeta[z][c])+eps)) * gradientBeta[z][c];
				//beta[z][c] = 0.0; // unweighted
			}
		}
		
		double dirp = 0.01; // Dirichlet hyperparameter
		double rho = 0.95; // AdaDelta weighting
		//double priorTemp = 0.0; // no prior
		double priorTemp = Math.pow(1.00, iter-199);
		System.out.println("priorTemp = "+priorTemp);
		double[][] prevBetaB = new double[Z][Cph];
		for (int z = 0; z < Z; z++) {
			double norm = 0.0;
			for (int c = 1; c < Cph; c++) {
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
			}
			for (int c = 0; c < 1; c++) {
				betaB[z][c] = 1.0;
			}
			//System.out.println("normalizer "+z+" "+norm);
			for (int c = 1; c < Cph; c++) {
				if (norm == 0.0 || norm == Double.POSITIVE_INFINITY) {
					System.out.println("BAD betaB");
					//betaB[z][c] = 1.0 / (Cph - 1);
					betaB[z][c] = prevBetaB[z][c]; // undo update if not well defined
				}
				else {
					betaB[z][c] /= norm;
				}
			}
		}
		
		for (int c = 0; c < Cph; c++) {
			for (int w = 0; w < W; w++) {
				gradientOmega[c][w] += -(omega[c][w]) / Math.pow(sigmaOmega, 2);
				adaOmega[c][w] += Math.pow(gradientOmega[c][w], 2);
				omega[c][w] += (step / (Math.sqrt(adaOmega[c][w])+eps)) * gradientOmega[c][w];
			}
		}
		
		for (int w = 0; w < W; w++) {
			gradientOmegaBias[w] += -(omegaBias[w]) / Math.pow(sigmaOmegaBias, 2);
			adaOmegaBias[w] += Math.pow(gradientOmegaBias[w], 2);
			omegaBias[w] += (step / (Math.sqrt(adaOmegaBias[w])+eps)) * gradientOmegaBias[w];
		}
		
		for (int d = 0; d < D; d++) {
			for (int c = 1; c < Cth; c++) {
				gradientAlpha[d][c] += -(alpha[d][c]) / Math.pow(sigmaAlpha, 2);
				adaAlpha[d][c] += Math.pow(gradientAlpha[d][c], 2);
				alpha[d][c] += (step / (Math.sqrt(adaAlpha[d][c])+eps)) * gradientAlpha[d][c];
				//if (alpha[d][c] < 1.0e-6) alpha[d][c] = 1.0e-6;
			}
		}
		
		for (int c = 0; c < 1; c++) {
			for (int z = 0; z < Z; z++) {
				delta[c][z] = beta[z][c];
			}
		}
		for (int c = 1; c < Cth; c++) {
			for (int z = 0; z < Z; z++) {
				gradientDelta[c][z] += -(delta[c][z]) / Math.pow(sigmaDelta, 2);
				adaDelta[c][z] += Math.pow(gradientDelta[c][z], 2);
				delta[c][z] += (step / (Math.sqrt(adaDelta[c][z])+eps)) * gradientDelta[c][z];
				//delta[c][z] = 0.0; // unweighted
			}
		}
		for (int z = 0; z < Z; z++) {
			gradientDeltaBias[z] += -(deltaBias[z]) / Math.pow(sigmaDeltaBias, 2);
			adaDeltaBias[z] += Math.pow(gradientDeltaBias[z], 2);
			deltaBias[z] += (step / (Math.sqrt(adaDeltaBias[z])+eps)) * gradientDeltaBias[z];
		}
	}
	
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
						updateGradient(iter, minZ, maxZ, minW, maxW);
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
		double[] thetaNorm = new double[D];
		double[] phiNorm   = new double[Z];
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
	public void doSampling(int iter) {
		long startTime = System.currentTimeMillis();
		
		// sample z values for all the tokens
		if (numThreads <= 1) {
			sampleBatch(0, D);
		}
		else {
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
		}
		
		//if (iter == 200) initializeBias();
		if (iter >= 200) {
			if (numThreads <= 1) {
				updateWeights(iter);
			}
			else {
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
			}
		}
		
		/*if (iter % 500 == 0) { // multiple GD iterations
			for (int i = 0; i < 1000; i++) {
				updateWeights(iter+i);
				
				thetaNorm = new double[D];
				phiNorm = new double[Z];
				for (int z = 0; z < Z; z++) {
					for (int d = 0; d < D; d++) {
						priorDZ[d][z] = priorA(d, z);
						thetaNorm[d] += priorDZ[d][z];
					}
					for (int w = 0; w < W; w++) {
						priorZW[z][w] = priorW(z, w);
						phiNorm[z] += priorZW[z][w];
					}
				}
			}
		}*/

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
		for (int z = 0; z < Z; z++) {
			System.out.print("betaB_"+z);
			for (int c = 0; c < Cph; c++) {
				System.out.print(" "+betaB[z][c]);
			}
			System.out.println();
		}
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

		// compute the priors with the new params and update the cached prior variables 
		// Single-threaded
		if (numThreads <= 1) {
			updatePriors(0, Z);
		}
		else {
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
			System.out.println("Log-likelihood: "+computeLL());
		}

		// collect samples (docsZZ) 
		if (burnedIn) {
			for (int d = 0; d < D; d++) {
				for (int n = 0; n < docs[d].length; n++) { 
					int x = docsZ[d][n];
					
					docsZZ[d][n][x] += 1;
				}
			}
		}
		long endTime = System.currentTimeMillis();
		
		System.out.println(String.format("Iteration time:\t%d\t%d", iter, endTime - startTime));
	}
    
	public void sampleBatch(int minD, int maxD) {
		for (int d = minD; d < maxD; d++) {
			for (int n = 0; n < docs[d].length; n++) {
				//System.out.println("Sampling " + d + " " + n);
				sample(d, n);
			}
			if (d % 10000 == 0) {
				System.out.println("Done document " + d);
			}
		}
	}
	
	public void sample(int d, int n) {
		int w = docs[d][n];
		int topic = docsZ[d][n];
		
		// decrement counts
		
		synchronized((Integer)topic) {
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
		
		synchronized((Integer)topic) {
			nZW[topic][w] += 1;	
			nZ[topic] += 1;
			nDZ[d][topic] += 1;
		}
		
		// set new assignments
		docsZ[d][n] = topic;
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

		docs = new int[D][];
		docsC = new double[D];
		
		fr.close();
		
		fr = new FileReader(filename);
		br = new BufferedReader(fr); 

		int d = 0;
		while ((s = br.readLine()) != null) {
			String[] tokens = s.split("\\s+");
			
			int N = tokens.length;
			
			docs[d] = new int[N-2];
			docsC[d] = Double.parseDouble(tokens[1]); 

			for (int n = 2; n < N; n++) {
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
		
		System.out.println(D+" documents");
		System.out.println(W+" word types");
	}
	
	public void writeOutput(String filename) throws Exception {
		File outputDir = new File(new File(filename).getParent(), String.format("output_%d_%d_%.2f", Z, Cph, deltaB));
		outputDir.mkdir();
		String baseName = new File(filename).getName();
		
		FileWriter fw = new FileWriter(new File(outputDir, baseName + ".assign"));
		BufferedWriter bw = new BufferedWriter(fw);
		
		for (int d = 0; d < D; d++) {
			bw.write(docsC[d]+" ");
			
			for (int n = 0; n < docs[d].length; n++) {
				String word = wordMapInv.get(docs[d][n]);

				//bw.write(word+":"+docsZ[d][n]+" "); // only current sample
				bw.write(word);  // for multiple samples
				for (int zz = 0; zz < Z; zz++) {
					bw.write(":"+docsZZ[d][n][zz]);
				}
				bw.write(" ");
			}
			bw.newLine();
		}
		
		bw.close();
		fw.close();

		fw = new FileWriter(new File(outputDir, baseName+".beta"));
		bw = new BufferedWriter(fw);
		for (int z = 0; z < Z; z++) { 
			for (int c = 0; c < 1; c++) {
				bw.write(beta[z][c]+" ");
			}
			for (int c = 1; c < Cph; c++) {
				//bw.write(beta[z][c]+" ");
				bw.write(Math.exp(beta[z][c])+" ");
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
				bw.write(" "+omega[c][w]);
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
			for (int c = 0; c < 1; c++) { 
				bw.write(alpha[d][c]+" ");
			}
			for (int c = 1; c < Cth; c++) { 
				//bw.write(alpha[d][c]+" ");
				bw.write(Math.exp(alpha[d][c])+" ");
			}
			bw.newLine();
		}
		bw.close();
		fw.close();

		fw = new FileWriter(new File(outputDir, baseName+".delta"));
		bw = new BufferedWriter(fw);
		for (int z = 0; z < Z; z++) {
			bw.write(""+z);

			for (int c = 0; c < 1; c++) { 
				bw.write(" "+delta[c][z]);
			}
			for (int c = 1; c < Cth; c++) { 
				//bw.write(" "+delta[c][z]);
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
	}
	
	public void cleanUp() {
		if (numThreads > 1) {
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
	}

	@Override
	public void logIteration() { }

	@Override
	public void collectSamples() { }

	@Override
	public double computeLL(int[][][] corpus) {
		// TODO Auto-generated method stub
		return 0;
	}
	
}

