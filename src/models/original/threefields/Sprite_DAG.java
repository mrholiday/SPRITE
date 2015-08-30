package models.original.threefields;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Random;
//import org.apache.commons.math.special.Gamma;

import main.TopicModel;


public class Sprite_DAG extends TopicModel {
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

	double[][] adaOmega;
	double[] adaOmegaBias;
	double[][] adaBeta;
	double[][] adaBetaB;
	double[][] adaDelta;
	double[] adaDeltaBias;
	double[][] adaAlpha;
	double[][] deltaBetaB;
	
	public Sprite_DAG(int z, double sigmaA0, double sigmaAB0, double sigmaW0, double sigmaWB0, double stepSizeADZ0, double stepSizeAZ0, double stepSizeAB0, double stepSizeW0, double stepSizeWB0, double stepSizeB0, double delta00, double delta10, double deltaB0, double omegaB0, int likelihoodFreq0, String prefix, double stepA0, int Cth0, int Cph0) {
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
	}
	
	public void initTrain() {
		System.out.println("Initializing...");
		r = new Random();

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
		deltaBetaB = new double[Z][Cph];

		docsZ = new int[D][];
		docsZZ = new int[D][][];

		nDZ = new int[D][Z];
		nD = new int[D];
		nZW = new int[Z][W];
		nZ = new int[Z];

		for (int d = 0; d < D; d++) {
			alpha[d][0] = 0.0;
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
				betaB[z][c] = 1.0; // / (Cph-1);
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
			
			for (int n = 0; n < docs[d].length; n += 1) {
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
	}

	// initialize bias varaibles by using moment-matching approx to Dirichlet params, then take the log
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

	public double logistic(double x) {
		return 1.0 / (1.0 + Math.exp(-1.0*x));
	}

	// derivative of logistic function
	public double dlogistic(double x) {
		return logistic(x) * (1.0 - logistic(x));
	}

	// gradient-based parameter update
	public void updateWeights(int iter) {
		double eps = 1.0e-6; // small epsilon to stabize digamma
		
		// compute gradients
		
		double[][] gradientOmega = new double[Cph][W];
		double[] gradientOmegaBias = new double[W];
		double[][] gradientBeta = new double[Z][Cph];
		double[][] gradientBetaB = new double[Z][Cph];

		for (int z = 0; z < Z; z++) {
			for (int w = 0; w < W; w++) {
				double dg1 = digamma0(phiNorm[z] + eps);
				double dg2 = digamma0(phiNorm[z] + nZ[z] + eps);
				double dgW1 = digamma0(priorZW[z][w] + nZW[z][w] + eps);
				double dgW2 = digamma0(priorZW[z][w] + eps);
				
				double gradientTerm = priorZW[z][w] * (dg1-dg2+dgW1-dgW2);

/*if (iter == 102) {
System.out.println("z"+z+" w"+w);
System.out.println(nZW[z][w]+" "+nZ[z]+" "+priorZW[z][w]+" "+phiNorm[z]);
System.out.println(dg1+" "+dg2+" "+dgW1+" "+dgW2);
System.out.println("term "+gradientTerm);
}*/

				for (int c = 0; c < 1; c++) { // factor
					gradientBeta[z][c] += omega[c][w] * gradientTerm;
//if (iter == 102) System.out.println("UW gradientBeta"+z+","+c+" "+gradientBeta[z][c]);
					gradientOmega[c][w] += beta[z][c] * gradientTerm;
//if (iter == 102) System.out.println("UW gradientOmega"+c+","+w+" "+gradientOmega[c][w]);
				}
				for (int c = 1; c < Cth; c++) { // hierarchy
					//gradientBeta[z][c] += betaB[z][c] * omega[c][w] * gradientTerm;
					//gradientBetaB[z][c] += beta[z][c] * omega[c][w] * gradientTerm;
					//gradientOmega[c][w] += betaB[z][c] * beta[z][c] * gradientTerm;
					gradientBeta[z][c] += betaB[z][c] * Math.exp(beta[z][c]) * omega[c][w] * gradientTerm;
					gradientBetaB[z][c] += Math.exp(beta[z][c]) * omega[c][w] * gradientTerm;
					gradientOmega[c][w] += betaB[z][c] * Math.exp(beta[z][c]) * gradientTerm;
					//gradientBeta[z][c] += betaB[z][c] * Math.exp(beta[z][c]) * Math.exp(omega[c][w]) * gradientTerm;
					//gradientBetaB[z][c] += Math.exp(beta[z][c]) * Math.exp(omega[c][w]) * gradientTerm;
					//gradientOmega[c][w] += Math.exp(omega[c][w]) * betaB[z][c] * Math.exp(beta[z][c]) * gradientTerm;
				}
				gradientOmegaBias[w] += gradientTerm;
			}
		}

		double[][] gradientDelta = new double[Cth][Z];
		double[] gradientDeltaBias = new double[Z];
		double[][] gradientAlpha = new double[D][Cth];

		for (int d = 0; d < D; d++) {
			for (int z = 0; z < Z; z++) {
				double dg1 = digamma0(thetaNorm[d] + eps);
				double dg2 = digamma0(thetaNorm[d] + nD[d] + eps);
				double dgW1 = digamma0(priorDZ[d][z] + nDZ[d][z] + eps);
				double dgW2 = digamma0(priorDZ[d][z] + eps);
				
				double gradientTerm = priorDZ[d][z] * (dg1-dg2+dgW1-dgW2);

/*if (iter == 102) {
System.out.println("d"+d+" z"+z);
System.out.println(dg1+" "+dg2+" "+dgW1+" "+dgW2);
System.out.println("term "+gradientTerm);
}*/
				for (int c = 0; c < 1; c++) { // factor
					gradientBeta[z][c] += alpha[d][c] * gradientTerm;
				}
				for (int c = 1; c < Cth; c++) { // hierarchy
					// alpha is unconstrained
					//gradientAlpha[d][c] += betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					//gradientDelta[c][z] += alpha[d][c] * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					////gradientBeta[z][c] += alpha[d][c] * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					//gradientBetaB[z][c] += alpha[d][c] * Math.exp(delta[c][z]) * gradientTerm;
					// alpha is exp'ed
					gradientAlpha[d][c] += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					gradientDelta[c][z] += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
					//gradientBeta[z][c] += Math.exp(alpha[d][c]) * betaB[z][c] * Math.exp(delta[c][z]) * gradientTerm;
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
		//double sigmaOmega = 0.5; //10.0;
		double sigmaOmega = 10.0;
		double sigmaOmegaBias = 10.0;
		double sigmaAlpha = 10.0;
		double sigmaDelta = 10.0;
		double sigmaDeltaBias = 10.0;

		double stepBeta = step * 1.0;
		for (int z = 0; z < Z; z++) {
			for (int c = 0; c < 1; c++) {
				gradientBeta[z][c] += -(beta[z][c]) / Math.pow(sigma0, 2);
				adaBeta[z][c] += Math.pow(gradientBeta[z][c], 2);
				beta[z][c] += (stepBeta / (Math.sqrt(adaBeta[z][c])+eps)) * gradientBeta[z][c];
//if (iter == 102) System.out.println("gradientBeta"+z+","+c+" "+gradientBeta[z][c]);
			}
			for (int c = 1; c < Cph; c++) {
				gradientBeta[z][c] += -(beta[z][c]) / Math.pow(sigmaBeta, 2);
				adaBeta[z][c] += Math.pow(gradientBeta[z][c], 2);
				beta[z][c] += (stepBeta / (Math.sqrt(adaBeta[z][c])+eps)) * gradientBeta[z][c];
//if (iter == 102) System.out.println("gradientBeta"+z+","+c+" "+gradientBeta[z][c]);
				//beta[z][c] = 0.0; // unweighted
			}
		}

		double dirp = 0.01; // Dirichlet hyperparameter
		double rho = 0.95; // AdaDelta weighting
		double priorTemp = 0;
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

				//adaBetaB[z][c] = (rho * adaBetaB[z][c]) + ((1.0-rho) * Math.pow(gradientBetaB[z][c], 2));
				//double update = Math.sqrt(deltaBetaB[z][c] + eps) / Math.sqrt(adaBetaB[z][c] + eps) * gradientBetaB[z][c];
				//betaB[z][c] *= Math.exp(step * update);
				//deltaBetaB[z][c] = (rho * deltaBetaB[z][c]) + ((1.0-rho) * Math.pow(update, 2));

				norm += betaB[z][c];
//if (iter == 102) System.out.println("gradientBetaB"+z+","+c+" "+gradientBetaB[z][c]);
//if (iter == 102) System.out.println("unnormalized betaB"+z+","+c+" "+betaB[z][c]);
			}
			for (int c = 0; c < 1; c++) {
				betaB[z][c] = 1.0;
			}
//if (iter == 102) System.out.println("normalizer "+norm);
//System.out.println("normalizer "+z+" "+norm);
			for (int c = 1; c < Cph; c++) {
				/*if (norm == 0.0 || norm == Double.POSITIVE_INFINITY) {
					System.out.println("BAD betaB");
					//betaB[z][c] = 1.0 / (Cph - 1);
					betaB[z][c] = prevBetaB[z][c]; // undo update if not well defined
				}
				else {
					betaB[z][c] /= norm;
				}
				//betaB[z][c] /= norm;*/
				betaB[z][c] = 1.0; // / (Cph-1); // fix to uniform
			}
		}
		
		for (int c = 0; c < Cph; c++) {
			for (int w = 0; w < W; w++) {
				gradientOmega[c][w] += -(omega[c][w]) / Math.pow(sigmaOmega, 2);
				adaOmega[c][w] += Math.pow(gradientOmega[c][w], 2);
				omega[c][w] += (step / (Math.sqrt(adaOmega[c][w])+eps)) * gradientOmega[c][w];
//if (iter == 102) System.out.println("gradientOmega"+c+","+w+" "+gradientOmega[c][w]);
			}
		}

		for (int w = 0; w < W; w++) {
			gradientOmegaBias[w] += -(omegaBias[w]) / Math.pow(sigmaOmegaBias, 2);
			adaOmegaBias[w] += Math.pow(gradientOmegaBias[w], 2);
			omegaBias[w] += (step / (Math.sqrt(adaOmegaBias[w])+eps)) * gradientOmegaBias[w];
//if (iter == 102) System.out.println("gradientOmegaBias"+w+" "+gradientOmegaBias[w]);
		}
		
		//stepSize = (0.2*Cph) / (100.0 + iter);
		for (int d = 0; d < D; d++) {
			for (int c = 1; c < Cth; c++) {
				gradientAlpha[d][c] += -(alpha[d][c]) / Math.pow(sigmaAlpha, 2);
				adaAlpha[d][c] += Math.pow(gradientAlpha[d][c], 2);
				alpha[d][c] += (step / (Math.sqrt(adaAlpha[d][c])+eps)) * gradientAlpha[d][c];
				//if (alpha[d][c] < 1.0e-6) alpha[d][c] = 1.0e-6;
//if (iter == 102) System.out.println("gradientAlpha"+d+","+c+" "+gradientAlpha[d][c]);
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
//if (iter == 102) System.out.println("gradientDelta"+c+","+z+" "+gradientDelta[c][z]);
				//delta[c][z] = 0.0; // unweighted
			}
		}
		for (int z = 0; z < Z; z++) {
			gradientDeltaBias[z] += -(deltaBias[z]) / Math.pow(sigmaDeltaBias, 2);
			adaDeltaBias[z] += Math.pow(gradientDeltaBias[z], 2);
			deltaBias[z] += (step / (Math.sqrt(adaDeltaBias[z])+eps)) * gradientDeltaBias[z];
//if (iter == 102) System.out.println("gradientDeltaBias"+z+" "+gradientDeltaBias[z]);
		}

	}


	// the E and M steps, for one iteration
	public void doSamplingIteration(int iter) {
		// sample z values for all the tokens
		for (int d = 0; d < D; d++) {
			for (int n = 0; n < docs[d].length; n += 1) {
				sample(d, n);
			}
		}

		//if (iter == 200) initializeBias();
		if (iter >= 200) updateWeights(iter);

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
		for (int w = 0; w < W; w += 1000) {
			System.out.println("omegaBias_"+w+" "+omegaBias[w]);
		}
		for (int w = 0; w < W; w += 1000) {
			System.out.print("omega_"+w);
			for (int c = 0; c < Cph; c++) {
				System.out.print(" "+omega[c][w]);
			}
			System.out.println();
		}

		// compute the priors with the new params and update the cached prior variables 
		thetaNorm = new double[D];
		phiNorm = new double[Z];
		for (int z = 0; z < Z; z++) {
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

		for (int d = 0; d < D; d++) {
			if (d % 100 == 0) {
				System.out.print("alpha_"+d);
				for (int c = 0; c < Cth; c++) {
					System.out.print(" "+alpha[d][c]);
				}
				System.out.println();
			}
		}
		for (int d = 0; d < D; d++) {
			if (d % 100 == 0) System.out.println("thetaNorm"+d+" "+thetaNorm[d]);
		}

		//if (!burnedIn) {
			System.out.println("Current perplexity: "+computeLL());
		//}

		// collect samples (docsZZ) 
		if (burnedIn) {
			for (int d = 0; d < D; d++) {
				for (int n = 0; n < docs[d].length; n++) { 
					int x = docsZ[d][n];
	
					docsZZ[d][n][x] += 1;
				}
			}
			System.out.println("Perplexity: "+computeLL());
		}

	}

	public void sample(int d, int n) {
		int w = docs[d][n];
		int topic = docsZ[d][n];
		
		// decrement counts

		nZW[topic][w] -= 1;	
		nZ[topic] -= 1;
		nDZ[d][topic] -= 1;
	
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

		nZW[topic][w] += 1;	
		nZ[topic] += 1;
		nDZ[d][topic] += 1;
		
		// set new assignments

		docsZ[d][n] = topic;
	}

	// computes the log-likelihood of the corpus
	// this marginalizes over the hidden variables but not the parameters
	// (i.e. we condition on the current estimates of \theta and \phi)
	public double computeLL() {
		double LL = 0;
		int N = 0;

		for (int d = 0; d < D; d++) {
			for (int n = 1; n < docs[d].length; n += 1) { 
				int w = docs[d][n];

				double tokenLL = 0;

				// marginalize over z

				for (int z = 0; z < Z; z++) {
					tokenLL += (nDZ[d][z] + priorDZ[d][z]) / (nD[d] + thetaNorm[d])*
				  		(nZW[z][w] + priorZW[z][w]) / (nZ[z] + phiNorm[z]);
				}

				LL += Math.log(tokenLL);
				N++;
			}
		}

		double perp = Math.exp(-LL / N);
		return perp;
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

		fr = new FileReader(filename);
		br = new BufferedReader(fr); 

		int d = 0;
		while ((s = br.readLine()) != null) {
			String[] tokens = s.split("\\s+");
			
			int N = tokens.length;
			
			docs[d] = new int[N-4]; // Ignore the two other features
			docsC[d] = Double.parseDouble(tokens[1]); 

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
				
				docs[d][n-4] = key;
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
		FileWriter fw = new FileWriter(filename+".assign");
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

		fw = new FileWriter(filename+".beta");
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

		fw = new FileWriter(filename+".betaB");
		bw = new BufferedWriter(fw);
		for (int z = 0; z < Z; z++) { 
			for (int c = 0; c < Cph; c++) {
				bw.write(betaB[z][c]+" ");
			}
			bw.newLine();
		}
		bw.close();
		fw.close();

		fw = new FileWriter(filename+".omega");
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

		fw = new FileWriter(filename+".omegaBias");
		bw = new BufferedWriter(fw);
		for (int w = 0; w < W; w++) {
			String word = wordMapInv.get(w);
			bw.write(word);
			bw.write(" "+omegaBias[w]);
			bw.newLine();
		}
		bw.close();
		fw.close();

		fw = new FileWriter(filename+".alpha");
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

		fw = new FileWriter(filename+".delta");
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

		fw = new FileWriter(filename+".deltaBias");
		bw = new BufferedWriter(fw);
		for (int z = 0; z < Z; z++) {
			bw.write(""+z);
			bw.write(" "+deltaBias[z]);
			bw.newLine();
		}
		bw.close();
		fw.close();
	}

	// Approximation to the digamma function, from Radford Neal.
	// can also use Gamma.digamma() from commons
	public static double digamma0(double x)
	{
		//return Gamma.digamma(x);

		double r = 0.0;

		while (x <= 5.0) {
			r -= 1.0 / x;
			x += 1.0;
		}

		double f = 1.0 / (x * x);
		double t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0 + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617.0 / 8160.0)))))));
		return r + Math.log(x) - 0.5 / x + t;
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
	public double computeLL(int[][][] corpus) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	protected void initTest() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void writeOutput(String filename, String outputDir) throws Exception {
		writeOutput(filename);
	}

	@Override
	public void cleanUp() throws Exception {
		// TODO Auto-generated method stub
		
	}
}
