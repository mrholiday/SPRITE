package models.factored;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import utils.Log;
import utils.MathUtils;

/**
 * 
 * Factor feeding into topics.  May be observed or latent.  Assumes
 * that this factor affects both \widetilde{\theta} and \widetilde{\phi}.
 * Used by SpriteFactoredTopicModel.
 * 
 * @author adrianb
 * 
 */
public class Factor implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5910933371949825941L;
	
	// Which parameters are restricted to be positive.
	public boolean alphaPositive = false;
	public boolean betaPositive  = false;
	public boolean deltaPositive = false;
	
	// Should beta and delta be tied?  This will set delta <- beta
	// after each gradient step.
	public boolean tieBetaAndDelta = false;
	
	// Parameters.  Note, no bias since those are in SpritePhiPrior and
	// SpriteThetaPrior.
	public double[][] alpha; // Document-to-Component, for theta.  Shared across all views with this factor
	public double[][][] delta; // View-to-Component-to-Topic, for theta.  Per view, since views have different number of topics.
	public double[][][] beta;  // View-to-Topic-to-component, for phi.  Per view, since views have different number of topics.
	public double[][][] betaB; // View-to-Topic-to-component, sparse indicator portion of beta.  Ignored if rho >= 1.  Sparsity enforced within a view.
	public double[][] omega;   // Component-to-token, for phi
	// public double[][][] omega;   // Component-to-token, for phi.  Per view since we have different vocabularies for each view.
	
	// Variance of Gaussian priors.  Default to 1.
	private double sigmaBeta  = 1.0;
	private double sigmaOmega = 1.0;
	private double sigmaAlpha = 1.0;
	private double sigmaDelta = 1.0;
	
	// Sigma squared
	protected double sigmaBeta_sqr  = sigmaBeta * sigmaBeta;
	protected double sigmaOmega_sqr = sigmaOmega * sigmaOmega;
	protected double sigmaAlpha_sqr = sigmaAlpha * sigmaAlpha;
	protected double sigmaDelta_sqr = sigmaDelta * sigmaDelta;
	
	// For learning these parameters.  Keeps track of gradient history.
	public double[][] adaAlpha; // Unused if this factor is observed
	public double[][][] adaDelta;
	public double[][][] adaBeta;
	public double[][][] adaBetaB;
	public double[][] adaOmega;
	//public double[][][] adaOmega;
	
	// Current estimate of gradient
	public double[][] gradientOmega;
	//public double[][][] gradientOmega;
	public double[][][] gradientBeta;
	public double[][][] gradientBetaB;
	public double[][][] gradientDelta;
	public double[][] gradientAlpha;
	
	// Is the value for this factor observed?  If so, then alpha is given.
	// Changing this mid-training should cause document labels to be adjusted.
	protected boolean observed;
	
	public int numViews;
	public int[] viewIndices; // The index of each view this factor affects
	
	public int C;
	public int[] Z; // Different number of topics per view
	public int W; // Vocabulary size is union across all views
	//public int[] W; // Different vocabulary per view
	public int D;
	
	public double rho;
	public boolean isSparse;
	public final String factorName;
	
	public Map<Integer, Integer> revViewIndices;
	
	/**
	 * Latent/observed factor.  If it is observed, you should intiialize
	 * alpha with the initialize method.
	 * 
	 * @param numComponents0 Number of components/super-topics
	 * @param viewIndices of each of the views this maps onto
	 * @param Z0 Number of topics for each view this feeds into
	 * @param rho0 If >=1 then we do not apply component->topic sparsity
	 *             (beta drawn from a zero-mean gaussian).
	 *             If <1, then B is drawn from symmetric Dirichlet prior
	 *             with rho being the weight on alpha (U-shaped since rho<1).
	 *             Beta is also weighted
	 * @param tieBetaAndDelta0 These values should be tied together when updating
	 * @param sigmaBeta0  Beta prior stddev
	 * @param sigmaOmega0 Omega prior stddev
	 * @param sigmaAlpha0 Alpha prior stddev
	 * @param sigmaDelta0 Delta prior stddev
	 * @param alphaPositive0 Alpha is strictly positive
	 * @param betaPositive0 Beta is strictly positive
	 * @param deltaPositive0 Delta is strictly positive
	 * @param factorName0 An informative name for this bag of doubles
	 * 
	 */
	public Factor(int numComponents0, int[] viewIndices0, int[] Z0, double rho0, boolean tieBetaAndDelta0,
				  double sigmaBeta0, double sigmaOmega0, double sigmaAlpha0, double sigmaDelta0, boolean alphaPositive0,
				  boolean betaPositive0, boolean deltaPositive0, String factorName0, boolean observed0) {
		observed = observed0;
		
		viewIndices = viewIndices0;
		numViews    = viewIndices.length;
		
		Z = Z0;
		
		C = numComponents0;
		rho = rho0;
		isSparse = rho < 1;
		
		sigmaBeta = sigmaBeta0;
		sigmaOmega = sigmaOmega0;
		sigmaAlpha = sigmaAlpha0;
		sigmaDelta = sigmaDelta0;
		
		alphaPositive = alphaPositive0;
		betaPositive  = betaPositive0;
		deltaPositive = deltaPositive0;
		
		tieBetaAndDelta = tieBetaAndDelta0;
		factorName = factorName0;
	}
	
	public void initialize(int W0, int D0) {
		initialize(null, W0, D0);
	}
	
	/**
	 * Painful initialization.  Silly me for trying to support more than one view.
	 * TODO: Figure out how to initialize more gracefully.
	 * 
	 * @param docScores If this factor is observed, alpha is set to these values.
	 * @param W0 Vocabulary size for all views
	 */
	public void initialize(double[][] docScores, int W0, int D0) {
		Log.info("factor_" + factorName,
				String.format("Initializing factor %s: C=%d, observed=%d, rho=%e",
					          factorName, C, observed ? 1 : 0, rho));
		
		W = W0;
		D = D0;
		
		revViewIndices = new HashMap<Integer, Integer>();
		for (int i = 0; i < viewIndices.length; i++)
			revViewIndices.put(viewIndices[i], i);
		
		sigmaBeta_sqr = Math.pow(sigmaBeta, 2.0);
		sigmaAlpha_sqr = Math.pow(sigmaAlpha, 2.0);
		sigmaDelta_sqr = Math.pow(sigmaDelta, 2.0);
		sigmaOmega_sqr = Math.pow(sigmaOmega, 2.0);
		
		adaAlpha = new double[D][C];
		adaDelta = new double[numViews][C][];
		adaBeta  = new double[numViews][][];
		adaBetaB = new double[numViews][][];
		adaOmega = new double[C][W];
		//adaOmega = new double[numViews][C][];
		
		gradientAlpha = new double[D][C];
		gradientDelta = new double[numViews][C][];
		gradientBeta  = new double[numViews][][];
		gradientBetaB = new double[numViews][][];
		gradientOmega = new double[C][W];
		//gradientOmega = new double[numViews][C][];
		
		// Init alpha, component weight for each document
		if (docScores == null) {
			alpha = new double[D][C]; // Latent factor
			for (int d = 0; d < D; d++) {
				for (int c = 0; c < C; c++) {
					if (!alphaPositive) {
						alpha[d][c] += (MathUtils.r.nextDouble() - 0.5) / 100.0;
					}
					else {
						alpha[d][c] = -2.0; // Small positive number when exped
					}
				}
			}
		}
		else {
			alpha = docScores; // Observed
		}
		
		delta = new double[numViews][C][]; // Component-to-topic, for theta
		beta  = new double[numViews][][]; // Topic-to-component, for phi
		betaB = new double[numViews][][]; // Topic-to-component, sparse indicator portion of beta.  Ignored if rho >= 1
		omega = new double[C][W]; // Component-to-token, for phi
		//omega = new double[numViews][C][]; // Component-to-token, for phi
		
		// Init omega
		for (int c = 0; c < C; c++) {
			for (int w = 0; w < W; w++) {
				omega[c][w] = (MathUtils.r.nextDouble() - 0.5) / 100.0;
			}
		}
		
		for (int v = 0; v < numViews; v++) {
			for (int c = 0; c < C; c++) {
				// Init delta
				adaDelta[v][c] = new double[Z[v]];
				gradientDelta[v][c] = new double[Z[v]];
				delta[v][c] = new double[Z[v]];
				for (int z = 0; z < Z[v]; z++) {
					if (!deltaPositive) {
						delta[v][c][z] = (MathUtils.r.nextDouble() - 0.5) / 100.0;
					}
					else {
						delta[v][c][z] = -2.0; // Small positive number when exped
					}
				}
			}
			
			// Init beta/beta bias
			adaBeta[v] = new double[Z[v]][C];
			adaBetaB[v] = new double[Z[v]][C];
			gradientBeta[v] = new double[Z[v]][C];
			gradientBetaB[v] = new double[Z[v]][C];
			beta[v] = new double[Z[v]][C];
			betaB[v] = new double[Z[v]][C];
			for (int z = 0; z < Z[v]; z++) {
				for (int c = 0; c < C; c++) {
					betaB[v][z][c] = 1.0;
					
					if (tieBetaAndDelta) {
						beta[v][z][c] = delta[v][c][z];
					}
					else {
						if (!betaPositive) {
							beta[v][z][c] = (MathUtils.r.nextDouble() - 0.5) / 100.0;
						}
						else {
							beta[v][z][c] = -2.0; // Small positive number when exped
						}
					}

				}
			}
			
			// Init omega
//			adaOmega[v]      = new double[C][W[v]];
//			gradientOmega[v] = new double[C][W[v]];
//			omega[v]         = new double[C][W[v]];
//			for (int c = 0; c < C; c++) {
//				for (int w = 0; w < W[v]; w++) {
//					omega[v][c][w] = (MathUtils.r.nextDouble() - 0.5) / 100.0;
//				}
//			}
		}
	}
	
	/**
	 * 
	 * @param gradientTerm Gradient for LL
	 * @param z Topic
	 * @param v View
	 * @param w Word
	 */
	public void updatePhiGradient(double gradientTerm, int z, int v, int w) {
		v = revViewIndices.get(v);
		
		for (int c = 0; c < C; c++) {
			double betaRightSign    = betaPositive ? Math.exp(beta[v][z][c]) : beta[v][z][c];
			
			if (isSparse) {
				if (betaB[v][z][c] > MathUtils.eps) {
					gradientBeta[v][z][c] += omega[c][w] * betaB[v][z][c] * gradientTerm;
				}
				gradientBetaB[v][z][c] += betaRightSign * omega[c][w] * gradientTerm;
			}
			else {
				gradientBeta[v][z][c] += omega[c][w] * gradientTerm;
			}
			
			gradientOmega[c][w] += betaRightSign * betaB[v][z][c] * gradientTerm;
		}
		
//		for (int c = 0; c < C; c++) {
//			double betaRightSign    = betaPositive ? Math.exp(beta[v][z][c]) : beta[v][z][c];
//			
//			if (isSparse) {
//				gradientBeta[v][z][c] += omega[v][c][w] * betaB[v][z][c] * gradientTerm;
//				gradientBetaB[v][z][c] += betaRightSign * omega[v][c][w] * gradientTerm;
//			}
//			else {
//				gradientBeta[v][z][c] += omega[v][c][w] * gradientTerm;
//			}
//
//			gradientOmega[v][c][w] += betaRightSign * betaB[v][z][c] * gradientTerm;
//		}
	}
	
	public void updateThetaGradient(double gradientTerm, int z, int v, int d) {
		v = revViewIndices.get(v);
		
		for (int c = 0; c < C; c++) {
			double deltaRightSign = deltaPositive ? Math.exp(delta[v][c][z]) : delta[v][c][z];
			double alphaRightSign = alphaPositive ? Math.exp(alpha[d][c]) : alpha[d][c];
			
			if (tieBetaAndDelta) { // Will assign beta to delta after the gradient step
				if (isSparse && (betaB[v][z][c] > MathUtils.eps)) {
					gradientBeta[v][z][c] += alphaRightSign * betaB[v][z][c] * gradientTerm;
				}
				else {
					gradientBeta[v][z][c] += alphaRightSign * gradientTerm;
				}
			}
			else {
				if (isSparse && (betaB[v][z][c] > MathUtils.eps)) {
					gradientDelta[v][c][z] += alphaRightSign * betaB[v][z][c] * gradientTerm;
				}
				else {
					gradientDelta[v][c][z] += alphaRightSign * gradientTerm;
				}
			}
			
			if (isSparse)
				gradientBetaB[v][z][c] += alphaRightSign * deltaRightSign * gradientTerm;
			
			if (!observed) {
				if (isSparse && (betaB[v][z][c] > MathUtils.eps)) {
					gradientAlpha[d][c] += betaB[v][z][c] * deltaRightSign * gradientTerm;
				}
				else {
					gradientAlpha[d][c] += deltaRightSign * gradientTerm;
				}
			}
		}
	}
	
	/**
	 * 
	 * @param v View
	 * @param minZ
	 * @param maxZ
	 * @param minD
	 * @param maxD
	 * @param minW
	 * @param maxW
	 * @param stepSize
	 */
	public void doGradientStep(int v, int minZ, int maxZ, int minD, int maxD, int minW, int maxW, double stepSize) {
		v = revViewIndices.get(v);
		
		for (int z = minZ; z < maxZ; z++) {
			for (int c = 0; c < C; c++) {
				gradientBeta[v][z][c] += -(beta[v][z][c]) / sigmaBeta_sqr;
				adaBeta[v][z][c] += Math.pow(gradientBeta[v][z][c], 2);
				beta[v][z][c] += (stepSize / (Math.sqrt(adaBeta[v][z][c]) + MathUtils.eps)) * gradientBeta[v][z][c];
				gradientBeta[v][z][c] = 0.; // Clear gradient for the next iteration
			}
		}
		
		if (isSparse) {
			double adadeltaRho = 0.95; // AdaDelta weighting
			//double priorTemp = Math.pow(1.00, 200-199);
			double priorTemp = 1.0; // TODO: Set to a constant, since I don't pass iteration number right now.
			//Log.info("factor_" + factorName, "priorTemp = " + priorTemp);
			double[][] prevBetaB = new double[Z[v]][C];
			for (int z = minZ; z < maxZ; z++) {
				double norm = 0.0;
				
				for (int c = 0; c < C; c++) {
					double prior = (rho - 1.0) / betaB[v][z][c];
					
					if (adaBetaB[v][z][c] == 0) adaBetaB[v][z][c] = 1.0;
					adaBetaB[v][z][c] = (adadeltaRho * adaBetaB[v][z][c]) + ((1.0 - adadeltaRho) * Math.pow(gradientBetaB[v][z][c], 2)); // average
					gradientBetaB[v][z][c] += priorTemp * prior; // exclude from adaBetaB
					prevBetaB[z][c] = betaB[v][z][c]; // store in case update goes wrong
					betaB[v][z][c] *= Math.exp((stepSize / (Math.sqrt(adaBetaB[v][z][c]) + MathUtils.eps)) * gradientBetaB[v][z][c]);
					
					norm += betaB[v][z][c];
					gradientBetaB[v][z][c] = 0.;
				}
				for (int c = 0; c < C; c++) {
					if (norm == 0.0 || norm == Double.POSITIVE_INFINITY) {
						Log.warn("factor_" + factorName, "Bad BetaB");
						//betaB[z][c] = 1.0 / (Cph - 1);
						betaB[v][z][c] = prevBetaB[z][c]; // undo update if not well defined
					}
					else {
						betaB[v][z][c] /= norm;
					}
				}
			}
		}
		else {
			for (int z = minZ; z < maxZ; z++) {
				for (int c = 0; c < C; c++) {
					betaB[v][z][c] = 1.0;
				}
			}
		}
		
		for (int c = 0; c < C; c++) {
			for (int w = minW; w < maxW; w++) {
				gradientOmega[c][w] += -(omega[c][w]) / sigmaOmega_sqr;
				adaOmega[c][w] += Math.pow(gradientOmega[c][w], 2);
				omega[c][w] += (stepSize / (Math.sqrt(adaOmega[c][w]) + MathUtils.eps)) * gradientOmega[c][w];
				gradientOmega[c][w] = 0.; // Clear gradient for the next iteration
			}
		}
		
//		for (int c = 0; c < C; c++) {
//			for (int w = minW; w < maxW; w++) {
//				gradientOmega[v][c][w] += -(omega[v][c][w]) / sigmaOmega_sqr;
//				adaOmega[v][c][w] += Math.pow(gradientOmega[v][c][w], 2);
//				omega[v][c][w] += (stepSize / (Math.sqrt(adaOmega[v][c][w]) + MathUtils.eps)) * gradientOmega[v][c][w];
//				gradientOmega[v][c][w] = 0.; // Clear gradient for the next iteration
//			}
//		}
		
		if (!observed) {
			for (int d = minD; d < maxD; d++) {
				for (int c = 0; c < C; c++) {
					gradientAlpha[d][c] += -(alpha[d][c]) / sigmaAlpha_sqr;
					adaAlpha[d][c] += Math.pow(gradientAlpha[d][c], 2);
					alpha[d][c] += (stepSize / (Math.sqrt(adaAlpha[d][c]) + MathUtils.eps)) * gradientAlpha[d][c];
					gradientAlpha[d][c] = 0.; // Clear gradient for the next iteration
				}
			}
		}
		
		if (tieBetaAndDelta) {
			for (int c = 0; c < C; c++) {
				for (int z = minZ; z < maxZ; z++) {
					delta[v][c][z] = beta[v][z][c];
				}
			}
		}
		else {
			for (int c = 0; c < C; c++) {
				for (int z = minZ; z < maxZ; z++) {
					gradientDelta[v][c][z] += -(delta[v][c][z]) / sigmaDelta_sqr;
					adaDelta[v][c][z] += Math.pow(gradientDelta[v][c][z], 2);
					delta[v][c][z] += (stepSize / (Math.sqrt(adaDelta[v][c][z]) + MathUtils.eps)) * gradientDelta[v][c][z];
					gradientDelta[v][c][z] = 0.; // Clear gradient for next iteration
				}
			}
		}
	}
	
	/**
	 * 
	 * @param v View index
	 * @param z Topic index
	 * @param w Word index
	 * @return Partial sum for \widetilde{phi} for components in this factor.
	 */
	public double getPriorPhi(int v, int z, int w) {
		v = revViewIndices.get(v);
		
		double weight = 0.0;
		
		for (int c = 0; c < C; c++) {
			if (betaB[v][z][c] > MathUtils.eps) {
				double betaRightSign = betaPositive ? Math.exp(beta[v][z][c]) : beta[v][z][c];
				weight += betaB[v][z][c] * betaRightSign * omega[c][w];
				//weight += betaB[v][z][c] * betaRightSign * omega[v][c][w];
			}
		}
		
		return weight; // SpritePhiPrior will exponentiate this
	}
	
	/**
	 * 
	 * @param v View index
	 * @param d Document index
	 * @param z Topic index
	 * 
	 * @return The partial sum for \widetilde{theta} for components in this factor.
	 */
	public double getPriorTheta(int v, int d, int z) {
		v = revViewIndices.get(v);
		
		double weight = 0.0;
		
		for (int c = 0; c < C; c++) {
			if (betaB[v][z][c] > MathUtils.eps) {
				double deltaRightSign = deltaPositive  ? Math.exp(delta[v][c][z]) : delta[v][c][z];
				double alphaRightSign = alphaPositive ? Math.exp(alpha[d][c]) : alpha[d][c];
				weight += alphaRightSign * betaB[v][z][c] * deltaRightSign;
			}
		}
		
		return weight; // SpriteThetaPrior will exponentiate this
	}
	
	/**
	 * Logs current values for this iteration.  Bias values are printed out by Sprite(Theta|Phi)Prior.
	 */
	public void logState() {
		for (int v = 0; v < numViews; v++) {
			for (int z = 0; z < Z[v]; z++) {
				StringBuilder b = new StringBuilder();
				
				b.append(String.format("delta_%d_%d", v, z));
				for (int c = 0; c < C; c++) {
					b.append(String.format(" %.3f", delta[v][c][z]));
				}
				
				Log.info("factor_" + factorName + "_iteration", b.toString());
			}
			for (int z = 0; z < Z[v]; z++) {
				StringBuilder b = new StringBuilder();
				
				b.append(String.format("beta_%d_%d", v, z));
				for (int c = 0; c < C; c++) {
					b.append(String.format(" %.3f", beta[v][z][c]));
				}
				
				Log.info("factor_" + factorName + "_iteration", b.toString());
			}
			
		}
		
		//for (int w = 0; w < W[v]; w += 10000) {
		for (int w = 0; w < W; w += 10000) {
			StringBuilder b = new StringBuilder();
			
			b.append(String.format("omega_%d", w));
			for (int c = 0; c < C; c++) {
				b.append(String.format(" %.3f", omega[c][w]));
				//b.append(String.format(" %.3f", omega[v][c][w]));
			}
			Log.info("factor_" + factorName + "_iteration", b.toString());
		}
	}
	
	public boolean isObserved() { return observed; }
	
	public void setObserved(boolean observed0) { observed = observed0; }
	
	/**
	 * For printing out document component assignments to .assign file.
	 * 
	 * @param d Document
	 * @return String to print to .assign
	 */
	public String getAlphaString(int d) {
		StringBuilder builder = new StringBuilder();
		
		builder.append("" + this.alpha[d][0]);
		for (int c = 1; c < C; c++) {
			builder.append(" " + this.alpha[d][c]);
		}
		
		return builder.toString();
	}
	
	public void writeBeta(BufferedWriter bw, boolean printBiasTerm) throws IOException {
		for (int v = 0; v < numViews; v++) {
			int view = viewIndices[v];
			for (int c = 0; c < C; c++) {
				bw.write(String.format("%d_%d", view, c));
				for (int z = 0; z < Z[v]; z++) {
					double value = printBiasTerm ? betaB[v][z][c] : beta[v][z][c];
					
					if (!printBiasTerm && betaPositive)
						value = Math.exp(beta[v][z][c]);
					
					bw.write(" " + value);
				}
				bw.newLine();
			}
		}
	}
	
//	public void writeOmega(BufferedWriter bw, int v, Map<Integer, String> wordMapInv) throws IOException {
//		v = revViewIndices.get(v);
	public void writeOmega(BufferedWriter bw, Map<Integer, String> wordMapInv) throws IOException {
		
		//for (int w = 0; w < W[v]; w++) {
		for (int w = 0; w < W; w++) {
			String word = wordMapInv.get(w);
			bw.write(word);
			
			for (int c = 0; c < C; c++) { 
				bw.write(" " + omega[c][w]);
			}
			bw.newLine();
		}
	}
	
	public void writeDelta(BufferedWriter bw) throws IOException {
		for (int v = 0; v < numViews; v++) {
			for (int c = 0; c < C; c++) {
				int view = revViewIndices.get(v);
				bw.write(String.format("%d_%d", view, c));
				
				for (int z = 0; z < Z[v]; z++) {
					double deltaRightSign = deltaPositive ? Math.exp(delta[v][c][z]) : delta[v][c][z];
					bw.write(" " + deltaRightSign);
				}
				bw.newLine();
			}
		}
	}
	
	public void writeAlpha(BufferedWriter bw) throws IOException {
		for (int d = 0; d < D; d++) {
			for (int c = 0; c < C; c++) { 
				double alphaRightSign = alphaPositive ? Math.exp(alpha[d][c]) : alpha[d][c];
				bw.write(alphaRightSign + " ");
			}
			
			bw.newLine();
		}
	}
	
}
