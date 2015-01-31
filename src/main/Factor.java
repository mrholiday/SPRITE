package main;

import java.io.Serializable;

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
	public double[][][] omega;   // Component-to-token, for phi.  Per view since we have different vocabularies for each view.
	
	// Variance of Gaussian priors.  Default to 1.
	private double sigmaBeta  = 1.0;
	private double sigmaOmega = 1.0;
	private double sigmaAlpha = 1.0;
	private double sigmaDelta = 1.0;
	
	// Sigma squared
	private double sigmaBeta_sqr  = sigmaBeta * sigmaBeta;
	private double sigmaOmega_sqr = sigmaOmega * sigmaOmega;
	private double sigmaAlpha_sqr = sigmaAlpha * sigmaAlpha;
	private double sigmaDelta_sqr = sigmaDelta * sigmaDelta;
	
	// For learning these parameters.  Keeps track of gradient history.
	public double[][] adaAlpha; // Unused if this factor is observed
	public double[][][] adaDelta;
	public double[][][] adaBeta;
	public double[][][] adaBetaB;
	public double[][][] adaOmega;
	
	// Current estimate of gradient
	public double[][][] gradientOmega;
	public double[][][] gradientBeta;
	public double[][][] gradientBetaB;
	
	public double[][][] gradientDelta;
	public double[][] gradientAlpha;
	
	// Is the value for this factor observed?  If so, then alpha is given.
	// Changing this mid-training should cause document labels to be adjusted.
	private boolean observed;
	
	public int numViews;
	public int[] viewIndices; // The index of each view this factor affects
	
	public int C;
	public int[] Z; // Different number of topics per view
	public int[] W; // Different vocabulary per view
	public int D;
	
	public double rho;
	public final String factorName;
	
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
	
	public void initialize(int[] W0) {
		initialize(null, W0);
	}
	
	/**
	 * Painful initialization.  Silly me for trying to support more than one view.
	 * TODO: Figure out how to initialize more gracefully.
	 * 
	 * @param docScores If this factor is observed, alpha is set to these values.
	 * @param W0 Vocabulary size for each view this factor is responsible for.
	 */
	public void initialize(double[][] docScores, int[] W0) {
		Log.info("Factor",
				String.format("Initializing factor %s: C=%d, observed=%d, rho=%e",
					          factorName, C, observed ? 1 : 0, rho));
		
		W = W0;
		
		sigmaBeta_sqr = Math.pow(sigmaBeta, 2.0);
		sigmaAlpha_sqr = Math.pow(sigmaAlpha, 2.0);
		sigmaDelta_sqr = Math.pow(sigmaDelta, 2.0);
		sigmaOmega_sqr = Math.pow(sigmaOmega, 2.0);
		
		adaAlpha = new double[D][C];
		adaDelta = new double[numViews][C][];
		adaBeta  = new double[numViews][][];
		adaBetaB = new double[numViews][][];
		adaOmega = new double[numViews][C][];
		
		gradientAlpha = new double[D][C];
		gradientDelta = new double[numViews][C][];
		gradientBeta  = new double[numViews][][];
		gradientBetaB = new double[numViews][][];
		gradientOmega = new double[numViews][C][];
		
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
		omega = new double[numViews][C][]; // Component-to-token, for phi
		
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
			adaOmega[v]      = new double[C][W[v]];
			gradientOmega[v] = new double[C][W[v]];
			omega[v]         = new double[C][W[v]];
			for (int c = 0; c < C; c++) {
				for (int w = 0; w < W[v]; w++) {
					omega[v][c][w] = (MathUtils.r.nextDouble() - 0.5) / 100.0;
				}
			}
		}
	}
	
	/**
	 * Returns the prior P(\alpha).  Assumes a sparse Dirichlet prior.
	 */
	public double[][] getPriorAlpha() {
		
		
		return null;
	}
	
	/**
	 * Returns the prior P(\Beta).
	 */
	public double[][] getPriorBeta() {
		
		
		return null;
	}
	
	/**
	 * Returns the prior P(b).  Sparse Dirichlet prior.
	 */
	public double[][] getPriorBetaB() {
		
		
		return null;
	}
	
	/**
	 * 
	 * @param v View index
	 * @param z Topic index
	 * @param w Word index
	 * @return Partial sum for \widetilde{phi} for components in this factor.
	 */
	public double getPriorPhi(int v, int z, int w) {
		double weight = 0.0;
		
		for (int c = 0; c < C; c++) {
			double betaRightSign = betaPositive ? Math.exp(beta[v][z][c]) : beta[v][z][c];
			weight += betaB[v][z][c] * betaRightSign * omega[v][c][w];
		}
		
		return Math.exp(weight);
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
		double weight = 0.0;
		
		for (int c = 0; c < C; c++) {
			double deltaRightSign = deltaPositive  ? Math.exp(delta[v][c][z]) : delta[v][z][c];
			double alphaRightSign = alphaPositive ? Math.exp(alpha[d][c]) : alpha[d][c];
			weight += alphaRightSign * betaB[v][z][c] * deltaRightSign;
		}
		
		return weight;
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
				
				Log.info("factor " + factorName + " iteration", b.toString());
			}
			for (int z = 0; z < Z[v]; z++) {
				StringBuilder b = new StringBuilder();
				
				b.append(String.format("beta_%d_%d", v, z));
				for (int c = 0; c < C; c++) {
					b.append(String.format(" %.3f", beta[v][z][c]));
				}
				
				Log.info("factor " + factorName + " iteration", b.toString());
			}
			
			for (int w = 0; w < W[v]; w += 10000) {
				StringBuilder b = new StringBuilder();
				
				b.append(String.format("omega_%d_%d", v, w));
				for (int c = 0; c < C; c++) {
					b.append(String.format(" %.3f", omega[v][c][w]));
				}
				Log.info("factor " + factorName + " iteration", b.toString());
			}
		}
	}
	
	public boolean isObserved() { return observed; }
	
	public void setObserved(boolean observed0) { observed = observed0; }
	
}
