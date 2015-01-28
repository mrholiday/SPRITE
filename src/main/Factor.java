package main;

import java.util.Random;

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
public class Factor {
	// Which parameters are restricted to be positive.
	public boolean alphaPositive = false;
	public boolean betaPositive  = false;
	public boolean deltaPositive = false;
	
	// Should we beta and delta be tied?  This will set delta <- beta
	// after each gradient step.
	public boolean tieBetaAndDelta = false;
	
	// Parameters.  Note, no bias since those are in SpritePhiPrior and
	// SpriteThetaPrior.
	public double[][] alpha; // Document-to-component, for theta
	public double[][] delta; // Component-to-topic, for theta
	
	public double[][] beta;  // Topic-to-component, for phi
	public double[][] betaB; // Topic-to-component, sparse indicator portion of beta.  Ignored if rho >= 1
	public double[][] omega; // Component-to-token, for phi
	
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
	public double[][] adaDelta;
	public double[][] adaBeta;
	public double[][] adaBetaB;
	public double[][] adaOmega;
	
	// Current estimate of gradient
	public double[][] gradientOmega;
	public double[][] gradientBeta;
	public double[][] gradientBetaB;
	
	public double[][] gradientDelta;
	public double[][] gradientAlpha;
	
	// Is the value for this factor observed?  If so, then alpha is given.
	// Changing this mid-training should cause document labels to be adjusted.
	private boolean observed;
	
	private final int C;
	private final int Z;
	private final int W;
	private final int D;
	private double rho;
	public final String factorName;
	
	/**
	 * Latent factor.  Need to infer component values for each document
	 * as well as weights for each from factor components to topics.
	 * 
	 * @param numComponents0
	 * @param Z0 Number of topics in the whole model
	 * @param W0 Number of token types
	 * @param D0 Number of documents
	 * @param rho0 If >=1 then we do not apply component->topic sparsity
	 *             (beta drawn from a zero-mean gaussian).
	 *             If <1, then beta is drawn from symmetric Dirichlet prior
	 *             with rho being the weight on alpha (U-shaped since rho<1).
	 *             Beta may also be weighted.
	 * @param tieBetaAndDelta0 These values should be tied together when updating.
	 * @param factorName0
	 * 
	 */
	public Factor(int numComponents0, int Z0, int W0, int D0, double rho0, boolean tieBetaAndDelta0,
				  double sigmaBeta0, double sigmaOmega0, double sigmaAlpha0, double sigmaDelta0, boolean alphaPositive0,
				  boolean betaPositive0, boolean deltaPositive0, String factorName0) {
		observed = false;
		C = numComponents0;
		Z = Z0;
		W = W0;
		D = D0;
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
		
		initialize();
	}
	
	/**
	 * Pass a weighting over components in this factor -- observed.
	 * 
	 * @param docLabels Real value for each document, for each component.
	 * @param Z0 Number of topics in the whole model
	 * @param D0 Number of documents
	 * @param rho If >=1 then we do not apply component->topic sparsity.
	 *            If <1, then beta is drawn from Dirichlet prior
	 *            with rho (U-shaped since rho<1).
	 * @param tieBetaAndDelta0 These values should be tied together when updating.
	 * 
	 */
	public Factor(double[][] docLabels0, int Z0, int W0, double rho0, boolean tieBetaAndDelta0,
					double sigmaBeta0, double sigmaOmega0, double sigmaAlpha0, double sigmaDelta0,
					boolean alphaPositive0, boolean betaPositive0, boolean deltaPositive0,
					String factorName0) {
		observed = true;
		alpha = docLabels0;
		C = alpha[0].length;
		Z = Z0;
		W = W0;
		D = alpha.length;
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
		
		initialize();
	}
	
	private void initialize() {
		System.out.println(String.format("Initializing factor %s: C=%d, observed=%d, rho=%e",
					        factorName, C, observed ? 1 : 0, rho)
					       );
		
		sigmaBeta_sqr = Math.pow(sigmaBeta, 2.0);
		sigmaAlpha_sqr = Math.pow(sigmaAlpha, 2.0);
		sigmaDelta_sqr = Math.pow(sigmaDelta, 2.0);
		sigmaOmega_sqr = Math.pow(sigmaOmega, 2.0);
		
		adaAlpha = new double[D][C]; // Unused if this factor is observed
		adaDelta = new double[C][Z];
		adaBeta  = new double[Z][C];
		adaBetaB = new double[Z][C];
		adaOmega = new double[C][W];
		
		gradientOmega = new double[C][W];
		gradientBeta  = new double[Z][C];
		gradientBetaB = new double[Z][C];
		
		gradientAlpha = new double[D][C];
		gradientDelta = new double[C][Z];
		
		beta  = new double[Z][C]; // Topic-to-component, for phi
		betaB = new double[Z][C]; // Topic-to-component, sparse indicator portion of beta.  Ignored if rho >= 1
		omega = new double[C][W]; // Component-to-token, for phi
		
		alpha = new double[D][C]; // Document-to-component, for theta
		delta = new double[C][Z]; // Component-to-topic, for theta
		
		if (!observed) {
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
		
		for (int c = 0; c < C; c++) { 
			for (int z = 0; z < Z; z++) {
				if (!deltaPositive) {
					delta[c][z] = (MathUtils.r.nextDouble() - 0.5) / 100.0;
				}
				else {
					delta[c][z] = -2.0; // Small positive number when exped
				}
			}
		}
		
		for (int c = 0; c < C; c++) {
			for (int w = 0; w < W; w++) {
				omega[c][w] = (MathUtils.r.nextDouble() - 0.5) / 100.0;
			}
		}
		
		for (int z = 0; z < Z; z++) {
			for (int c = 0; c < C; c++) {
				betaB[z][c] = 1.0;
				
				if (tieBetaAndDelta) {
					beta[z][c] = delta[c][z];
				}
				else {
					if (!betaPositive) {
						beta[z][c] = (MathUtils.r.nextDouble() - 0.5) / 100.0;
					}
					else {
						beta[z][c] = -2.0; // Small positive number when exped
					}
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
	 * @param z Topic index
	 * @param w Word index
	 * @return Partial sum for \widetilde{phi} for components in this factor.
	 */
	public double getPriorPhi(int z, int w) {
		double weight = 0.0;
		
		for (int c = 0; c < C; c++) {
			double betaRightSign = betaPositive ? Math.exp(beta[z][c]) : beta[z][c];
			weight += betaB[z][c] * betaRightSign * omega[c][w];
		}
		
		return Math.exp(weight);
	}
	
	/**
	 * 
	 * @param d Document index
	 * @param z Topic index
	 * @return The partial sum for \widetilde{theta} for components in this factor.
	 */
	public double getPriorTheta(int d, int z) {
		double weight = 0.0;
		
		for (int c = 0; c < C; c++) {
			double deltaRightSign = deltaPositive  ? Math.exp(delta[c][z]) : delta[z][c];
			double alphaRightSign = alphaPositive ? Math.exp(alpha[d][c]) : alpha[d][c];
			weight += alphaRightSign * betaB[z][c] * deltaRightSign;
		}
		
		return weight;
	}
	
	public boolean getObserved() { return observed; }
	
	public void setObserved(boolean observed0) { observed = observed0; }
	
}
