package models.factored;

import utils.MathUtils;

/**
 * Factor where the beta/delta for each topic is 1 for all views -> component.
 * This assumes that this factor has the same number of components as all topics
 * within views.
 * 
 * @author adrianb
 *
 */
public class LinkedFactor extends Factor {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2509749995832748702L;
	
	private double betaDeltaWeight = 1.0;
	
	public LinkedFactor(int numComponents0, int[] viewIndices0, int[] Z0,
			double rho0, boolean tieBetaAndDelta0, double sigmaBeta0,
			double sigmaOmega0, double sigmaAlpha0, double sigmaDelta0,
			boolean alphaPositive0, boolean betaPositive0,
			boolean deltaPositive0, String factorName0, boolean observed0, double betaDeltaWeight0) {
		super(numComponents0, viewIndices0, Z0, rho0, tieBetaAndDelta0,
				sigmaBeta0, sigmaOmega0, sigmaAlpha0, sigmaDelta0,
				alphaPositive0, betaPositive0, deltaPositive0, factorName0,
				observed0);
		betaDeltaWeight = betaDeltaWeight0;
	}
	
	@Override
	public void initialize(double[][] docScores, int W0, int D0) {
		super.initialize(docScores, W0, D0);
		
		// Set beta and delta to 1 only if the component index is the same as the topic index
		beta  = new double[numViews][][];
		delta = new double[numViews][C][];
		
		for (int v = 0; v < numViews; v++) {
			for (int c = 0; c < C; c++) {
				delta[v][c] = new double[Z[v]];
				for (int z = 0; z < Z[v]; z++) {
					if (c == z) {
						delta[v][c][z] = betaDeltaWeight;
					}
					else {
						delta[v][c][z] = 0.0;
					}
				}
			}
			
			// Init beta/beta bias
			beta[v] = new double[Z[v]][C];
			for (int z = 0; z < Z[v]; z++) {
				for (int c = 0; c < C; c++) {
					
					if (c == z) {
						betaB[v][z][c] = 1.0;
						beta[v][z][c] = betaDeltaWeight;
					}
					else {
						betaB[v][z][c] = 0.0;
						beta[v][z][c] = 0.0;
					}
				}
			}
		}
	}
	
	@Override // Do not take gradient steps for beta
	public void updatePhiGradient(double gradientTerm, int z, int v, int w) {
		v = revViewIndices.get(v);
		
		for (int c = 0; c < C; c++) {
			double betaRightSign = beta[v][z][c];
			
			gradientOmega[c][w] += betaRightSign * gradientTerm;
		}
	}
	
	@Override // Don't take gradient steps for delta
	public void updateThetaGradient(double gradientTerm, int z, int v, int d) {
		v = revViewIndices.get(v);
		
		for (int c = 0; c < C; c++) {
			double deltaRightSign = delta[v][c][z];
			
			if (!observed) {
				gradientAlpha[d][c] += deltaRightSign * gradientTerm;
			}
		}
	}
	
	@Override
	public void doGradientStep(int v, int minZ, int maxZ, int minD, int maxD,
			int minW, int maxW, double stepSize) {
		v = revViewIndices.get(v);
		
		for (int z = minZ; z < maxZ; z++) {
			for (int c = 0; c < C; c++) {
				if (c == z) {
					betaB[v][z][c] = 1.0;
					beta[v][z][c] = betaDeltaWeight;
					delta[v][c][z] = betaDeltaWeight;
				}
				else {
					betaB[v][z][c] = 0.0;
					beta[v][z][c] = 0.0;
					delta[v][c][z] = 0.0;
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
		
	}
	
	/**
	 * Only updates the component with the same index as z.
	 * 
	 * @param v View index
	 * @param z Topic index
	 * @param w Word index
	 * @return Partial sum for \widetilde{phi} for components in this factor.
	 */
	@Override
	public double getPriorPhi(int v, int z, int w) {
		v = revViewIndices.get(v);
		
		double weight = 0.0;
		
		int c = z;
		
		if (betaB[v][z][c] > MathUtils.eps) {
			double betaRightSign = betaPositive ? Math.exp(beta[v][z][c]) : beta[v][z][c];
			weight += betaB[v][z][c] * betaRightSign * omega[c][w];
			//weight += betaB[v][z][c] * betaRightSign * omega[v][c][w];
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
		
		int c = z;
		
		if (betaB[v][z][c] > MathUtils.eps) {
			double deltaRightSign = deltaPositive  ? Math.exp(delta[v][c][z]) : delta[v][c][z];
			double alphaRightSign = alphaPositive ? Math.exp(alpha[d][c]) : alpha[d][c];
			weight += alphaRightSign * betaB[v][z][c] * deltaRightSign;
		}
		
		return weight; // SpriteThetaPrior will exponentiate this
	}
	
}
