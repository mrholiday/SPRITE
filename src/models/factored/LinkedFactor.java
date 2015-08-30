package models.factored;

import java.util.Map;

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
			boolean deltaPositive0, String factorName0, boolean observed0, boolean optimizeMeTheta0,
			boolean optimizeMePhi0, double betaDeltaWeight0) {
		super(numComponents0, viewIndices0, Z0, rho0, tieBetaAndDelta0,
				sigmaBeta0, sigmaOmega0, sigmaAlpha0, sigmaDelta0,
				alphaPositive0, betaPositive0, deltaPositive0, factorName0,
				observed0, optimizeMeTheta0, optimizeMePhi0);
		betaDeltaWeight = betaDeltaWeight0;
	}
	
	@Override
	public void initialize(double[][] docScores, int W0, int D0, Map<String, Integer> wtoi, Map<Integer, String> itow) {
		super.initialize(docScores, W0, D0, wtoi, itow);
		
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
		gradientOmega[z][w] += gradientTerm;
	}
	
	@Override // Don't take gradient steps for delta
	public void updateThetaGradient(double gradientTerm, int z, int v, int d) {
		gradientAlpha[d][z] += gradientTerm;
	}
	
	@Override
	public void doGradientStep(int v, int minZ, int maxZ, int minD, int maxD,
			int minW, int maxW, double stepSize) {
		v = revViewIndices.get(v);
		
		// Make sure these don't change...
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
		
//		StringBuilder b = new StringBuilder();
//		for (int c = 0; c < C; c++) {
//			for (int w = minW; w < maxW; w++) {
//				b.append(String.format("%d:%.3e,", w, gradientOmega[c][w]));
//			}
//			Log.info(String.format("%s_c%d_v%d", factorName, c, v), "Gradient \\omega: " + b.toString());
//			b.delete(0, b.length());
//		}
		
		if (v == 0) { // Hack to make sure we only take a gradient step once for all views
		for (int c = 0; c < C; c++) {
			for (int w = minW; w < maxW; w++) {
				gradientOmega[c][w] += -(omega[c][w]) / sigmaOmega_sqr;
				adaOmega[c][w] += Math.pow(gradientOmega[c][w], 2);
				omega[c][w] += (stepSize / (Math.sqrt(adaOmega[c][w]) + MathUtils.eps)) * gradientOmega[c][w];
//				gradientOmega[c][w] = 0.; // Clear gradient for the next iteration
			}
		}
		}
		
//		for (int c = 0; c < C; c++) {
//			for (int d = minD; d < maxD; d += D/20) {
//				b.append(String.format("%d:%.3e,", d, gradientAlpha[d][c]));
//			}
//			Log.info(String.format("%s_c%d_v%d", factorName, c, v), "Gradient \\alpha sample: " + b.toString());
//			b.delete(0, b.length());
//		}
		
		if (!observed) {
			if (v == 0) { // Hack to make sure we only take a gradient step once for all views
				for (int d = minD; d < maxD; d++) {
					for (int c = 0; c < C; c++) {
						gradientAlpha[d][c] += -(alpha[d][c]) / sigmaAlpha_sqr;
						adaAlpha[d][c] += Math.pow(gradientAlpha[d][c], 2);
						alpha[d][c] += (stepSize / (Math.sqrt(adaAlpha[d][c]) + MathUtils.eps)) * gradientAlpha[d][c];
						//					gradientAlpha[d][c] = 0.; // Clear gradient for the next iteration
					}
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
		double weight = omega[z][w];
		
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
		double weight = alphaPositive ? Math.exp(alpha[d][z]) : alpha[d][z];
		
		return weight; // SpriteThetaPrior will exponentiate this
	}
	
}
