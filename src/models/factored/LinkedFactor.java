package models.factored;

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

	public LinkedFactor(int numComponents0, int[] viewIndices0, int[] Z0,
			double rho0, boolean tieBetaAndDelta0, double sigmaBeta0,
			double sigmaOmega0, double sigmaAlpha0, double sigmaDelta0,
			boolean alphaPositive0, boolean betaPositive0,
			boolean deltaPositive0, String factorName0, boolean observed0) {
		super(numComponents0, viewIndices0, Z0, rho0, tieBetaAndDelta0,
				sigmaBeta0, sigmaOmega0, sigmaAlpha0, sigmaDelta0,
				alphaPositive0, betaPositive0, deltaPositive0, factorName0,
				observed0);
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
						delta[v][c][z] = 1.0;
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
					betaB[v][z][c] = 1.0;
					
					if (c == z) {
						beta[v][z][c] = 1.0;
					}
					else {
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
	
}