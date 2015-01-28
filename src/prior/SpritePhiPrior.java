package prior;

import main.Factor;

/**
 * Defines a Dirichlet prior over the vocabulary.  This prior is defined as a
 * weighted sum over components, which are grouped into factors.
 * 
 * @author adrianb
 *
 */
public class SpritePhiPrior {
	// The factors that feed into this prior.
	private Factor[] factors;
	
	public int Z; // Number of topics
	public int V; // Vocabulary size
	
	// Bias term.  Assuming this is ever-present.
	private double[] omegaBias;
	private double initOmegaBias; // Initial value for omegaBias
	
	// Sum over vocabulary for each topic.
	private double[] phiNorm;
	
	// Topic -> Word -> weight.  \widetilde{phi} in TACL paper
	private double[][] phiTilde;
	
	// For optimization
	//private double[][] gradientPhi;
	//private double[][] adaDeltaPhi;
	
	public SpritePhiPrior(Factor[] factors0, int Z0, int V0, double omegaInitBias0) {
		factors = factors0;
		Z = Z0;
		V = V0;
		initOmegaBias = omegaInitBias0;
		
		init();
	}
	
	private void init() {
		omegaBias = new double[V];
		for (int i = 0; i < V; i++) {
			omegaBias[i] = initOmegaBias;
		}
		
		updatePhiTilde();
		updatePhiNorm();
	}
	
	// Returns the phi_zw prior given all the parameters.  Factors are
	// responsible for computing their portion of the sums.
	private double priorZW(int z, int w) {
		double weight = omegaBias[w];
		
		for (Factor f : factors) {
			weight += f.getPriorPhi(z, w);
		}
		
		return Math.exp(weight);
	}
	
	/**
	 * Updates the word distribution prior for a range of documents.
	 * Updates both \widetilde{\phi} and the sum over the vocabulary.
	 * 
	 * @param minZ Minimum topic index
	 * @param maxZ Maximum topic index
	 */
	public void updatePrior(int minZ, int maxZ) {
		updatePhiTilde(minZ, maxZ);
		updatePhiNorm(minZ, maxZ);
	}
	
	private void updatePhiTilde(int minZ, int maxZ) {
		for (int z = minZ; z < maxZ; z++) {
			for (int w = 0; w < V; w++) {
				phiTilde[z][w] = priorZW(z, w);
			}
		}
	}
	
	private void updatePhiNorm(int minZ, int maxZ) {
		for (int j = minZ; j < maxZ; j++) {
			phiNorm[j] = 0.;
			for (int i = 0; i < V; i++) {
				phiNorm[j] += phiTilde[j][i];
			}
		}
	}
	
	private void updatePhiTilde() {
		updatePhiTilde(0, Z);
	}
	
	private void updatePhiNorm() {
		updatePhiNorm(0, Z);
	}
	
}
