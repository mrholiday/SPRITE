package prior;

import utils.Log;
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
	public int W; // Vocabulary size
	
	// Bias term.  Assuming this is ever-present.
	private double[] omegaBias;
	private double initOmegaBias; // Initial value for omegaBias
	
	// Sum over vocabulary for each topic.
	private double[] phiNorm;
	
	// Topic -> Word -> weight.  \widetilde{phi} in TACL paper
	private double[][] phiTilde;
	
	private int currentView; // The view this \widetilde{\phi} is responsible for.
	
	// For optimization
	//private double[][] gradientPhi;
	//private double[][] adaDeltaPhi;
	
	public SpritePhiPrior(Factor[] factors0, int Z0, int W0, int currentView0, double omegaInitBias0) {
		factors = factors0;
		Z = Z0;
		W = W0;
		currentView = currentView0;
		initOmegaBias = omegaInitBias0;
		
		init();
	}
	
	private void init() {
		omegaBias = new double[W];
		for (int i = 0; i < W; i++) {
			omegaBias[i] = initOmegaBias;
		}
		
		updatePhiTilde();
		updatePhiNorm();
	}
	
	// Returns the phi_zw prior given all the parameters.  Factors are
	// responsible for computing their portion of the sums.
	private double priorZW(int v, int z, int w) {
		double weight = omegaBias[w];
		
		for (Factor f : factors) {
			weight += f.getPriorPhi(v, z, w);
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
			for (int w = 0; w < W; w++) {
				phiTilde[z][w] = priorZW(currentView, z, w);
			}
		}
	}
	
	private void updatePhiNorm(int minZ, int maxZ) {
		for (int j = minZ; j < maxZ; j++) {
			phiNorm[j] = 0.;
			for (int i = 0; i < W; i++) {
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
	
	/**
	 * Logs bias term for this iteration.
	 */
	public void logState() {
		StringBuilder b = new StringBuilder();
		b.append(String.format("omegaBias_%d", currentView));
		for (int z = 0; z < Z; z++) {
			b.append(String.format(" %.3f", omegaBias[z]));
		}
		
		Log.info("phiPrior_" + currentView + " iteration", b.toString());
	}
	
}
