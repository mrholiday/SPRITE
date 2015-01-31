package prior;

import java.io.Serializable;

import utils.Log;
import main.Factor;

/**
 * Keeps track of \widetilde{\theta}.  Its factors are responsible for
 * calculating the partial sums.
 * 
 * @author adrianb
 *
 */
public class SpriteThetaPrior implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 4886307604396203824L;

	// The factors that feed into this prior.
	private Factor[] factors;
	
	public int Z; // Number of topics
	public int D; // Number of documents
	
	// Bias term.  Assuming this is ever-present.
	private double[] deltaBias;
	private double initDeltaBias; // Initial value for deltaBias
	
	// Sum over documents for each topic.
	private double[] thetaNorm;
	
	// Document -> Topic -> weight.  \widetilde{theta} in TACL paper
	private double[][] thetaTilde;
	
	private int currentView; // View this prior over theta is responsible for.
	
	public SpriteThetaPrior(Factor[] factors0, int Z0, int D0, int currentView0, double initDeltaBias0) {
		factors = factors0;
		Z = Z0;
		D = D0;
		initDeltaBias = initDeltaBias0;
		currentView = currentView0;
		
		initialize();
	}
	
	private void initialize() {
		deltaBias = new double[Z];
		for (int i = 0; i < Z; i++) {
			deltaBias[i] = initDeltaBias;
		}
		
		thetaNorm = new double[D];
		thetaTilde = new double[D][Z];
		
		updateThetaTilde();
		updateThetaNorm();
	}
	
	// Returns the phi_zw prior given all the parameters.  Factors are
	// responsible for computing their portion of the sums.
	private double priorDZ(int d, int z) {
		double weight = deltaBias[z];
		
		for (Factor f : factors) {
			weight += f.getPriorTheta(currentView, d, z);
		}
		
		return Math.exp(weight);
	}
	
	/**
	 * Updates the topic distribution prior for a range of documents.
	 * Updates both \widetilde{\theta} and the sum across topics.
	 * 
	 * @param minD Minimum document index
	 * @param maxD Maximum document index
	 */
	public void updatePrior(int minD, int maxD) {
		updateThetaTilde(minD, maxD);
		updateThetaNorm(minD, maxD);
	}
	
	private void updateThetaTilde(int minD, int maxD) {
		for (int d = minD; d < maxD; d++) {
			for (int z = 0; z < Z; z++) {
				thetaTilde[d][z] = priorDZ(d, z);
			}
		}
	}
	
	private void updateThetaNorm(int minD, int maxD) {
		for (int d = minD; d < maxD; d++) {
			thetaNorm[d] = 0.;
			for (int z = 0; z < Z; z++) {
				thetaNorm[d] += thetaTilde[d][z];
			}
		}
	}
	
	private void updateThetaTilde() {
		updateThetaTilde(0, Z);
	}
	
	private void updateThetaNorm() {
		updateThetaNorm(0, Z);
	}
	

	/**
	 * Logs bias term for this iteration.
	 */
	public void logState() {
		StringBuilder b = new StringBuilder();
		b.append(String.format("deltaBias_%d", currentView));
		for (int z = 0; z < Z; z++) {
			b.append(String.format(" %.3f", deltaBias[z]));
		}
		
		Log.info("thetaPrior_" + currentView + " iteration", b.toString());
	}
	
}
