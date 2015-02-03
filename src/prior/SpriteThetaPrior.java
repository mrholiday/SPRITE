package prior;

import java.io.Serializable;

import utils.Log;
import utils.MathUtils;
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
	
	// Sum over topics for each document.
	public double[] thetaNorm;
	
	// Document -> Topic -> weight.  \widetilde{theta} in TACL paper
	public double[][] thetaTilde;
	
	private int currentView; // View this prior over theta is responsible for.
	
	private double[] gradientDeltaBias;
	private double[] adaDeltaBias;
	private double   sigmaDeltaBias;
	
	public SpriteThetaPrior(Factor[] factors0, int Z0, int D0, int currentView0, double initDeltaBias0, double sigmaDeltaBias0) {
		factors = factors0;
		Z = Z0;
		D = D0;
		initDeltaBias = initDeltaBias0;
		currentView = currentView0;
		sigmaDeltaBias = sigmaDeltaBias0;
		
		initialize();
	}
	
	private void initialize() {
		deltaBias = new double[Z];
		gradientDeltaBias = new double[Z];
		for (int i = 0; i < Z; i++) {
			deltaBias[i] = initDeltaBias;
		}
		
		thetaNorm = new double[D];
		thetaTilde = new double[D][Z];
		
		updateThetaTilde();
		updateThetaNorm();
	}
	
	/**
	 * 
	 * @param z Topic
	 * @param v View
	 * @param d Document
	 * @param docCount Number of samples for document d
	 * @param docTopicCount Number of samples of topic z for document d
	 */
	public void updateGradient(int z, int d, int docCount, int docTopicCount, Integer docLock) {
		double priorDZ    = thetaTilde[d][z];
		double thetaNormZ = thetaNorm[z];
		
		double dg1  = MathUtils.digamma0(thetaNormZ + MathUtils.eps);
		double dg2  = MathUtils.digamma0(thetaNormZ + docCount + MathUtils.eps);
		double dgW1 = MathUtils.digamma0(priorDZ + docTopicCount + MathUtils.eps);
		double dgW2 = MathUtils.digamma0(priorDZ + MathUtils.eps);
		
		double gradientTerm = priorDZ * (dg1-dg2+dgW1-dgW2);
		
		synchronized(docLock) {
			for (Factor f : factors) {
				f.updateThetaGradient(gradientTerm, z, currentView, d);
			}
			gradientDeltaBias[z] += gradientTerm;
		}
	}
	
	public void doGradientStep(int minZ, int maxZ, double stepSize) {
		for (int z = minZ; z < maxZ; z++) {
			gradientDeltaBias[z] += -(deltaBias[z]) / Math.pow(sigmaDeltaBias, 2);
			adaDeltaBias[z] += Math.pow(gradientDeltaBias[z], 2);
			deltaBias[z] += (stepSize / (Math.sqrt(adaDeltaBias[z]) + MathUtils.eps)) * gradientDeltaBias[z];
			gradientDeltaBias[z] = 0.;
		}

	}
	
	// Returns the phi_dz prior given all the parameters.  Factors are
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
