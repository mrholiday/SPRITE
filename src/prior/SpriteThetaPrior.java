package prior;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;

import utils.Log;
import utils.MathUtils;
import models.factored.Factor;

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
	
	private int[] views; // Views this prior is responsible for
//	private int currentView; // View this prior over theta is responsible for.
	
	private double[] gradientDeltaBias;
	private double[] adaDeltaBias;
	private double   sigmaDeltaBias;
	
	public void writeDeltaBias(BufferedWriter bw) throws IOException {
		for (int z = 0; z < Z; z++) {
			bw.write(String.format("%d %f\n", z, deltaBias[z]));
		}
	}
	
	public SpriteThetaPrior(Factor[] factors0, int Z0, int[] views0, double initDeltaBias0, double sigmaDeltaBias0) {
		factors = factors0;
		Z = Z0;
		initDeltaBias = initDeltaBias0;
		views = views0;
		sigmaDeltaBias = sigmaDeltaBias0;
	}
	
	public void initialize(int D0) {
		D = D0;
		
		deltaBias = new double[Z];
		gradientDeltaBias = new double[Z];
		adaDeltaBias = new double[Z];
		for (int i = 0; i < Z; i++) {
			deltaBias[i] = initDeltaBias;
		}
		
		thetaNorm = new double[D];
		thetaTilde = new double[D][Z];
		
		updateThetaTilde();
		updateThetaNorm();
	}
	
	public void initializeNewCorpus(int D0) {
		D = D0;
		
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
	public void updateGradient(int z, int v, int d, int docCount, int docTopicCount, Integer docLock) {
		double priorDZ    = thetaTilde[d][z];
		double thetaNormZ = thetaNorm[z];
		
		double dg1  = MathUtils.digamma(thetaNormZ + MathUtils.eps);
		double dg2  = MathUtils.digamma(thetaNormZ + docCount + MathUtils.eps);
		double dgW1 = MathUtils.digamma(priorDZ + docTopicCount + MathUtils.eps);
		double dgW2 = MathUtils.digamma(priorDZ + MathUtils.eps);
		
		double gradientTerm = priorDZ * (dg1-dg2+dgW1-dgW2);
		
		synchronized(docLock) {
			for (Factor f : factors) {
				f.updateThetaGradient(gradientTerm, z, v, d);
			}
			
			if (v == views[0]) { // Only update this once for all views it is associated with
				gradientDeltaBias[z] += gradientTerm;
				gradientDeltaBias[z] += -(deltaBias[z]) / (Math.pow(sigmaDeltaBias, 2) * D); // Regularize \delta^{BIAS}
			}
		}
	}
	
	public void clearGradient(int v, int minZ, int maxZ) {
		if (v == views[0]) {
			for (int z = minZ; z < maxZ; z++) {
				gradientDeltaBias[z] = 0.;
			}
		}
	}
	
	/**
	 * For applying to a new corpus
	 * 
	 * @param z Topic
	 * @param v View
	 * @param d Document
	 * @param docCount Number of samples for document d
	 * @param docTopicCount Number of samples of topic z for document d
	 */
	public void updateAlphaGradient(int z, int v, int d, int docCount, int docTopicCount, Integer docLock) {
		double priorDZ    = thetaTilde[d][z];
		double thetaNormZ = thetaNorm[z];
		
		double dg1  = MathUtils.digamma(thetaNormZ + MathUtils.eps);
		double dg2  = MathUtils.digamma(thetaNormZ + docCount + MathUtils.eps);
		double dgW1 = MathUtils.digamma(priorDZ + docTopicCount + MathUtils.eps);
		double dgW2 = MathUtils.digamma(priorDZ + MathUtils.eps);
		
		double gradientTerm = priorDZ * (dg1-dg2+dgW1-dgW2);
		
		synchronized(docLock) {
			for (Factor f : factors) {
				f.updateAlphaGradient(gradientTerm, z, v, d);
			}
		}
	}
	
	public void doGradientStep(int v, int minZ, int maxZ, double stepSize) {
//		StringBuilder b = new StringBuilder();
//		for (int z = minZ; z < maxZ; z++) {
//			b.append(String.format("%d:%.3e,", z, deltaBias[z]));
//		}
//		Log.info(String.format("thetaPrior_%d", currentView), "Gradient \\delta^{BIAS}: " + b.toString());
		
		if (v == views[0]) {
			for (int z = minZ; z < maxZ; z++) {
				// gradientDeltaBias[z] += -(deltaBias[z]) / Math.pow(sigmaDeltaBias, 2);  // This is rightly done in updateGradient now.
				adaDeltaBias[z] += Math.pow(gradientDeltaBias[z], 2);
				deltaBias[z] += (stepSize / (Math.sqrt(adaDeltaBias[z]) + MathUtils.eps)) * gradientDeltaBias[z];
			}
		}
	}
	
	// Returns the phi_dz prior given all the parameters.  Factors are
	// responsible for computing their portion of the sums.
	private double priorDZ(int d, int z) {
		double weight = deltaBias[z];
		
		for (Factor f : factors) {
			for (int i = 0; i < views.length; i++) {
				weight += f.getPriorTheta(views[i], d, z);
			}
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
	public void updatePrior(int v, int minD, int maxD) {
		updateThetaTilde(v, minD, maxD);
		updateThetaNorm(v, minD, maxD);
	}
	
	private void updateThetaTilde(int v, int minD, int maxD) {
		if (v == views[0]) {
			for (int d = minD; d < maxD; d++) {
				for (int z = 0; z < Z; z++) {
					thetaTilde[d][z] = priorDZ(d, z);
				}
			}
		}
	}
	
	/**
	 * To debug LinkedFactor -- hold ThetaTilde fixed to make sure omega distributions are being learned properly
	 * 
	 */
	/*
	private void updateThetaTilde(int minD, int maxD) {
		for (int d = minD; d < maxD; d++) {
			for (int z = 0; z < Z; z++) {
				thetaTilde[d][z] = 0.1;
			}
		}
	}
	*/
	
	private void updateThetaNorm(int v, int minD, int maxD) {
		if (v == views[0]) {
			for (int d = minD; d < maxD; d++) {
				thetaNorm[d] = 0.;
				for (int z = 0; z < Z; z++) {
					thetaNorm[d] += thetaTilde[d][z];
				}
			}
		}
	}
	
	private void updateThetaTilde() {
		updateThetaTilde(views[0], 0, Z);
	}
	
	private void updateThetaNorm() {
		updateThetaNorm(views[0], 0, Z);
	}
	
	/**
	 * Logs bias term for this iteration.
	 */
	public void logState() {
		StringBuilder b = new StringBuilder();
		
		for (int i = 0; i < views.length; i++) {
			b.append(String.format("deltaBias_%d", views[i]));
			for (int z = 0; z < Z; z++) {
				b.append(String.format(" %.3f", deltaBias[z]));
			}
			
			Log.info("thetaPrior_" + views[i] + "_iteration", b.toString());
			b.delete(0, b.length());
		}
	}
	
}
