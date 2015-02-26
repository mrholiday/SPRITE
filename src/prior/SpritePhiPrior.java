package prior;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.Map;

import utils.Log;
import utils.MathUtils;
import models.factored.Factor;

/**
 * Defines a Dirichlet prior over the vocabulary.  This prior is defined as a
 * weighted sum over components, which are grouped into factors.
 * 
 * @author adrianb
 *
 */
public class SpritePhiPrior implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7782802097754446722L;
	
	// The factors that feed into this prior.
	private Factor[] factors;
	
	public int Z; // Number of topics
	public int W; // Vocabulary size
	
	// Bias term.  Assuming this is ever-present.
	private double[] omegaBias;
	private double initOmegaBias; // Initial value for omegaBias
	
	// Sum over vocabulary for each topic.
	public double[] phiNorm;
	
	// Topic -> Word -> weight.  \widetilde{phi} in TACL paper
	public double[][] phiTilde;
	
	private double[] gradientOmegaBias;
	private double[] adaOmegaBias;
	private double   sigmaOmegaBias;
	
	private int[] views; // The views that \widetilde{\phi} is responsible for.
//	private int currentView; // The view this \widetilde{\phi} is responsible for.
	
	public void writeOmegaBias(BufferedWriter bw, Map<Integer, String> wordMapInv) throws IOException {
		for (int w = 0; w < W; w++) {
			String word = wordMapInv.get(w);
			bw.write(String.format("%s %f\n", word, omegaBias[w]));
		}
	}
	
	public SpritePhiPrior(Factor[] factors0, int Z0, int[] views0, double omegaInitBias0, double sigmaOmegaBias0) {
		factors = factors0;
		Z = Z0;
		views = views0;
//		currentView = currentView0;
		initOmegaBias = omegaInitBias0;
		sigmaOmegaBias = sigmaOmegaBias0;
	}
	
	public void initialize(int W0) {
		W = W0;
		
		omegaBias = new double[W];
		gradientOmegaBias = new double[W];
		adaOmegaBias = new double[W];
		for (int i = 0; i < W; i++) {
			omegaBias[i] = initOmegaBias;
		}
		
		phiNorm = new double[Z];
		phiTilde = new double[Z][W];
		
		updatePhiTilde();
		updatePhiNorm();
	}
	
	public void initializeNewCorpus() {
		updatePhiTilde();
		updatePhiNorm();
	}
	
	// Returns the phi_zw prior given all the parameters.  Factors are
	// responsible for computing their portion of the sums.
	private double priorZW(int z, int w) {
		double weight = omegaBias[w];
		
		double factorWeight = 0.0;
		for (int i = 0; i < views.length; i++) {
			for (Factor f : factors) {
				factorWeight += f.getPriorPhi(views[i], z, w);
			}
		}
		
		//if (w == 1000) {
		//	Log.info(String.format("phiPrior_v%d_z%d_w%d_weight", v, z, 1000),
		//			 String.format("Bias: %.3f, Factor Weight: %.3f", weight, factorWeight));
		//}
		
		weight += factorWeight;
		
		return Math.exp(weight);
	}
	
	/**
	 * Updates the gradient for factor parameters feeding into \widetilde{\phi}
	 * 
	 * @param z Topic
	 * @param v View, updates only for the first view
	 * @param w Word
	 * @param topicCount Number of times topic z was sampled for this view
	 * @param topicWordCount Number of times word w was sampled for topic z of this view
	 * @param wordLock Lock for this thread
	 */
	public void updateGradient(int z, int v, int w, int topicCount, int topicWordCount, Integer wordLock) {
		double priorZW  = phiTilde[z][w];
		double phiNormZ = phiNorm[z];
		
		double dg1  = MathUtils.digamma(phiNormZ + MathUtils.eps);
		double dg2  = MathUtils.digamma(phiNormZ + topicCount + MathUtils.eps);
		double dgW1 = MathUtils.digamma(priorZW  + topicWordCount + MathUtils.eps);
		double dgW2 = MathUtils.digamma(priorZW  + MathUtils.eps);
		
		double gradientTerm = priorZW * (dg1-dg2+dgW1-dgW2);
		
		synchronized(wordLock) {
			for (Factor f : factors) {
				f.updatePhiGradient(gradientTerm, z, v, w);
			}
			
			if (v == views[0]) {
				gradientOmegaBias[w] += gradientTerm;
				gradientOmegaBias[w] += -(omegaBias[w]) / (Math.pow(sigmaOmegaBias, 2) * Z);
			}
		}
	}
	
	public void doGradientStep(int v, int minW, int maxW, double stepSize) {
//		StringBuilder b = new StringBuilder();
//		for (int w = minW; w < maxW; w++) {
//			b.append(String.format("%d:%.3e,", w, omegaBias[w]));
//		}
//		Log.info(String.format("phiPrior_%d", currentView), "Gradient \\omega^{BIAS}: " + b.toString());
		
		if (v == views[0]) {
			for (int w = minW; w < maxW; w++) {
				// gradientOmegaBias[w] += -(omegaBias[w]) / Math.pow(sigmaOmegaBias, 2); // Now done in updateGradient
				adaOmegaBias[w] += Math.pow(gradientOmegaBias[w], 2);
				omegaBias[w] += (stepSize / (Math.sqrt(adaOmegaBias[w]) + MathUtils.eps)) * gradientOmegaBias[w];
			}
		}
	}
	
	public void clearGradient(int v, int minW, int maxW) {
		if (v == views[0]) {
			for (int w = minW; w < maxW; w++) {
				gradientOmegaBias[w] = 0.;
			}
		}
	}
	
	/**
	 * Updates the word distribution prior for a range of documents.
	 * Updates both \widetilde{\phi} and the sum over the vocabulary.
	 * 
	 * @param minZ Minimum topic index
	 * @param maxZ Maximum topic index
	 */
	public void updatePrior(int v, int minZ, int maxZ) {
		updatePhiTilde(v, minZ, maxZ);
		updatePhiNorm(v, minZ, maxZ);
	}
	
	private void updatePhiTilde(int v, int minZ, int maxZ) {
		if (v == views[0]) {
			for (int z = minZ; z < maxZ; z++) {
				for (int w = 0; w < W; w++) {
					phiTilde[z][w] = priorZW(z, w);
				}
			}
		}
	}
	
	private void updatePhiNorm(int v, int minZ, int maxZ) {
		if (v == views[0]) {
			for (int j = minZ; j < maxZ; j++) {
				phiNorm[j] = 0.;
				for (int i = 0; i < W; i++) {
					phiNorm[j] += phiTilde[j][i];
				}
			}
		}
	}
	
	private void updatePhiTilde() {
		updatePhiTilde(views[0], 0, Z);
	}
	
	private void updatePhiNorm() {
		updatePhiNorm(views[0], 0, Z);
	}
	
	/**
	 * Logs bias term for this iteration.
	 */
	public void logState() {
		StringBuilder b = new StringBuilder();
		
		for (int i = 0; i < views.length; i++) {
			b.append(String.format("omegaBias_sample_%d", views[i]));
			int wStepSize = W > 20 ? W/20 : 1;

			for (int w = 0; w < W; w += wStepSize) {
				b.append(String.format(" %d:%.3f", w, omegaBias[w]));
			}
			
			Log.info("phiPrior_" + views[i] + "_iteration", b.toString());
			
			b = new StringBuilder();
			b.append(String.format("phiNorm_%d", views[i]));
			for (int z = 0; z < Z; z++) {
				b.append(String.format(" %.3f", phiNorm[z]));
			}
			
			Log.info("phiPrior_" + views[i] + "_iteration", b.toString());
		}
	}
	
}
