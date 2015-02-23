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
	
	private int currentView; // The view this \widetilde{\phi} is responsible for.
	
	public void writeOmegaBias(BufferedWriter bw, Map<Integer, String> wordMapInv) throws IOException {
		for (int w = 0; w < W; w++) {
			String word = wordMapInv.get(w);
			bw.write(String.format("%s %f\n", word, omegaBias[w]));
		}
	}
	
	public SpritePhiPrior(Factor[] factors0, int Z0, int currentView0, double omegaInitBias0, double sigmaOmegaBias0) {
		factors = factors0;
		Z = Z0;
		currentView = currentView0;
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
	
	// Returns the phi_zw prior given all the parameters.  Factors are
	// responsible for computing their portion of the sums.
	private double priorZW(int v, int z, int w) {
		double weight = omegaBias[w];
		
		double factorWeight = 0.0;
		for (Factor f : factors) {
			factorWeight += f.getPriorPhi(v, z, w);
		}
		
		if (w == 1000) {
			Log.info(String.format("phiPrior_v%d_z%d_w%d_weight", v, z, 1000),
					 String.format("Bias: %.3f, Factor Weight: %.3f", weight, factorWeight));
		}
		
		weight += factorWeight;
		
		return Math.exp(weight);
	}
	
	/**
	 * Updates the gradient for factor parameters feeding into \widetilde{\phi}
	 * 
	 * @param z Topic
	 * @param w Word
	 * @param topicCount Number of times topic z was sampled for this view
	 * @param topicWordCount Number of times word w was sampled for topic z of this view
	 * @param wordLock Lock for this thread
	 */
	public void updateGradient(int z, int w, int topicCount, int topicWordCount, Integer wordLock) {
		double priorZW  = phiTilde[z][w];
		double phiNormZ = phiNorm[z];
		
		double dg1  = MathUtils.digamma(phiNormZ + MathUtils.eps);
		double dg2  = MathUtils.digamma(phiNormZ + topicCount + MathUtils.eps);
		double dgW1 = MathUtils.digamma(priorZW  + topicWordCount + MathUtils.eps);
		double dgW2 = MathUtils.digamma(priorZW  + MathUtils.eps);
		
		double gradientTerm = priorZW * (dg1-dg2+dgW1-dgW2);
		
		synchronized(wordLock) {
			for (Factor f : factors) {
				f.updatePhiGradient(gradientTerm, z, currentView, w);
			}
			gradientOmegaBias[w] += gradientTerm;
		}
	}
	
	public void doGradientStep(int minW, int maxW, double stepSize) {
		for (int w = minW; w < maxW; w++) {
			gradientOmegaBias[w] += -(omegaBias[w]) / Math.pow(sigmaOmegaBias, 2);
			adaOmegaBias[w] += Math.pow(gradientOmegaBias[w], 2);
			omegaBias[w] += (stepSize / (Math.sqrt(adaOmegaBias[w]) + MathUtils.eps)) * gradientOmegaBias[w];
			gradientOmegaBias[w] = 0.; // Clear gradient for the next iteration
		}
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
		b.append(String.format("omegaBias_sample_%d", currentView));
		int wStepSize = W/20;
		for (int w = 0; w < W; w += wStepSize) {
			b.append(String.format(" %d:%.3f", w, omegaBias[w]));
		}
		
		Log.info("phiPrior_" + currentView + "_iteration", b.toString());
		
		b = new StringBuilder();
		b.append(String.format("phiNorm_%d", currentView));
		for (int z = 0; z < Z; z++) {
			b.append(String.format(" %.3f", phiNorm[z]));
		}
		
		Log.info("phiPrior_" + currentView + "_iteration", b.toString());
	}
	
}
