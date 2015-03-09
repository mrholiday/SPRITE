package models.factored.impl;

import models.factored.Factor;
import models.factored.SpriteFactoredTopicModel;
import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;
import utils.ArgParse;
import utils.Log;
import utils.ArgParse.Arguments;
import utils.Tup3;

/**
 * Topic model with single latent supertopic factor spanning two views.
 * Delta/Beta are tied, sparsity is enforced within each view, and
 * Alpha/Beta/Delta are constrained to be positive.
 * 
 * @author adrianb
 *
 */
public class Sprite2ViewSupertopic extends SpriteFactoredTopicModel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -8356557959212672700L;
	
	public Sprite2ViewSupertopic(SpriteThetaPrior[] thetaPriors0,
			SpritePhiPrior[] phiPriors0, Factor[] factors0, int numThreads0,
			double stepSize0) {
		super(thetaPriors0, phiPriors0, factors0, numThreads0, stepSize0);
	}
	
	private static Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> buildPriors(int Z, int C,
				double sigmaDeltaBias, double initDeltaBias, double sigmaOmegaBias, double initOmegaBias,
				double sigmaBeta, double sigmaOmega, double sigmaAlpha, double sigmaDelta) {
		Factor[] factors = new Factor[] {new Factor(C, new int[] {0, 1}, new int[] {Z, Z}, 0.01, true,
										 sigmaBeta, sigmaOmega, sigmaAlpha, sigmaDelta, true,
										 true, true, "supertopic", false, true, true)};
		
		SpriteThetaPrior[] tpriors = {new SpriteThetaPrior(factors, Z, new int[] {0}, initDeltaBias, sigmaDeltaBias, true),
									  new SpriteThetaPrior(factors, Z, new int[] {1}, initDeltaBias, sigmaDeltaBias, true)};
		SpritePhiPrior[]   ppriors = {new SpritePhiPrior(factors, Z, new int[] {0}, initOmegaBias, sigmaOmegaBias, true),
									  new SpritePhiPrior(factors, Z, new int[] {1}, initOmegaBias, sigmaOmegaBias, true)};
		
		return new Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]>(factors, tpriors, ppriors);
	}
	
	public static void main(String[] args) {
		SpriteFactoredTopicModel topicModel = null;
		
		Arguments p = ArgParse.parseArgs(args);
		
		if (p != null) {
			Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> graph = buildPriors(p.z, p.C, p.sigmaDeltaBias, p.deltaBias, p.sigmaOmegaBias,
																					 p.omegaBias, p.sigmaBeta, p.sigmaOmega, p.sigmaAlpha, p.sigmaDelta);
			topicModel = new Sprite2ViewSupertopic(graph._2(), graph._3(), graph._1(), p.numThreads, p.step);
			topicModel.outputDir = p.outDir;
			topicModel.TIME_ITERATIONS = true;
			
			try {
				topicModel.train(p.iters, p.samples, p.filename);
			} catch (Exception e) {
				Log.error("train", "Error in training...", e);
				e.printStackTrace();
			}
		}
	}
	
}
