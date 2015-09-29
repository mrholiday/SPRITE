package models.factored.impl;

import models.factored.Factor;
import models.factored.LinkedFactor;
import models.factored.SpriteFactoredTopicModel;
import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;
import utils.ArgParse;
import utils.Log;
import utils.ArgParse.Arguments;
import utils.Tup3;

/**
 * Topic model with single latent supertopic factor spanning two views.
 * Delta/Beta ensure that topic i in both views are the same (1), and
 * Alpha is constrained to be positive.  \widetilde{\theta} is shared
 * across both views.
 * 
 * @author adrianb
 *
 */
public class Sprite2ViewLinkedSharedThetaTopics extends SpriteFactoredTopicModel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -8356557959212672700L;
	
	/**
	 * WARINING: THIS MAY BE BROKEN -- HAVEN'T TESTED SINGLE VIEW PRIOR ENOUGH.
	 */
	
	public Sprite2ViewLinkedSharedThetaTopics(SpriteThetaPrior[] thetaPriors0,
			SpritePhiPrior[] phiPriors0, Factor[] factors0, int numThreads0,
			double stepSize0) {
		super(thetaPriors0, phiPriors0, factors0, numThreads0, stepSize0);
	}
	
	public Sprite2ViewLinkedSharedThetaTopics(SpriteThetaPrior[] thetaPriors0,
			SpritePhiPrior[] phiPriors0, Factor[] factors0, int numThreads0,
			double stepSize0, double priorWeight0) {
		super(thetaPriors0, phiPriors0, factors0, numThreads0, stepSize0, priorWeight0);
	}
	
	private static Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> buildPriors(int Z, int C,
				double sigmaDeltaBias, double initDeltaBias, double sigmaOmegaBias, double initOmegaBias,
				double sigmaBeta, double sigmaOmega, double sigmaAlpha, double sigmaDelta, double factorWeight0) {
		Factor[] factors = new Factor[] {new LinkedFactor(C, new int[] {0, 1}, new int[] {Z, Z}, 1.0, true,
										 sigmaBeta, sigmaOmega, sigmaAlpha, sigmaDelta, true,
										 false, false, "supertopic", false, true, true, factorWeight0)};
//		Factor[] factors = new Factor[] {new LinkedFactor(C, new int[] {0}, new int[] {Z}, 1.0, true,
//				 sigmaBeta, sigmaOmega, sigmaAlpha, sigmaDelta, true,
//				 false, false, "supertopic", false, 1.0)};
		
		SpriteThetaPrior thetaPrior1 = new SpriteThetaPrior(factors, Z, new int[] {0, 1}, initDeltaBias, sigmaDeltaBias, true);
		//SpriteThetaPrior thetaPrior2 = new SpriteThetaPrior(factors, Z, 1, initDeltaBias, sigmaDeltaBias);
		
		SpriteThetaPrior[] tpriors = {thetaPrior1, thetaPrior1};
		SpritePhiPrior[]   ppriors = {new SpritePhiPrior(factors, Z, new int[] {0}, initOmegaBias, sigmaOmegaBias, true),
									  new SpritePhiPrior(factors, Z, new int[] {1}, initOmegaBias, sigmaOmegaBias, true)};
//		SpriteThetaPrior[] tpriors = {thetaPrior};
//		SpritePhiPrior[]   ppriors = {new SpritePhiPrior(factors, Z, 0, initOmegaBias, sigmaOmegaBias)};
		
		return new Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]>(factors, tpriors, ppriors);
		//return new Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]>(factors, tpriors, ppriors, priorWeight0);
	}
	
	public static void main(String[] args) {
		SpriteFactoredTopicModel topicModel = null;
		
		Arguments p = ArgParse.parseArgs(args);
		
		if (p != null) {
			Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> graph = buildPriors(p.z, p.C, p.sigmaDeltaBias, p.deltaBias, p.sigmaOmegaBias,
																					 p.omegaBias, p.sigmaBeta, p.sigmaOmega, p.sigmaAlpha, p.sigmaDelta,
																					 p.priorWeight);
//			topicModel = new Sprite2ViewLinkedSharedThetaTopics(graph._2(), graph._3(), graph._1(), p.numThreads, p.step);
			topicModel = new Sprite2ViewLinkedSharedThetaTopics(graph._2(), graph._3(), graph._1(), p.numThreads, p.step, p.priorWeight);
			topicModel.outputDir = p.outDir;
			topicModel.TIME_ITERATIONS = true;
			
			try {
				topicModel.train(p.iters, p.samples, p.filename, p.outDir);
			} catch (Exception e) {
				Log.error("train", "Error in training...", e);
				e.printStackTrace();
			}
		}
	}
	
}
