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
 * Topic model with one single-component perspective factor for one view.
 * 
 * @author adrianb
 *
 */
public class Sprite2ViewTopicPerspective extends SpriteFactoredTopicModel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -8356557959212672700L;
	
	public Sprite2ViewTopicPerspective(SpriteThetaPrior[] thetaPriors0,
			SpritePhiPrior[] phiPriors0, Factor[] factors0, int numThreads0,
			double stepSize0) {
		super(thetaPriors0, phiPriors0, factors0, numThreads0, stepSize0);
	}
	
	private static Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> buildPriors(int Z,
				double sigmaDeltaBias, double initDeltaBias, double sigmaOmegaBias, double initOmegaBias,
				double sigmaBeta, double sigmaOmega, double sigmaAlpha, double sigmaDelta) {
		Factor[] factors = new Factor[] {new Factor(1, new int[] {0, 1}, new int[] {Z, Z}, 1.0, true,
										 sigmaBeta, sigmaOmega, sigmaAlpha, sigmaDelta, false,
										 false, false, "perspective", true)};
		
		SpriteThetaPrior[] tpriors = {new SpriteThetaPrior(factors, Z, new int[] {0}, initDeltaBias, sigmaDeltaBias),
				new SpriteThetaPrior(factors, Z, new int[] {1}, initDeltaBias, sigmaDeltaBias)};
		SpritePhiPrior[]   ppriors = {new SpritePhiPrior(factors, Z, new int[] {0}, initOmegaBias, sigmaOmegaBias),
				new SpritePhiPrior(factors, Z, new int[] {1}, initOmegaBias, sigmaOmegaBias)};
		
		return new Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]>(factors, tpriors, ppriors);
	}
	
	public static void main(String[] args) {
		SpriteFactoredTopicModel topicModel = null;
		
		Arguments p = ArgParse.parseArgs(args);
		
		if (p != null) {
			Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> graph = buildPriors(p.z, p.sigmaDeltaBias, p.deltaBias, p.sigmaOmegaBias,
																					 p.omegaBias, p.sigmaBeta, p.sigmaOmega, p.sigmaAlpha, p.sigmaDelta);
			topicModel = new Sprite2ViewTopicPerspective(graph._2(), graph._3(), graph._1(), p.numThreads, p.step);
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
