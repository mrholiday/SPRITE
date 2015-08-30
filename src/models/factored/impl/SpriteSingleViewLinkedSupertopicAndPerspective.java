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
 * Topic model with one single-component perspective factor for one view, and a
 * latent supertopic factor as well.  Just concatenation of factors defined in
 * SpriteSupertopic and SpriteTopicPerspective.
 * 
 * @author adrianb
 *
 */
public class SpriteSingleViewLinkedSupertopicAndPerspective extends SpriteFactoredTopicModel {
	/**
	 * 
	 */
	private static final long serialVersionUID = -8356557959212672700L;
	
	public SpriteSingleViewLinkedSupertopicAndPerspective(SpriteThetaPrior[] thetaPriors0,
			SpritePhiPrior[] phiPriors0, Factor[] factors0, int numThreads0,
			double stepSize0) {
		super(thetaPriors0, phiPriors0, factors0, numThreads0, stepSize0);
	}
	
	private static Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> buildPriors(int Z, int C,
				double sigmaDeltaBias, double initDeltaBias, double sigmaOmegaBias, double initOmegaBias,
				double sigmaBeta, double sigmaOmega, double sigmaAlpha, double sigmaDelta) {
		
		
		Factor[] factors = new Factor[] {
				new Factor(1, new int[] {0}, new int[] {Z}, 1.0, true,
						sigmaBeta, sigmaOmega, sigmaAlpha, sigmaDelta, false,
						false, false, "perspective", true, true, true),
				// This is the only change from the Topic-Perspective model
				new LinkedFactor(C, new int[] {0}, new int[] {Z}, 1.0, true,
						   sigmaBeta, sigmaOmega, sigmaAlpha, sigmaDelta,
						   true, true, true, "supertopic", false, true, true, 1.0)};
		
		SpriteThetaPrior[] tpriors = {new SpriteThetaPrior(factors, Z, new int[] {0}, initDeltaBias, sigmaDeltaBias, true)};
		SpritePhiPrior[]   ppriors = {new SpritePhiPrior(factors, Z, new int[] {0}, initOmegaBias, sigmaOmegaBias, true)};
		
		return new Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]>(factors, tpriors, ppriors);
	}
	
	public static void main(String[] args) {
		SpriteFactoredTopicModel topicModel = null;
		
		Arguments p = ArgParse.parseArgs(args);
		
		if (p != null) {
			Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> graph = buildPriors(p.z, p.C, p.sigmaDeltaBias, p.deltaBias, p.sigmaOmegaBias,
																					 p.omegaBias, p.sigmaBeta, p.sigmaOmega, p.sigmaAlpha, p.sigmaDelta);
			topicModel = new SpriteSingleViewLinkedSupertopicAndPerspective(graph._2(), graph._3(), graph._1(), p.numThreads, p.step);
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
