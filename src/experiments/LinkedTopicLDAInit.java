package experiments;

import models.factored.Factor;
import models.factored.LinkedFactor;
import models.factored.SpriteFactoredTopicModel;
import models.factored.impl.Sprite2ViewLinkedSharedThetaTopics;
import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;
import utils.ArgParse;
import utils.Log;
import utils.Tup3;
import utils.ArgParse.Arguments;

/**
 * Initializes a linked-topic 2-view SPRITE model with the majority topic
 * sampled for each word by another model (LDA in this case).
 * 
 * @author adrianb
 *
 */
public class LinkedTopicLDAInit {
	
	private static Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> buildPriors(int Z, int C,
				double sigmaDeltaBias, double initDeltaBias, double sigmaOmegaBias, double initOmegaBias,
				double sigmaBeta, double sigmaOmega, double sigmaAlpha, double sigmaDelta) {
		Factor[] factors = new Factor[] {new LinkedFactor(C, new int[] {0, 1}, new int[] {Z, Z}, 1.0, true,
										 sigmaBeta, sigmaOmega, sigmaAlpha, sigmaDelta, true,
										 false, false, "supertopic", false, true, true, 1.0)};
		
		SpriteThetaPrior thetaPrior = new SpriteThetaPrior(factors, Z, new int[] {0}, initDeltaBias, sigmaDeltaBias, true);
		
		SpriteThetaPrior[] tpriors = {thetaPrior, thetaPrior};
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
			topicModel = new Sprite2ViewLinkedSharedThetaTopics(graph._2(), graph._3(), graph._1(), p.numThreads, p.step);
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
