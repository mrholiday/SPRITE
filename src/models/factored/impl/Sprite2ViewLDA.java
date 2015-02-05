package models.factored.impl;

import models.factored.Factor;
import models.factored.SpriteFactoredTopicModel;
import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;
import utils.ArgParse;
import utils.ArgParse.Arguments;
import utils.Log;

/**
 * Implementation of LDA with two views.  Testing out multi-view models.
 * The topics learned for each view are not tied together in anyway.
 * 
 * @author adrianb
 *
 */
public class Sprite2ViewLDA extends SpriteFactoredTopicModel {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5262199993040278930L;
	
	public Sprite2ViewLDA(int Z, int numThreads, double stepSize, double sigmaDeltaBias, double sigmaOmegaBias, double initDeltaBias, double initOmegaBias) {
		super(Sprite2ViewLDA.buildThetaPrior(Z, sigmaDeltaBias, initDeltaBias),
			  Sprite2ViewLDA.buildPhiPrior(Z, sigmaOmegaBias, initOmegaBias),
			  new Factor[] {}, numThreads, stepSize);
	}
	
	private static SpriteThetaPrior[] buildThetaPrior(int Z, double sigmaDeltaBias, double initDeltaBias) {
		return new SpriteThetaPrior[] {new SpriteThetaPrior(new Factor[] {}, Z, 0, initDeltaBias, sigmaDeltaBias),
									   new SpriteThetaPrior(new Factor[] {}, Z, 1, initDeltaBias, sigmaDeltaBias)};
	}
	
	private static SpritePhiPrior[] buildPhiPrior(int Z, double sigmaOmegaBias, double initOmegaBias) {
		return new SpritePhiPrior[] {new SpritePhiPrior(new Factor[] {}, Z, 0, initOmegaBias, sigmaOmegaBias),
									 new SpritePhiPrior(new Factor[] {}, Z, 1, initOmegaBias, sigmaOmegaBias)};
	}
	
	public static void main(String[] args) {
		SpriteFactoredTopicModel topicModel = null;
		
		Arguments p = ArgParse.parseArgs(args);
		
		if (p != null) {
			topicModel = new Sprite2ViewLDA(p.z, p.numThreads, p.step, p.sigmaDeltaBias, p.sigmaOmegaBias, p.deltaBias, p.omegaBias);
			topicModel.outputDir = p.outDir;
			topicModel.TIME_ITERATIONS = true;
			
			try {
				topicModel.train(p.iters, p.samples, p.filename);
			} catch (Exception e) {
				Log.error("train", "Error in training...", e);
			}
		}
	}
	
}
