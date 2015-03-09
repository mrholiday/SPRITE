package models.factored.impl;

import models.factored.Factor;
import models.factored.SpriteFactoredTopicModel;
import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;
import utils.ArgParse;
import utils.ArgParse.Arguments;
import utils.Log;

/**
 * To test out new factored SPRITE code.  Implementation of LDA.
 * No factors, single view.
 * 
 * @author adrianb
 *
 */
public class SpriteLDA extends SpriteFactoredTopicModel {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5262199993040278930L;
	
	public SpriteLDA(int Z, int numThreads, double stepSize, double sigmaDeltaBias, double sigmaOmegaBias, double initDeltaBias, double initOmegaBias) {
		super(SpriteLDA.buildThetaPrior(Z, sigmaDeltaBias, initDeltaBias),
			  SpriteLDA.buildPhiPrior(Z, sigmaOmegaBias, initOmegaBias),
			  new Factor[] {}, numThreads, stepSize);
	}
	
	private static SpriteThetaPrior[] buildThetaPrior(int Z, double sigmaDeltaBias, double initDeltaBias) {
		return new SpriteThetaPrior[] {new SpriteThetaPrior(new Factor[] {}, Z, new int[] {0}, initDeltaBias, sigmaDeltaBias, true)};
	}
	
	private static SpritePhiPrior[] buildPhiPrior(int Z, double sigmaOmegaBias, double initOmegaBias) {
		return new SpritePhiPrior[] {new SpritePhiPrior(new Factor[] {}, Z, new int[] {0}, initOmegaBias, sigmaOmegaBias, true)};
	}
	
	public static void main(String[] args) {
		SpriteFactoredTopicModel topicModel = null;
		
		Arguments p = ArgParse.parseArgs(args);
		
		if (p != null) {
			topicModel = new SpriteLDA(p.z, p.numThreads, p.step, p.sigmaDeltaBias, p.sigmaOmegaBias, p.deltaBias, p.omegaBias);
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
