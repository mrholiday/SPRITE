package main;

import java.io.File;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

import cslda.CSLDA;
import cslda.CSLDAPred;

import models.original.DMR;
import models.original.DMRPred;
import models.original.SpriteICWSM;
import models.original.SpriteICWSMNoSupervisedOmega;
import models.original.SpriteICWSMNoSupervisedOmegaFixedMean;
import models.original.SpriteICWSMNoSupervisedOmegaPred;
import models.original.SpriteICWSMPred;
import models.original.SpriteJoint;
import models.original.SpriteJointThreeFactor;
import models.original.SpriteJointThreeFactorPred;
import models.original.SpriteLDA;
import models.original.SpriteLDAPred;
import models.original.SpriteUnsupervised;
import models.original.SpriteUnsupervisedPred;
import models.original.threefields.DMROmegaThreeFields;
import models.original.threefields.DMRPredOmegaThreeFields;
import models.original.threefields.DMRPredThreeFields;
import models.original.threefields.DMRThreeFields;
import models.original.threefields.SpriteICWSMNoSupervisedOmegaPredThreeFields;
import models.original.threefields.SpriteICWSMNoSupervisedOmegaThreeFields;
import models.original.threefields.SpriteICWSMOmegaInitPredThreeFields;
import models.original.threefields.SpriteICWSMOmegaInitThreeFields;
import models.original.threefields.SpriteICWSMPredThreeFields;
import models.original.threefields.SpriteICWSMThreeFields;
import models.original.threefields.Sprite_DAG;

import utils.MathUtils;
import utils.Log;

public class LearnTopicModel {
	
	private static class CmdArgs {
		@Parameter(names = "-model", description = "Type of topic model to run", required=true)
		private String model;
		
		@Parameter(names="-input", description="Path to training file", required=true)
		String filename;
		
		@Parameter(names="-Z", description="Number of topics", required=true)
		int z;
		
		@Parameter(names="-sigmaAlpha", description="Stddev for component to document assignments.  Defaults to 1.0")
		double sigmaAlpha = 1.0;
		
		@Parameter(names="-sigmaDelta", description="Stddev for delta.  Defaults to 1.0")
		double sigmaDelta = 1.0;
		
		@Parameter(names="-sigmaDeltaB", description="Stddev for delta bias.  Defaults to 1.0")
		double sigmaDeltaBias = 1.0;
		
		@Parameter(names="-sigmaOmega", description="Stddev for omega.  Defaults to 1.0")
		double sigmaOmega = 1.0;

		@Parameter(names="-sigmaOmegaB", description="Stddev for omega bias.  Defaults to 10.0")
		double sigmaOmegaBias = 10.0;

		@Parameter(names="-deltaB", description="Initial value for delta bias (on \\widetilde{\\theta}).  Defaults to -5.0")
		double deltaBias = -5.0;
		
		@Parameter(names="-omegaB", description="Initial value for omega bias (on \\widetilde{\\phi}).  Defaults to -5.0")
		double omegaBias = -5.0;
		
		@Parameter(names="-step", description="Master step size.  Defaults to 0.01")
		double step = 0.01;

		@Parameter(names="-Cth", description="Number of components across \\theta's factors.  Defaults to 11.")
		int Cth = 11;

		@Parameter(names="-Cph", description="Number of components across \\phi's factors.  Defaults to 11.")
		int Cph = 11;
		
		@Parameter(names="-sigmaCSLDA", description="Lower -> CSLDA favors topics that better predict labels")
		double sigmaCSLDA = 10.0;
		
		//@Parameter(names="-temper", description="Temperature constant")
		//double temper = 1.0;
		
		@Parameter(names="-seed", description="Random seed.  Default is to use machine clock time")
		int seed = -1;

		@Parameter(names="-likelihoodFreq", description="How often to print out likelihood/perplexity.  Setting less than 1 will disable printing.  Defaults to every 100 iterations.")
		int likelihoodFreq = 100;

		@Parameter(names="-nthreads", description="Number of threads that will sample and take gradient steps in parallel.  Defaults to 1.")
		int numThreads = 1;
		
		@Parameter(names="-computePerplexity", description="Whether held-out perplexity should be computed on half the data (other half used for training)")
		boolean computePerplexity = false;

		// The fold to assign topic vectors to (excluded from training)
		// This is for the case where we want to predict document scores using inferred topic vectors
		@Parameter(names="-predFold", description="Which fold to exclude from training.  Used for the prediction task.  If non-negative, input file should be in the correct format for the prediction task (second field is the fold index per document)")
		int predFold = -1;
		
		@Parameter(names="-iters", description="Number of iterations for training total.  Defaults to 5000")
		int iters = 5000;
		
		@Parameter(names="-samples", description="Number of samples to take for the final estimate.  Defaults to 100")
		int samples = 100;
		
		@Parameter(names="-logPath", description="Where to log the training output.  Defaults to stdout")
		String logPath = null;
		
		@Parameter(names="-omegaPath", description="Path to already-initialized omega to load")
		String omegaPath = null;
		
		@Parameter(names = "--help", help = true)
		private boolean help;
	}
	
	private static void init(int seed, String logPath) {
		if (seed == -1) {
			MathUtils.initRandomStream();
		}
		else {
			MathUtils.initRandomStream(seed);
		}
		
		if (logPath != null) {
			Log.initFileLogger(logPath);
		}
	}
	
	public static void main(String[] args) throws Exception {
		TopicModel topicModel = null;
		
		CmdArgs c = new CmdArgs();
		new JCommander(c, args);
		
		if (c.model.equals("sprite")) {
			//topicModel = new SpriteJoint(z, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ, stepSizeAZ, stepSizeAB, stepSizeW,
			//		                     stepSizeWB, stepSizeB, delta0, delta1, deltaB, omegaB, likelihoodFreq, priorPrefix,
			//		                     stepA, Cth, Cph, seed, numThreads);
			if (c.predFold >= 0) {
				// -1 or "" are for arguments that were obligatory but never used...
				topicModel = new SpriteICWSMPred(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1,
						-1, -1, -1, -1, -1, -1, -1, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.predFold);
			}
			else {
				//topicModel = new SpriteJointThreeFactor(z, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ, stepSizeAZ, stepSizeAB, stepSizeW,
				//		                     stepSizeWB, stepSizeB, delta0, delta1, deltaB, omegaB, likelihoodFreq, priorPrefix,
				//		                     stepA, Cth, Cph, seed, numThreads, computePerplexity);
				topicModel = new SpriteICWSM(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("sprite_joint")) {
			topicModel = new SpriteJoint(c.z, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega, c.sigmaOmegaBias, c.step, c.step,
					                     c.step, c.step, c.step, c.step, -1, -1, c.deltaBias, c.omegaBias,
					                     c.likelihoodFreq, "sprite_joint", c.step, c.Cth, c.Cph, c.seed, c.numThreads);
		}
		else if (c.model.equals("cslda")) {
			if (c.predFold < 0) {
				topicModel = new CSLDA(c.z, c.deltaBias, c.omegaBias, c.sigmaCSLDA, c.step, c.sigmaDeltaBias,
						c.sigmaOmegaBias, c.computePerplexity); 
			}
			else {
				topicModel = new CSLDAPred(c.z, c.deltaBias, c.omegaBias, c.sigmaCSLDA, c.step, c.sigmaDeltaBias,
						c.sigmaOmegaBias, c.predFold); 
			}
		}
		else if (c.model.equals("sprite_lda")) {
			if (c.predFold >= 0) {
				  topicModel = new SpriteLDAPred(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
							c.sigmaOmegaBias, c.deltaBias,
							c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.predFold);
			}
			else {
				topicModel = new SpriteLDA(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
							c.sigmaOmegaBias, c.deltaBias,
							c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("sprite_unsupervised")) {
			if (c.predFold >= 0) {
				  topicModel = new SpriteUnsupervisedPred(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
							c.sigmaOmegaBias, -1,
							-1, -1, -1, -1, -1, -1, -1, c.deltaBias,
							c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.predFold);
			}
			else {
				  topicModel = new SpriteUnsupervised(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
							c.sigmaOmegaBias, -1,
							-1, -1, -1, -1, -1, -1, -1, c.deltaBias,
							c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("dmr")) {
			if (c.predFold >= 0) {
				// -1 or "" are for arguments that were obligatory but never used...
				topicModel = new DMRPred(c.z, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, 3, 3, c.seed, c.numThreads, c.predFold);
			}
			else {
				topicModel = new DMR(c.z, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, 3, 3, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("sprite_icwsm_noomega")) {
			if (c.predFold >= 0) {
				// -1 or "" are for arguments that were obligatory but never used...
				topicModel = new SpriteICWSMNoSupervisedOmegaPred(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1,
						-1, -1, -1, -1, -1, -1, -1, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.predFold);
			}
			else {
				topicModel = new SpriteICWSMNoSupervisedOmega(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("sprite_icwsm_noomega_fixedmean")) {
			topicModel = new SpriteICWSMNoSupervisedOmegaFixedMean(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
					c.sigmaOmegaBias, c.deltaBias,
					c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.computePerplexity);
		}
		else if (c.model.equals("sprite_3factor")) {
			if (c.predFold >= 0) {
				// -1 or "" are for arguments that were obligatory but never used...
				topicModel = new SpriteJointThreeFactorPred(c.z, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, 3, 3, c.seed, c.numThreads, c.predFold);
			}
			else {
				topicModel = new SpriteJointThreeFactor(c.z, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, 3, 3, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("dmr_threefields")) {
			if (c.predFold >= 0) {
				// -1 or "" are for arguments that were obligatory but never used...
				topicModel = new DMRPredThreeFields(c.z, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, 3, 3, c.seed, c.numThreads, c.predFold);
			}
			else {
				topicModel = new DMRThreeFields(c.z, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, 3, 3, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("dmr_omega_threefields")) {
			if (c.predFold >= 0) {
				// -1 or "" are for arguments that were obligatory but never used...
				topicModel = new DMRPredOmegaThreeFields(c.z, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, 3, 3, c.seed, c.numThreads, c.predFold);
			}
			else {
				topicModel = new DMROmegaThreeFields(c.z, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, 3, 3, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("sprite_threefields")) {
			//topicModel = new SpriteJoint(z, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ, stepSizeAZ, stepSizeAB, stepSizeW,
			//		                     stepSizeWB, stepSizeB, delta0, delta1, deltaB, omegaB, likelihoodFreq, priorPrefix,
			//		                     stepA, Cth, Cph, seed, numThreads);
			if (c.predFold >= 0) {
				// -1 or "" are for arguments that were obligatory but never used...
				topicModel = new SpriteICWSMPredThreeFields(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1,
						-1, -1, -1, -1, -1, -1, -1, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.predFold);
			}
			else {
				//topicModel = new SpriteJointThreeFactor(z, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ, stepSizeAZ, stepSizeAB, stepSizeW,
				//		                     stepSizeWB, stepSizeB, delta0, delta1, deltaB, omegaB, likelihoodFreq, priorPrefix,
				//		                     stepA, Cth, Cph, seed, numThreads, computePerplexity);
				topicModel = new SpriteICWSMThreeFields(c.z, c.sigmaAlpha, c.sigmaDelta,
						c.sigmaDeltaBias, c.sigmaOmega, c.sigmaOmegaBias, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("sprite_omegaInit_threefields")) {
			//topicModel = new SpriteJoint(z, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ, stepSizeAZ, stepSizeAB, stepSizeW,
			//		                     stepSizeWB, stepSizeB, delta0, delta1, deltaB, omegaB, likelihoodFreq, priorPrefix,
			//		                     stepA, Cth, Cph, seed, numThreads);
			if (c.predFold >= 0) {
				// -1 or "" are for arguments that were obligatory but never used...
				topicModel = new SpriteICWSMOmegaInitPredThreeFields(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1,
						-1, -1, -1, -1, -1, -1, -1, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.predFold, c.omegaPath);
			}
			else {
				//topicModel = new SpriteJointThreeFactor(z, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ, stepSizeAZ, stepSizeAB, stepSizeW,
				//		                     stepSizeWB, stepSizeB, delta0, delta1, deltaB, omegaB, likelihoodFreq, priorPrefix,
				//		                     stepA, Cth, Cph, seed, numThreads, computePerplexity);
				topicModel = new SpriteICWSMOmegaInitThreeFields(c.z, c.sigmaAlpha, c.sigmaDelta,
						c.sigmaDeltaBias, c.sigmaOmega, c.sigmaOmegaBias, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.computePerplexity, c.omegaPath);
			}
		}
		else if (c.model.equals("sprite_icwsm_noomega_threefields")) {
			if (c.predFold >= 0) {
				// -1 or "" are for arguments that were obligatory but never used...
				topicModel = new SpriteICWSMNoSupervisedOmegaPredThreeFields(c.z, c.sigmaAlpha, c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega,
						c.sigmaOmegaBias, -1,
						-1, -1, -1, -1, -1, -1, -1, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.predFold);
			}
			else {
				topicModel = new SpriteICWSMNoSupervisedOmegaThreeFields(c.z, c.sigmaAlpha,
						c.sigmaDelta, c.sigmaDeltaBias, c.sigmaOmega, c.sigmaOmegaBias, c.deltaBias,
						c.omegaBias, c.likelihoodFreq, "", c.step, c.seed, c.numThreads, c.computePerplexity);
			}
		}
		else if (c.model.equals("sprite_full_dag")) {
			topicModel = new Sprite_DAG(c.z, c.sigmaDelta, c.sigmaDeltaBias,
					c.sigmaOmega, c.sigmaOmegaBias, 0.01, 0.01, 0.01, 0.01, 0.01,
					0.01, -1.0, -1.0, c.deltaBias, c.omegaBias, c.likelihoodFreq,
					"full_dag", 0.01, c.Cth, c.Cph);
		}
		else {
			System.out.println("Invalid model specification. Options: sprite sprite_lda sprite_unsupervised sprite_icwsm_noomega dmr sprite_3factor dmr_threefields dmr_omega_threefields sprite_threefields sprite_icwsm_noomega_threefields");
			return;
		}
		
//		topicModel.likelihoodFreq = c.likelihoodFreq;
		
		// Initializes random stream and logger
		init(c.seed, c.logPath);
		
		// Write output to a directory directly underneath the input directory
		String inDir = new File(c.filename).getParent();
		File outputDirFile = new File(inDir,
				String.format("%s_%d_%d_%.2f_%.2f_%.2f_%.2f_%.3f_%.3f_%d_%d_output", c.model, c.Cth, c.z,
						   c.step, c.sigmaAlpha, c.deltaBias, c.omegaBias, c.sigmaDeltaBias, c.sigmaOmegaBias,
						   c.computePerplexity ? 1 : 0, c.predFold));
		String outputDir = outputDirFile.getAbsolutePath();
		
		outputDirFile.mkdir();
		
		System.out.println("Likelihood freq: " + topicModel.likelihoodFreq);
		topicModel.train(c.iters, c.samples, c.filename, outputDir, c.likelihoodFreq);
	}

}
