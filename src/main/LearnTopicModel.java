package main;

import java.util.HashMap;

public class LearnTopicModel {
	
	public static HashMap<String,String> arguments;
	
	public static void main(String[] args) throws Exception {
		arguments = new HashMap<String,String>();
		
		for (int i = 0; i < args.length; i += 2) {
			arguments.put(args[i], args[i+1]);
		}
		
		String model = arguments.get("-model");
		String filename = arguments.get("-input");
		
		if (model == null) {
			System.out.println("No model specified.");
			return;
		}
		
		if (filename == null) {
			System.out.println("No input file given.");
			return;
		}
		
		TopicModel topicModel = null;

			if (!arguments.containsKey("-Z")) {
				System.out.println("Must specify number of topics using -Z");
				return;
			}
			
			int z = Integer.parseInt(arguments.get("-Z"));
			
			double sigmaAlpha = 1.0;
			if (arguments.containsKey("-sigmaAlpha"))
				sigmaAlpha = Double.parseDouble(arguments.get("-sigmaAlpha"));
			
			double sigmaA = 1.0;
			if (arguments.containsKey("-sigmaA")) 
				sigmaA = Double.parseDouble(arguments.get("-sigmaA"));
			double sigmaAB = 1.0;
			if (arguments.containsKey("-sigmaAB")) 
				sigmaAB = Double.parseDouble(arguments.get("-sigmaAB"));
			double sigmaW = 1.0;
			if (arguments.containsKey("-sigmaW")) 
				sigmaW = Double.parseDouble(arguments.get("-sigmaW"));
			double sigmaWB = 10.0;
			if (arguments.containsKey("-sigmaWB")) 
				sigmaWB = Double.parseDouble(arguments.get("-sigmaWB"));
			double delta0 = 0.1;
			if (arguments.containsKey("-delta0")) 
				delta0 = Double.parseDouble(arguments.get("-delta0"));
			double delta1 = 0.1;
			if (arguments.containsKey("-delta1")) 
				delta1 = Double.parseDouble(arguments.get("-delta1"));
			double deltaB = -5.0;
			if (arguments.containsKey("-deltaB")) 
				deltaB = Double.parseDouble(arguments.get("-deltaB"));
			double omegaB = -5.0;
			if (arguments.containsKey("-omegaB")) 
				omegaB = Double.parseDouble(arguments.get("-omegaB"));
			double stepSizeADZ = 1e-2;
			if (arguments.containsKey("-stepSizeADZ")) 
				stepSizeADZ = Double.parseDouble(arguments.get("-stepSizeADZ"));
			double stepSizeAZ = stepSizeADZ / 100.0;
			if (arguments.containsKey("-stepSizeAZ")) 
				stepSizeAZ = Double.parseDouble(arguments.get("-stepSizeAZ"));
			double stepSizeAB = stepSizeADZ / 100.0;
			if (arguments.containsKey("-stepSizeAB")) 
				stepSizeAB = Double.parseDouble(arguments.get("-stepSizeAB"));
			double stepSizeW = 1e-3;
			if (arguments.containsKey("-stepSizeW")) 
				stepSizeW = Double.parseDouble(arguments.get("-stepSizeW"));
			double stepSizeWB = stepSizeW / 100.0;
			if (arguments.containsKey("-stepSizeWB")) 
				stepSizeWB = Double.parseDouble(arguments.get("-stepSizeWB"));
			double stepSizeB = 1e-3;
			if (arguments.containsKey("-stepSizeB")) 
				stepSizeB = Double.parseDouble(arguments.get("-stepSizeB"));

			double stepA = 0.1;
			if (arguments.containsKey("-step")) 
				stepA = Double.parseDouble(arguments.get("-step"));
			int Cth = 11;
			if (arguments.containsKey("-Cth")) 
				Cth = Integer.parseInt(arguments.get("-Cth"));
			int Cph = 11;
			if (arguments.containsKey("-Cph")) 
				Cph = Integer.parseInt(arguments.get("-Cph"));
			double temper = 1.0;
			if (arguments.containsKey("-temper")) 
				temper = Double.parseDouble(arguments.get("-temper"));
			int seed = -1;
			if (arguments.containsKey("-seed")) 
				seed = Integer.parseInt(arguments.get("-seed"));

			String priorPrefix = "";
			if (arguments.containsKey("-priorPrefix")) 
				priorPrefix = arguments.get("-priorPrefix");

			int likelihoodFreq = 100;
			if (arguments.containsKey("-likelihoodFreq")) 
				likelihoodFreq = Integer.parseInt(arguments.get("-likelihoodFreq"));
			if (likelihoodFreq == 0) likelihoodFreq = 1;
			else if (likelihoodFreq == -1) likelihoodFreq = Integer.MAX_VALUE; 
			else if (likelihoodFreq < -1) {
				System.out.println("Invalid value for likelihoodFreq; must be positive");
				return;	
			}
			
			int numThreads = 1;
			if (arguments.containsKey("-nthreads")) 
				numThreads = Integer.parseInt(arguments.get("-nthreads"));
			boolean computePerplexity = false;
			if (arguments.containsKey("-computePerplexity")) 
				computePerplexity = Boolean.parseBoolean(arguments.get("-computePerplexity"));
			
			// The fold to assign topic vectors to (excluded from training)
			// This is for the case where we want to predict document scores using inferred topic vectors
			int predFold = -1;
			if (arguments.containsKey("-predFold"))
				predFold = Integer.parseInt(arguments.get("-predFold"));
			
		if (model.equals("sprite")) {
			//topicModel = new SpriteJoint(z, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ, stepSizeAZ, stepSizeAB, stepSizeW,
			//		                     stepSizeWB, stepSizeB, delta0, delta1, deltaB, omegaB, likelihoodFreq, priorPrefix,
			//		                     stepA, Cth, Cph, seed, numThreads);
			if (predFold >= 0) {
				topicModel = new SpriteICWSMPred(z, sigmaAlpha, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ,
					stepSizeAZ, stepSizeAB, stepSizeW, stepSizeWB, stepSizeB, delta0, delta1, deltaB,
					omegaB, likelihoodFreq, priorPrefix, stepA, seed, numThreads, predFold);
			}
			else {
				//topicModel = new SpriteJointThreeFactor(z, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ, stepSizeAZ, stepSizeAB, stepSizeW,
				//		                     stepSizeWB, stepSizeB, delta0, delta1, deltaB, omegaB, likelihoodFreq, priorPrefix,
				//		                     stepA, Cth, Cph, seed, numThreads, computePerplexity);
				topicModel = new SpriteICWSM(z, sigmaAlpha, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ,
						stepSizeAZ, stepSizeAB, stepSizeW, stepSizeWB, stepSizeB, delta0, delta1, deltaB,
						omegaB, likelihoodFreq, priorPrefix, stepA, seed, numThreads, computePerplexity);
			}
		}
		
		else if (model.equals("sprite_lda")) {
			if (predFold >= 0) {
				  topicModel = new SpriteLDAPred(z, sigmaAlpha, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ,
							stepSizeAZ, stepSizeAB, stepSizeW, stepSizeWB, stepSizeB, delta0, delta1, deltaB,
							omegaB, likelihoodFreq, priorPrefix, stepA, seed, numThreads, predFold);
			}
			else {
				  topicModel = new SpriteLDA(z, sigmaAlpha, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ,
							stepSizeAZ, stepSizeAB, stepSizeW, stepSizeWB, stepSizeB, delta0, delta1, deltaB,
							omegaB, likelihoodFreq, priorPrefix, stepA, seed, numThreads, computePerplexity);
			}
		}
		else if (model.equals("sprite_unsupervised")) {
			if (predFold >= 0) {
				  topicModel = new SpriteUnsupervisedPred(z, sigmaAlpha, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ,
							stepSizeAZ, stepSizeAB, stepSizeW, stepSizeWB, stepSizeB, delta0, delta1, deltaB,
							omegaB, likelihoodFreq, priorPrefix, stepA, seed, numThreads, predFold);
			}
			else {
				  topicModel = new SpriteUnsupervised(z, sigmaAlpha, sigmaA, sigmaAB, sigmaW, sigmaWB, stepSizeADZ,
							stepSizeAZ, stepSizeAB, stepSizeW, stepSizeWB, stepSizeB, delta0, delta1, deltaB,
							omegaB, likelihoodFreq, priorPrefix, stepA, seed, numThreads, computePerplexity);
			}
		}
		else {
			System.out.println("Invalid model specification. Options: sprite sprite_lda sprite_unsupervised");
			return;
		}
		
		int iters = 5000;
		if (arguments.containsKey("-iters")) 
			iters = Integer.parseInt(arguments.get("-iters"));
		int samples = 100;
		if (arguments.containsKey("-samples")) 
			samples = Integer.parseInt(arguments.get("-samples"));
		
		topicModel.train(iters, samples, filename);
	}

}
