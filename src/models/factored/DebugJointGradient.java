package models.factored;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Random;

import main.SpriteWorker.ThreadCommunication;
import main.SpriteWorker.ThreadCommunication.ThreadCommand;
import models.factored.impl.SpriteSupertopicAndPerspective;
import models.original.SpriteJoint;
import prior.SpritePhiPrior;
import prior.SpriteThetaPrior;
import utils.ArgParse;
import utils.ArgParse.Arguments;
import utils.Log;
import utils.MathUtils;
import utils.Tup3;
import utils.Utils;

/**
 * Debug refactored SPRITE gradient by comparing to old joint model.
 * 
 * @author adrianb
 *
 */
public class DebugJointGradient {

	public DebugJointGradient() { }
	
	private static SpriteSupertopicAndPerspective buildNewModel(Arguments p) {
		SpriteSupertopicAndPerspective tm = null;
		
		Tup3<Factor[], SpriteThetaPrior[], SpritePhiPrior[]> graph = SpriteSupertopicAndPerspective.buildPriors(p.z, p.C - 1, p.sigmaDeltaBias, p.deltaBias, p.sigmaOmegaBias,
																				 p.omegaBias, p.sigmaBeta, p.sigmaOmega, p.sigmaAlpha, p.sigmaDelta);
		tm = new SpriteSupertopicAndPerspective(graph._2(), graph._3(), graph._1(), p.numThreads, p.step);
		
		return tm;
	}
	
	private static SpriteJoint buildOldModel(Arguments p) {
		SpriteJoint tm = null;
		
		tm = new SpriteJoint(p.z, p.sigmaDelta, p.sigmaDeltaBias, p.sigmaOmega, p.sigmaOmegaBias, p.step, p.step,
                p.step, p.step, p.step, p.step, -1, -1, p.deltaBias, p.omegaBias,
                p.likelihoodFreq, "sprite_joint", p.step, p.C, p.C, p.seed, p.numThreads);
		
		return tm;
	}
	
	public static void logParams(String outPath, SpriteJoint oldModel, SpriteSupertopicAndPerspective newModel) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(new File(outPath)));
			
			writer.write("[check_grad deltaBias] " + Utils.mkStr(oldModel.deltaBias) + '\n');
			writer.write("[check_grad omegaBias] " + Utils.mkStr(oldModel.omegaBias) + '\n');
			writer.write("[check_grad new_thetaTilde] " + Utils.mkStr(newModel.thetaPriors[0].thetaTilde) + '\n');
			writer.write("[check_grad old_thetaTilde] " + Utils.mkStr(oldModel.priorDZ) + '\n');
			writer.write("[check_grad new_thetaNorm] " + Utils.mkStr(newModel.thetaPriors[0].thetaNorm) + '\n');
			writer.write("[check_grad old_thetaNorm] " + Utils.mkStr(oldModel.thetaNorm) + '\n');
			writer.write("[check_grad new_phiTilde] " + Utils.mkStr(newModel.phiPriors[0].phiTilde) + '\n');
			writer.write("[check_grad old_phiTilde] " + Utils.mkStr(oldModel.priorZW) + '\n');
			writer.write("[check_grad new_phiNorm] " + Utils.mkStr(newModel.phiPriors[0].phiNorm) + '\n');
			writer.write("[check_grad old_phiNorm] " + Utils.mkStr(oldModel.phiNorm) + '\n');
			writer.write("[check_grad new_nD]" + Utils.mkStr(newModel.nD) + '\n');
			writer.write("[check_grad new_nDZ] " + Utils.mkStr(newModel.nDZ) + '\n');
			writer.write("[check_grad new_nZ] " + Utils.mkStr(newModel.nZ[0]) + '\n');
			writer.write("[check_grad new_nZW] " + Utils.mkStr(newModel.nZW[0]) + '\n');
			writer.write("[check_grad old_nD]" + Utils.mkStr(oldModel.nD) + '\n');
			writer.write("[check_grad old_nDZ] " + Utils.mkStr(oldModel.nDZ) + '\n');
			writer.write("[check_grad old_nZ] " + Utils.mkStr(oldModel.nZ) + '\n');
			writer.write("[check_grad old_nZW] " + Utils.mkStr(oldModel.nZW) + '\n');
			
			writer.write("[check_grad new_perspOmega] " + Utils.mkStr(newModel.factors[0].omega) + '\n');
			writer.write("[check_grad new_perspBeta] " + Utils.mkStr(newModel.factors[0].beta) + '\n');
			writer.write("[check_grad new_perspBetaB] " + Utils.mkStr(newModel.factors[0].betaB) + '\n');
			writer.write("[check_grad new_perspDelta] " + Utils.mkStr(newModel.factors[0].delta) + '\n');
			writer.write("[check_grad new_perspAlpha] " + Utils.mkStr(newModel.factors[0].alpha) + '\n');
			
			writer.write("[check_grad new_hierOmega] " + Utils.mkStr(newModel.factors[1].omega) + '\n');
			writer.write("[check_grad new_hierBeta] " + Utils.mkStr(newModel.factors[1].beta) + '\n');
			writer.write("[check_grad new_hierBetaB] " + Utils.mkStr(newModel.factors[1].betaB) + '\n');
			writer.write("[check_grad new_hierDelta] " + Utils.mkStr(newModel.factors[1].delta) + '\n');
			writer.write("[check_grad new_hierAlpha] " + Utils.mkStr(newModel.factors[1].alpha) + '\n');
			
			writer.write("[check_grad old_omega] " + Utils.mkStr(oldModel.omega) + '\n');
			writer.write("[check_grad old_beta] " + Utils.mkStr(oldModel.beta) + '\n');
			writer.write("[check_grad old_betaB] " + Utils.mkStr(oldModel.betaB) + '\n');
			writer.write("[check_grad old_delta] " + Utils.mkStr(oldModel.delta) + '\n');
			writer.write("[check_grad old_alpha] " + Utils.mkStr(oldModel.alpha) + '\n');
			
			writer.write("[check_grad old_gradientDeltaBias] " + Utils.mkStr(oldModel.gradientDeltaBias) + '\n');
			writer.write("[check_grad old_gradientOmegaBias] " + Utils.mkStr(oldModel.gradientOmegaBias) + '\n');
			writer.write("[check_grad old_gradientOmega] " + Utils.mkStr(oldModel.gradientOmega) + '\n');
			writer.write("[check_grad old_gradientBeta] " + Utils.mkStr(oldModel.gradientBeta) + '\n');
			writer.write("[check_grad old_gradientBetaB] " + Utils.mkStr(oldModel.gradientBetaB) + '\n');
			writer.write("[check_grad old_gradientDelta] " + Utils.mkStr(oldModel.gradientDelta) + '\n');
			writer.write("[check_grad old_gradientAlpha] " + Utils.mkStr(oldModel.gradientAlpha) + '\n');
			
			writer.write("[check_grad new_gradientDeltaBias] " + Utils.mkStr(newModel.thetaPriors[0].gradientDeltaBias) + '\n');
			writer.write("[check_grad new_gradientOmegaBias] " + Utils.mkStr(newModel.phiPriors[0].gradientOmegaBias) + '\n');
			writer.write("[check_grad new_perspGradientOmega] " + Utils.mkStr(newModel.factors[0].gradientOmega) + '\n');
			writer.write("[check_grad new_perspGradientBeta] " + Utils.mkStr(newModel.factors[0].gradientBeta) + '\n');
			writer.write("[check_grad new_perspGradientBetaB] " + Utils.mkStr(newModel.factors[0].gradientBetaB) + '\n');
			writer.write("[check_grad new_perspGradientDelta] " + Utils.mkStr(newModel.factors[0].gradientDelta) + '\n');
			writer.write("[check_grad new_perspGradientAlpha] " + Utils.mkStr(newModel.factors[0].gradientAlpha) + '\n');
			writer.write("[check_grad new_hierGradientOmega] " + Utils.mkStr(newModel.factors[1].gradientOmega) + '\n');
			writer.write("[check_grad new_hierGradientBeta] " + Utils.mkStr(newModel.factors[1].gradientBeta) + '\n');
			writer.write("[check_grad new_hierGradientBetaB] " + Utils.mkStr(newModel.factors[1].gradientBetaB) + '\n');
			writer.write("[check_grad new_hierGradientDelta] " + Utils.mkStr(newModel.factors[1].gradientDelta) + '\n');
			writer.write("[check_grad new_hierGradientAlpha] " + Utils.mkStr(newModel.factors[1].gradientAlpha) + '\n');
			
			writer.flush();
			writer.close();
			
			System.out.println("Params written.");
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void copyCounts(SpriteJoint oldModel, SpriteSupertopicAndPerspective newModel) {
		// Initialize new model with same parameters as the joint model
		newModel.nD = new int[oldModel.D][1];
		for (int d = 0; d < oldModel.D; d++) {
			newModel.nD[d][0] = oldModel.nD[d];
		}
		
		for (int d = 0; d < oldModel.nDZ.length; d++) {
			newModel.nDZ[d][0] = Arrays.copyOf(oldModel.nDZ[d], oldModel.nDZ[d].length);
		}
		
		newModel.nDZ = new int[oldModel.D][1][oldModel.Z];
		for (int d = 0; d < oldModel.D; d++) {
			for (int z = 0; z < oldModel.Z; z++) {
				newModel.nDZ[d][0][z] = oldModel.nDZ[d][z];
			}
		}
		
		newModel.nZ[0] = Arrays.copyOf(oldModel.nZ, oldModel.nZ.length);
		for (int z = 0; z < oldModel.nZW.length; z++) {
			newModel.nZW[0][z] = Arrays.copyOf(oldModel.nZW[z], oldModel.nZW[z].length);
		}
		
		for (int d = 0; d < oldModel.D; d++) {
			newModel.docsZ[d][0] = Arrays.copyOf(oldModel.docsZ[d], oldModel.docsZ[d].length);
			for (int n = 0; n < oldModel.docsZZ[d].length; n++) {
				newModel.docsZZ[d][0][n] = Arrays.copyOf(oldModel.docsZZ[d][n], oldModel.docsZZ[d][n].length);
			}
		}
	}
	
	public static void copyParams(SpriteJoint oldModel, SpriteSupertopicAndPerspective newModel) {
		// Initialize new model with same parameters as the joint model
		copyCounts(oldModel, newModel);
		
		// Bias
		newModel.thetaPriors[0].deltaBias = Arrays.copyOf(oldModel.deltaBias, oldModel.deltaBias.length);
		newModel.phiPriors[0].omegaBias = Arrays.copyOf(oldModel.omegaBias, oldModel.omegaBias.length);
		
		// Perspective
		newModel.factors[0].alpha = new double[oldModel.D][1];
		newModel.factors[0].beta[0] = new double[oldModel.Z][1];
		newModel.factors[0].betaB[0] = new double[oldModel.Z][1];
		newModel.factors[0].delta[0] = new double[1][oldModel.Z];
		newModel.factors[0].omega = new double[1][newModel.W];
		for (int d = 0; d < oldModel.D; d++) {
			newModel.factors[0].alpha[d][0] = oldModel.alpha[d][0];
		}
		for (int z = 0; z < oldModel.Z; z++) {
			newModel.factors[0].beta[0][z][0]  = oldModel.beta[z][0];
			newModel.factors[0].betaB[0][z][0] = 1.0;
			newModel.factors[0].delta[0][0][z] = oldModel.delta[0][z];
		}
		for (int w = 0; w < newModel.W; w++) {
			newModel.factors[0].omega[0][w] = oldModel.omega[0][w];
		}
		
		int C = oldModel.Cth;
		
		// Hierarchy
		newModel.factors[1].alpha = new double[oldModel.D][C-1];
		newModel.factors[1].beta[0] = new double[oldModel.Z][C-1];
		newModel.factors[1].betaB[0] = new double[oldModel.Z][C-1];
		newModel.factors[1].delta[0] = new double[C-1][oldModel.Z];
		newModel.factors[1].omega = new double[C-1][newModel.W];
		for (int d = 0; d < oldModel.D; d++) {
			for (int c = 1; c < C; c++) {
				newModel.factors[1].alpha[d][c-1] = oldModel.alpha[d][c];
			}
		}
		for (int z = 0; z < oldModel.Z; z++) {
			for (int c = 1; c < C; c++) {
				newModel.factors[1].beta[0][z][c-1]  = oldModel.beta[z][c];
				newModel.factors[1].betaB[0][z][c-1] = oldModel.betaB[z][c];
				newModel.factors[1].delta[0][c-1][z] = oldModel.delta[c][z];
			}
		}
		for (int w = 0; w < newModel.W; w++) {
			for (int c = 1; c < C; c++) {
				newModel.factors[1].omega[c-1][w] = oldModel.omega[c][w];
			}
		}
	}
	
	public static void takeStep(SpriteJoint oldModel, SpriteSupertopicAndPerspective newModel) {
		oldModel.clearGradient();
		oldModel.updateGradient(200, 0, oldModel.Z, 0, oldModel.W);
		oldModel.doGradientStep(200, 0, oldModel.Z, 0, oldModel.W, 0, oldModel.D);
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.CLEAR_GRADIENT, newModel.threadName));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.CALC_GRADIENT, newModel.threadName));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.GRADIENT_STEP, newModel.threadName));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.UPDATE_PRIOR, newModel.threadName));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.CLEAR_GRADIENT, newModel.threadName));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.CALC_GRADIENT, newModel.threadName));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		oldModel.updatePriors(0, oldModel.Z, 0, oldModel.D);
		oldModel.updateGradient(200, 0, oldModel.Z, 0, oldModel.W);
	}
	
	public static void updatePriorsAndGrad(SpriteJoint oldModel, SpriteSupertopicAndPerspective newModel) {
		oldModel.updatePriors(0, oldModel.Z, 0, oldModel.D);
		oldModel.updateGradient(200, 0, oldModel.Z, 0, oldModel.W);
		
		// Update gradients
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.UPDATE_PRIOR, newModel.threadName));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.CALC_GRADIENT, newModel.threadName));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < newModel.numThreads; i++) {
			try {
				newModel.THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static void main(String[] args) {
		Arguments p = ArgParse.parseArgs(args);
		String inPath = p.filename;
		
//		Log.initFileLogger(p.logPath);
		
		SpriteSupertopicAndPerspective newModel = buildNewModel(p);
		SpriteJoint oldModel = buildOldModel(p);
		
//		String[] argsPrime = Arrays.copyOf(args, args.length + 2);
//		argsPrime[args.length] = "-model";
//		argsPrime[args.length+1] = "sprite_joint";
		
		
		try {
//			newModel.readDocs(inPath);
//			newModel.initTrain();
//			oldModel.readDocs(inPath);
//			oldModel.initTrain();
			
//			copyParams(oldModel, newModel);
			
			// Test if same topics are sampled by each model
			
			oldModel.r = new Random(1000);
			MathUtils.initRandomStream(1000);
			
			// Train both models independently, but with same random seeds...
			Log.initFileLogger(p.logPath, true);
			newModel.train(3000, 100, inPath);
			Log.initFileLogger(p.logPath);
			oldModel.train(3000, 100, inPath);
			Log.initFileLogger(p.logPath, true);
			
//			Log.initFileLogger(p.logPath, true);
			
			/*
			for (int j = 0; j < 10; j++) {
				oldModel.sampleBatch(0, oldModel.D);

				for (int i = 0; i < newModel.numThreads; i++) {
					try {
						newModel.THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.SAMPLE, newModel.threadName));
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}

				for (int i = 0; i < newModel.numThreads; i++) {
					try {
						newModel.THREAD_MASTER_QUEUE.take();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
			
			updatePriorsAndGrad(oldModel, newModel);
			*/
			
			// Sample for a bit...
//			oldModel.train(199, 1, inPath);
//			System.out.println("Breakpt");
			
//			Log.initFileLogger(p.logPath, true);
			
//			newModel.train(1000, 1, inPath);
			
//			Log.initFileLogger(p.logPath, true);
			
//			System.out.println("Breakpt");
			
//			copyCounts(oldModel, newModel);
//			copyParams(oldModel, newModel);
			
			/*
			updatePriorsAndGrad(oldModel, newModel);
			
			for (int i = 0; i < 10; i++) {
				takeStep(oldModel, newModel);
			}
			*/
			
			// ==== Record parameters for debugging gradients ====
			
			/*
			Log.error("check_grad deltaBias", Utils.mkStr(oldModel.deltaBias));
			Log.error("check_grad omegaBias", Utils.mkStr(oldModel.omegaBias));
			Log.error("check_grad new_thetaTilde", Utils.mkStr(newModel.thetaPriors[0].thetaTilde));
			Log.error("check_grad old_thetaTilde", Utils.mkStr(oldModel.priorDZ));
			Log.error("check_grad new_thetaNorm", Utils.mkStr(newModel.thetaPriors[0].thetaNorm));
			Log.error("check_grad old_thetaNorm", Utils.mkStr(oldModel.thetaNorm));
			Log.error("check_grad new_phiTilde", Utils.mkStr(newModel.phiPriors[0].phiTilde));
			Log.error("check_grad old_phiTilde", Utils.mkStr(oldModel.priorZW));
			Log.error("check_grad new_phiNorm", Utils.mkStr(newModel.phiPriors[0].phiNorm));
			Log.error("check_grad old_phiNorm", Utils.mkStr(oldModel.phiNorm));
			Log.error("check_grad nD", Utils.mkStr(newModel.nD));
			Log.error("check_grad nDZ", Utils.mkStr(newModel.nDZ));
			Log.error("check_grad nZ", Utils.mkStr(newModel.nZ[0]));
			Log.error("check_grad nZW", Utils.mkStr(newModel.nZW[0]));
			
			Log.error("check_grad new_perspOmega", Utils.mkStr(newModel.factors[0].omega));
			Log.error("check_grad new_perspBeta",  Utils.mkStr(newModel.factors[0].beta));
			Log.error("check_grad new_perspBetaB", Utils.mkStr(newModel.factors[0].betaB));
			Log.error("check_grad new_perspDelta", Utils.mkStr(newModel.factors[0].delta));
			Log.error("check_grad new_perspAlpha", Utils.mkStr(newModel.factors[0].alpha));
			
			Log.error("check_grad new_hierOmega", Utils.mkStr(newModel.factors[1].omega));
			Log.error("check_grad new_hierBeta",  Utils.mkStr(newModel.factors[1].beta));
			Log.error("check_grad new_hierBetaB", Utils.mkStr(newModel.factors[1].betaB));
			Log.error("check_grad new_hierDelta", Utils.mkStr(newModel.factors[1].delta));
			Log.error("check_grad new_hierAlpha", Utils.mkStr(newModel.factors[1].alpha));
			
			Log.error("check_grad old_omega", Utils.mkStr(oldModel.omega));
			Log.error("check_grad old_beta",  Utils.mkStr(oldModel.beta));
			Log.error("check_grad old_betaB", Utils.mkStr(oldModel.betaB));
			Log.error("check_grad old_delta", Utils.mkStr(oldModel.delta));
			Log.error("check_grad old_alpha", Utils.mkStr(oldModel.alpha));
			
			Log.error("check_grad old_gradientDeltaBias", Utils.mkStr(oldModel.gradientDeltaBias));
			Log.error("check_grad old_gradientOmegaBias", Utils.mkStr(oldModel.gradientOmegaBias));
			Log.error("check_grad old_gradientOmega", Utils.mkStr(oldModel.gradientOmega));
			Log.error("check_grad old_gradientBeta",  Utils.mkStr(oldModel.gradientBeta));
			Log.error("check_grad old_gradientBetaB", Utils.mkStr(oldModel.gradientBetaB));
			Log.error("check_grad old_gradientDelta", Utils.mkStr(oldModel.gradientDelta));
			Log.error("check_grad old_gradientAlpha", Utils.mkStr(oldModel.gradientAlpha));
			
			Log.error("check_grad new_gradientDeltaBias", Utils.mkStr(newModel.thetaPriors[0].gradientDeltaBias));
			Log.error("check_grad new_gradientOmegaBias", Utils.mkStr(newModel.phiPriors[0].gradientOmegaBias));
			Log.error("check_grad new_perspGradientOmega", Utils.mkStr(newModel.factors[0].gradientOmega));
			Log.error("check_grad new_perspGradientBeta",  Utils.mkStr(newModel.factors[0].gradientBeta));
			Log.error("check_grad new_perspGradientBetaB", Utils.mkStr(newModel.factors[0].gradientBetaB));
			Log.error("check_grad new_perspGradientDelta", Utils.mkStr(newModel.factors[0].gradientDelta));
			Log.error("check_grad new_perspGradientAlpha", Utils.mkStr(newModel.factors[0].gradientAlpha));
			Log.error("check_grad new_hierGradientOmega", Utils.mkStr(newModel.factors[1].gradientOmega));
			Log.error("check_grad new_hierGradientBeta",  Utils.mkStr(newModel.factors[1].gradientBeta));
			Log.error("check_grad new_hierGradientBetaB", Utils.mkStr(newModel.factors[1].gradientBetaB));
			Log.error("check_grad new_hierGradientDelta", Utils.mkStr(newModel.factors[1].gradientDelta));
			Log.error("check_grad new_hierGradientAlpha", Utils.mkStr(newModel.factors[1].gradientAlpha));
			*/
			
			// ==== Check if gradients match up ====
			
			for (int z = 0; z < oldModel.Z; z++) {
				if (Math.abs(oldModel.gradientDeltaBias[z] - newModel.thetaPriors[0].gradientDeltaBias[z]) > 1.e-6) {
					System.out.println("[check_grad] " + "gradientDeltaBias not the same: " + z +
							" -- " + oldModel.gradientDeltaBias[z] + " | " +
							newModel.thetaPriors[0].gradientDeltaBias[z]);
					break;
				}
			}
			
			for (int w = 0; w < oldModel.W; w++) {
				if (Math.abs(oldModel.gradientOmegaBias[w] - newModel.phiPriors[0].gradientOmegaBias[w]) > 1.e-6) {
					System.out.println("[check_grad]" + "gradientOmegaBias not the same: " + w +
							" -- " + oldModel.gradientOmegaBias[w] + " | " +
							newModel.phiPriors[0].gradientOmegaBias[w]);
					break;
				}
			}
			
			for (int c = 1; c < oldModel.Cph; c++) {
				double[][] newOmega = c == 0 ? newModel.factors[0].gradientOmega : newModel.factors[1].gradientOmega;
				for (int w = 0; w < oldModel.W; w++) {
					if (Math.abs(oldModel.gradientOmega[c][w] - newOmega[c-1][w]) > 1.e-6) {
						System.out.println("[check_grad] " + "gradientOmega not the same: " + c + " " + w + 
								" -- " + oldModel.gradientOmega[c][w] + " | " +
								newOmega[c-1][w]);
						break;
					}
				}
			}
			
			for (int c = 1; c < oldModel.Cth; c++) {
				double[][] newAlpha = c == 0 ? newModel.factors[0].gradientAlpha : newModel.factors[1].gradientAlpha;
				for (int d = 0; d < oldModel.D; d++) {
					if (Math.abs(oldModel.gradientAlpha[d][c] - newAlpha[d][c-1]) > 1.e-6) {
						System.out.println("[check_grad] " + "gradientAlpha not the same: " + d + " " + c + 
								" -- " + oldModel.gradientAlpha[d][c] + " | " +
								newAlpha[d][c-1]);
						break;
					}
				}
			}
			
			for (int z = 0; z < oldModel.Z; z++) {
				for (int c = 1; c < oldModel.Cph; c++) {
					if (Math.abs(oldModel.gradientBeta[z][c] - newModel.factors[1].gradientBeta[0][z][c-1]) > 1.e-6) {
						System.out.println("[check_grad] " + "gradientBeta not the same: " + z + " " + c + 
								" -- " + oldModel.gradientBeta[z][c] + " | " +
								newModel.factors[1].gradientBeta[0][z][c-1]);
						break;
					}
				}
			}
			
			for (int z = 0; z < oldModel.Z; z++) {
				for (int c = 1; c < oldModel.Cph; c++) {
					if (Math.abs(oldModel.gradientBetaB[z][c] - newModel.factors[1].gradientBetaB[0][z][c-1]) > 1.e-6) {
						System.out.println("[check_grad] " + "gradientBetaB not the same: " + z + " " + c + 
								" -- " + oldModel.gradientBetaB[z][c] + " | " +
								newModel.factors[1].gradientBetaB[0][z][c-1]);
						break;
					}
				}
			}
			
			for (int z = 0; z < oldModel.Z; z++) {
				for (int c = 1; c < oldModel.Cph; c++) {
					if (Math.abs(oldModel.gradientDelta[c][z] - newModel.factors[1].gradientDelta[0][c-1][z]) > 1.e-6) {
						System.out.println("[check_grad] " + "gradientDelta not the same: " + c + " " + z + 
								" -- " + oldModel.gradientDelta[c][z] + " | " +
								newModel.factors[1].gradientDelta[0][c-1][z]);
						break;
					}
				}
			}
			
			logParams("/home/adrianb/Desktop/michael_sprite_data/debugJointGradient_ratemd_sameInitTrain3000.txt", oldModel, newModel);
			
			System.out.println("Breakpt");
			
			File outputDirFile = new File("/home/adrianb/Desktop/michael_sprite_data/",
					String.format("%s_%d_%d_%.2f_%.2f_%.2f_%.2f_%.3f_%.3f_%d_%d_output", "sprite_joint", p.C, p.z,
							   p.step, 1.0, p.deltaBias, p.omegaBias, p.sigmaDeltaBias, p.sigmaOmegaBias,
							   0, -1));
			String outputDir = outputDirFile.getAbsolutePath();
			(new File(outputDir)).mkdir();
			
			oldModel.writeOutput(p.filename, outputDir);
			newModel.writeOutput(p.filename, "/home/adrianb/Desktop/michael_sprite_data/supertopic_perspective/");
			
			oldModel.cleanUp();
			newModel.cleanUp();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
}
