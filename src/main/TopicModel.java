package main;

import java.io.File;
import java.io.Serializable;
import java.math.BigInteger;

import models.factored.Trainable;

import utils.Log;

public abstract class TopicModel implements Trainable, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -5947800023944902504L;
	
	public String inputFilename;
	
	protected int burnInIters = 200;
	protected boolean burnedIn = false;
	protected int writeFreq = 1000;
	protected int logFreq   = 10;
	
	protected int[][][] docs;
	protected BigInteger[] docIds; // Unused except for printing to output
	
	public void train(int iters, int samples, String filename) throws Exception {
		try {
			inputFilename = filename;
			readDocs(filename);
			initTrain();
			
			Log.info("train", "Sampling...");
			
			for (int iter = 1; iter <= iters; iter++) {
				if (iter >= (iters - burnInIters)) burnedIn = true; // Keep the last couple hundred samples for final estimates
				
				Log.info("train", "Iteration " + iter);
				doTrainSampling(iter);
				
				// save the output periodically
//				if (iter % writeFreq == 0) {
//					System.out.println("Saving output...");
//					writeOutput(filename + iter);
//				}
			}
			
			writeOutput(filename);
		}
		catch (Exception e) {
			Log.error("train", "Error in training model.", e);
			e.printStackTrace();
		}
		finally {
			cleanUp();
		}
		
		Log.info("train", "...done.");
		Log.closeLogger();
	}
	
	/**
	 * Samples topics and infers \alpha for factors on a new corpus.
	 * 
	 * @param iters Number of iterations to run for
	 * @param samples Number of samples to keep
	 * @param filename Path to the unlabeled examples
	 * @throws Exception When it messes up
	 */
	public void test(int iters, String filename) throws Exception {
		try {
			inputFilename = filename;
			readDocs(filename);
			initTrain();
			
			Log.info("test", "Sampling...");
			
			for (int iter = 1; iter <= iters; iter++) {
				if (iter >= (iters - burnInIters)) burnedIn = true; // Keep the last couple hundred samples for final estimates
				
				Log.info("test", "Iteration " + iter);
				doInference(iter);
				
				// save the output periodically
//				if (iter % writeFreq == 0) {
//					System.out.println("Saving output...");
//					writeOutput(filename + iter);
//				}
			}
			
			writeOutput(filename);
		}
		catch (Exception e) {
			Log.error("test", "Error in inferring alpha/topic assignments", e);
			e.printStackTrace();
		}
		finally {
			cleanUp();
		}
		
		Log.info("test", "...done.");
		Log.closeLogger();
	}
	
	public double computeLL() { return computeLL(docs); } // Compute log-likelihood on training data
	
	protected abstract void initTrain();
	
	protected abstract void initTest();
	
	/**
	 * A single iteration of the main training loop.
	 * 
	 * @param iter The iteration number
	 */
	public abstract void doTrainSampling(int iter);
	
	/**
	 * Takes a set of unlabeled documents, samples topics for words and documents, and tried to infer alpha values for them.
	 * 
	 * @param iter0 Current iteration
	 */
	public abstract void doInference(int iter0);
	
	public abstract void readDocs(String filename) throws Exception;
	
	public abstract void writeOutput(String filename, String outputDir) throws Exception;
	
	public void writeOutput(String filename) throws Exception {
		writeOutput(filename, new File(filename).getParent());
	}
	
	public abstract void cleanUp() throws Exception;
	
}
