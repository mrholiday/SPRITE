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
	
	protected int[][][] docs;
	protected BigInteger[] docIds; // Unused except for printing to output
	
	public void train(int iters, int samples, String filename) throws Exception {
		try {
			inputFilename = filename;
			readDocs(filename);
			initialize();
			
			Log.info("train", "Sampling...");
			
			for (int iter = 1; iter <= iters; iter++) {
				if (iter >= (iters - burnInIters)) burnedIn = true; // Keep the last couple hundred samples for final estimates
				
				Log.info("train", "Iteration " + iter);
				doSampling(iter);
				
				// save the output periodically
//				if (iter % writeFreq == 0) {
//					System.out.println("Saving output...");
//					writeOutput(filename + iter);
//				}
			}
			
			// write variable assignments
			
			writeOutput(filename);
		}
		catch (Exception e) {
			Log.error("train", "Error in training model.", e);
			e.printStackTrace();
		}
		finally {
			cleanUp();
			Log.closeLogger();
		}
		Log.info("train", "...done.");
	}
	
	public double computeLL() { return computeLL(docs); } // Compute log-likelihood on training data
	
	protected abstract void initialize();
	
	/**
	 * A single iteration of the main training loop.
	 * 
	 * @param iter The iteration number
	 */
	public abstract void doSampling(int iter);
	
	public abstract void readDocs(String filename) throws Exception;
	
	public abstract void writeOutput(String filename, String outputDir) throws Exception;
	
	public void writeOutput(String filename) throws Exception {
		writeOutput(filename, new File(filename).getParent());
	}
	
	public abstract void cleanUp() throws Exception;
	
}
