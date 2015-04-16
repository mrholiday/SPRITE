package main;

import java.io.File;
import java.io.Serializable;
import java.math.BigInteger;

import models.factored.Trainable;

import utils.Log;
import utils.Tup4;

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
	protected int likelihoodFreq = 100;
	
	protected int[][][] docs;
	protected BigInteger[] docIds; // Unused except for printing to output
	
	public int[][][] nDZ; // Document -> View -> Topic samples
	public int[][]    nD; // Document -> View samples (number of tokens in each, set at initialization)
	public int[][][] nZW; // View -> Topic -> Word samples
	public int[][]    nZ; // View -> Topic samples
	
	public void train(int iters, int samples, String filename) throws Exception {
		String outputDir = new File(filename).getParent();
		
		train(iters, samples, filename, outputDir);
	}
	
	public void train(int iters, int samples, String filename, String outputDir) throws Exception {
		train(iters, samples, filename, outputDir, 100);
	}
	
	public void train(int iters, int samples, String filename, String outputDir, int likelihoodFreq0) throws Exception {
		likelihoodFreq = likelihoodFreq0;
		try {
			inputFilename = filename;
			readDocs(filename);
			initTrain();
			
			Log.info("train", "Sampling...");
			
			for (int iter = 1; iter <= iters; iter++) {
				if (iter > (iters - burnInIters)) burnedIn = true; // Keep the last couple hundred samples for final estimates
				
				Log.info("train", "Iteration " + iter);
				doSamplingIteration(iter);
				
				// save the output periodically
//				if (iter % writeFreq == 0) {
//					System.out.println("Saving output...");
//					writeOutput(filename + iter);
//				}
			}
			
			if (outputDir != null) {
				writeOutput(filename, outputDir);
			}
			else {
				writeOutput(filename);
			}
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
	 * 
	 * @throws Exception When it messes up
	 */
	public void test(int iters, String filename) throws Exception {
		try {
			inputFilename = filename;
			readTestDocs(filename);
			initTrain();
			
			Log.info("test", "Sampling...");
			
			for (int iter = 1; iter <= iters; iter++) {
				if (iter >= (iters - burnInIters)) burnedIn = true; // Keep the last couple hundred samples for final estimates
				
				Log.info("test", "Iteration " + iter);
				doSamplingIteration(iter);
				
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
	
	public void setSamples(int[][][] nDZ0, int[][] nD0, int[][][] nZW0, int[][] nZ0) {
		nDZ = nDZ0;
		nD  = nD0;
		nZW = nZW0;
		nZ  = nZ0;
	}
	
	public Tup4<int[][][], int[][], int[][][], int[][]> getSamples() {
		return new Tup4<int[][][], int[][], int[][][], int[][]>(nDZ, nD, nZW, nZ);
	}
	
	public double computeLL() { return computeLL(docs); } // Compute log-likelihood on training data
	
	protected abstract void initTrain();
	
	protected abstract void initTest();
	
	/**
	 * A single iteration of the main training loop.
	 * 
	 * @param iter The iteration number
	 */
	public abstract void doSamplingIteration(int iter);
	
	public abstract void readDocs(String filename) throws Exception;
	
	public void readTestDocs(String filename) throws Exception {
		throw new UnsupportedOperationException("Not implemented!");
	}
	
	public abstract void writeOutput(String filename, String outputDir) throws Exception;
	
	public void writeOutput(String filename) throws Exception {
		writeOutput(filename, new File(filename).getParent());
	}
	
	public abstract void cleanUp() throws Exception;
	
}
