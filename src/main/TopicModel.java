package main;

import java.io.File;
import java.io.Serializable;

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
	
	public void train(int iters, int samples, String filename) throws Exception {
		try {
			inputFilename = filename;
			readDocs(filename);
			initialize();
			
			System.out.println("Sampling...");
			
			for (int iter = 1; iter <= iters; iter++) {
				if (iter >= (iters - burnInIters)) burnedIn = true; // Keep the last couple hundred samples for final estimates
				
				System.out.println("Iteration " + iter);
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
			System.out.println("Error in training model.");
			e.printStackTrace();
		}
		finally {
			cleanUp();
		}
		System.out.println("...done.");
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
