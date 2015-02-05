package models.factored;

/**
 * Methods models should implement.
 * 
 * @author adrianb
 *
 */
public interface Trainable {
	
	public int x = 0;
	
	/*
	 * Prints out current model state
	 */
	public abstract void logIteration();
	
	/*
	 * Collects current topic samples of words in some structure.
	 */
	public abstract void collectSamples();
	
	/*
	 * Compute log-likelihood of corpus -- array of document to view to token IDs.
	 */
	public abstract double computeLL(int[][][] corpus);
	
}
