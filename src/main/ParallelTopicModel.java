package main;

import java.util.concurrent.ArrayBlockingQueue;

import utils.Log;
import utils.Tup2;

import main.SpriteWorker.ThreadCommand;
import main.SpriteWorker.ThreadCommunication;

public abstract class ParallelTopicModel extends TopicModel implements Trainable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -3336308017072889625L;
	
	protected int numThreads    = 1;
	protected String threadName = "MASTER";
	
	public ArrayBlockingQueue<ThreadCommunication> THREAD_WORKER_QUEUE = null; // Signals workers have a job to do
    public ArrayBlockingQueue<ThreadCommunication> THREAD_MASTER_QUEUE = null; // Signals work is done
    
	private SpriteWorker[] THREADS = null;
	
	private boolean TIME_ITERATIONS = false;
	private int likelihoodFreq      = 100;
	
	private int iter = -1;
	
	// The dimension of parameters we want to parallelize over.  Subclasses
	// need to set to something reasonable.
	protected int[][] varDims;
	
	// For locking during the parallel sampling/hyperparameter gradient step
	protected Integer[] wordLocks;
	protected Integer[] topicLocks;
	protected Integer[] docLocks;
	
	protected void setParallelParams(int numThreads0, int[][] varDims0) {
		numThreads = numThreads0;
		varDims = varDims0;
	}
	
	@Override
	@SuppressWarnings("unchecked")
	protected void initialize() {
		/*
		 * Spins up worker threads.  Implementation will want to add to this
		 * by initializing parameters.
		 */
		
		// Spin up worker threads.  Will only work when ThreadComm message is received.
		THREAD_WORKER_QUEUE = new ArrayBlockingQueue<ThreadCommunication>(numThreads);
		THREAD_MASTER_QUEUE = new ArrayBlockingQueue<ThreadCommunication>(numThreads);
		THREADS     = new SpriteWorker[numThreads];
		
		int[][] stepSizes = new int[varDims.length][];
		for (int i = 0; i < varDims.length; i++) {
			stepSizes[i] = new int[varDims[i].length];
			for (int j = 0; j < varDims[i].length; j++) {
				stepSizes[i][j] = varDims[i][j]/numThreads;
			}
		}
		
		for (int i = 0; i < numThreads; i++) {
			Tup2<Integer, Integer>[][] paramRanges = new Tup2[varDims.length][];
			
			for (int j = 0; j < varDims.length; j++) {
				paramRanges[j] = new Tup2[varDims[j].length];
				
				for (int k = 0; k < varDims[j].length; k++) {
					int minRange = stepSizes[j][k]*i;
					int maxRange = i < (numThreads - 1) ? stepSizes[j][k] * (i+1) : varDims[j][k];
					
					paramRanges[j][k] = new Tup2<Integer, Integer>(minRange, maxRange);
				}
			}
			THREADS[i] = new SpriteWorker(paramRanges, this);
			THREADS[i].start();
		}
	}
	
	@Override
	public void doSampling(int iter0) {
		iter = iter0;
		
		long startTime = System.currentTimeMillis();
		
		// sample topic values for all the tokens
		try {
			for (int i = 0; i < numThreads; i++) {
				THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.SAMPLE, threadName));
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		for (int i = 0; i < numThreads; i++) {
			try {
				THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		// Begin hyperparameter updates after a couple hundred iterations
		if (iter >= burnInIters) {
			// Calculate gradient for hyperparameters (if any)
			try {
				for (int i = 0; i < numThreads; i++) {
					THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.CALC_GRADIENT, threadName));
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			for (int i = 0; i < numThreads; i++) {
				try {
					THREAD_MASTER_QUEUE.take();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			// Take a step along the gradient
			try {
				for (int i = 0; i < numThreads; i++) {
					THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.GRADIENT_STEP, threadName));
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			for (int i = 0; i < numThreads; i++) {
				try {
					THREAD_MASTER_QUEUE.take();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		
		// Compute the priors with the new params and update the cached prior variables 
		for (int i = 0; i < numThreads; i++) {
			try {
				THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.UPDATE_PRIOR, threadName));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0; i < numThreads; i++) {
			try {
				THREAD_MASTER_QUEUE.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		// Log parameter values
		logIteration();
		
		if (((iter % likelihoodFreq) == 0) || (likelihoodFreq == 0)) {
			Log.info("topic_model", "Log-likelihood: " + computeLL());
		}
		
		collectSamples();
		
		if (TIME_ITERATIONS) {
			long endTime = System.currentTimeMillis();
			Log.info("topic_model", String.format("Iteration time:\t%d\t%d", iter, endTime - startTime));
		}
	}
	
	/**
	 * Updates the priors for this topic model for a subset of parameters.
	 * 
	 * @param parameterRanges Range this thread works over.
	 */
	public abstract void updatePriors(Tup2<Integer, Integer>[][] parameterRanges); 
	
	/**
	 * Samples new topics for a subset of the tokens in our corpus.
	 * 
	 * @param parameterRanges Range this thread works over.
	 */
	public abstract void sampleBatch(Tup2<Integer, Integer>[][] parameterRanges); 
	
	/**
	 * Updates the gradient of the hyperparameters given current estimate.
	 * 
	 * @param parameterRanges Range this thread works over.
	 */
	public abstract void updateGradient(Tup2<Integer, Integer>[][] parameterRanges); 

	/**
	 * Take a step along the just-computed gradient.
	 * 
	 * @param parameterRanges Range this thread works over.
	 */
	public abstract void doGradientStep(Tup2<Integer, Integer>[][] parameterRanges); 
	
	@Override
	public void cleanUp() {
		/*
		 * Shuts down all worker threads and waits for them to
		 * acknowledge their demise.
		 */
		
		for (int i = 0; i < numThreads; i++) {
			try {
				THREAD_WORKER_QUEUE.put(new ThreadCommunication(ThreadCommand.KILL, threadName, "kill"));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		for (int i = 0 ; i < numThreads; i++) {
			try {
				THREADS[i].join(10000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		System.out.println("Killed worker threads");
	}
	
}
