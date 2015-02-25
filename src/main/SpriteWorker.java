package main;

import utils.Log;
import utils.Tup2;
import models.factored.ParallelTopicModel;

/**
 * Does work for the topic model based on messages received (sample, update priors,
 * calculate gradient, take a gradient step.)
 */
public abstract class SpriteWorker extends Thread {
	
	public String threadName;
	
	// ViewIndex -> {(minZ, maxZ), (minD, maxD), (minV, maxV)}
	protected Tup2<Integer, Integer>[][] parameterRanges;
	protected ParallelTopicModel tm;
	
	/**
	 * Pass list of data ranges this worker has in its domain.  It is the
	 * topic model's job to interpret these ranges in the methods called by
	 * worker.
	 */
	public SpriteWorker(Tup2<Integer, Integer>[][] parameterRanges0, ParallelTopicModel tm0) {
		parameterRanges = parameterRanges0;
		tm = tm0;
		
		threadName = "Worker";
		for (int i = 0; i < parameterRanges.length; i++) {
			Tup2<Integer, Integer>[] zdv = parameterRanges[i];
			for (Tup2<Integer, Integer> minMax : zdv) {
				
				threadName += "_" + minMax._1() + "-" + minMax._2();
			}
		}
	}
	
	public void start() {
	      Log.info("worker_thread", "Starting " +  threadName);
	      Thread t = new Thread (this, threadName);
	      t.start();
	}
	
	public static class ThreadCommunication {
		/**
		 * Protocol for master-worker communication.
		 */
		
		public static enum ThreadCommand {UPDATE_PRIOR, KILL, SAMPLE, CALC_GRADIENT, GRADIENT_STEP, CLEAR_GRADIENT, DONE};
		
		public String source, msg;
		public ThreadCommand cmd;
		
		public ThreadCommunication(ThreadCommand cmd0, String source0) {
			this(cmd0, source0, "");
		}
		
		public ThreadCommunication(ThreadCommand cmd0, String source0, String msg0) {
			source = source0;
			msg    = msg0;
			cmd    = cmd0;
		}
	}
	
}
