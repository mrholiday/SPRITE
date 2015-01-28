package main;

import java.util.List;

import utils.Tup2;

/**
 * Does work for the topic model based on messages received (sample, update priors,
 * calculate gradient, take a gradient step.)
 */
public class SpriteWorker extends Thread {
	
	public static enum ThreadCommand {UPDATE_PRIOR, KILL, SAMPLE, CALC_GRADIENT, GRADIENT_STEP, DONE};
	
	public String threadName;
	
	// ViewIndex -> {(minZ, maxZ), (minD, maxD), (minV, maxV)}
	private Tup2<Integer, Integer>[][] parameterRanges;
	private ParallelTopicModel tm;
	
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
				
				threadName += "_" + i + "-" + minMax._1() + "-" + minMax._2();
			}
		}
	}
	
	@Override
	public void run() {
		
		while (true) {
			try {
				ThreadCommunication msg = tm.THREAD_WORKER_QUEUE.take();
				if (msg.equals(ThreadCommand.KILL)) {
					break;
				}
				else if (msg.equals(ThreadCommand.UPDATE_PRIOR)) {
					tm.updatePriors(parameterRanges);
				}
				else if (msg.equals(ThreadCommand.SAMPLE)) {
					tm.sampleBatch(parameterRanges);
				}
				else if (msg.equals(ThreadCommand.CALC_GRADIENT)) {
					tm.updateGradient(parameterRanges);
				}
				else if (msg.equals(ThreadCommand.GRADIENT_STEP)) {
					tm.doGradientStep(parameterRanges);
				}
				tm.THREAD_MASTER_QUEUE.put(new ThreadCommunication(ThreadCommand.DONE, threadName, "success"));
			} catch (InterruptedException e) {
				e.printStackTrace();
				try {
					tm.THREAD_MASTER_QUEUE.put(new ThreadCommunication(ThreadCommand.DONE, threadName, "failure"));
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}
			}
		}
	}
	
	public void start() {
	      System.out.println("Starting " +  threadName );
	      Thread t = new Thread (this, threadName);
	      t.start();
	}
	
	public static class ThreadCommunication {
		/**
		 * Protocol for master-worker communication.
		 */
		
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
