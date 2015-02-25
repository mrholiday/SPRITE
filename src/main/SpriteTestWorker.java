package main;

import main.SpriteWorker.ThreadCommunication.ThreadCommand;
import models.factored.ParallelTopicModel;
import utils.Log;
import utils.Tup2;

/**
 * Works on sampling topics for a new set of documents.
 * 
 * @author adrianb
 *
 */
public class SpriteTestWorker extends SpriteWorker {

	public SpriteTestWorker(Tup2<Integer, Integer>[][] parameterRanges0,
			ParallelTopicModel tm0) {
		super(parameterRanges0, tm0);
	}

	@Override
	public void run() {
		
		while (true) {
			try {
				ThreadCommunication msg = tm.THREAD_WORKER_QUEUE.take();
				
				if (msg.cmd.equals(ThreadCommand.KILL)) {
					break;
				}
				else if (msg.cmd.equals(ThreadCommand.UPDATE_PRIOR)) {
					tm.updatePriors(parameterRanges);
				}
				else if (msg.cmd.equals(ThreadCommand.SAMPLE)) {
					tm.sampleBatchTest(parameterRanges);
				}
				else if (msg.cmd.equals(ThreadCommand.CALC_GRADIENT)) {
					tm.updateGradientTest(parameterRanges);
				}
				else if (msg.cmd.equals(ThreadCommand.GRADIENT_STEP)) {
					tm.doGradientStep(parameterRanges);
				}
				else if (msg.cmd.equals(ThreadCommand.CLEAR_GRADIENT)) {
					tm.clearGradient(parameterRanges);
				}
				else {
					Log.error("worker_" + threadName, "Unrecognized message: " + msg.cmd + " " + msg.msg);
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

}
