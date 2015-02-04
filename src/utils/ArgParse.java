package utils;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;

/**
 * Command-line arguments for training/running new SPRITE implementation.
 * 
 * @author adrianb
 * 
 */
public class ArgParse {
	
	public static class Arguments {
		@Parameter(names="-input", description="Path to training file", required=true)
		public String filename;
		
		@Parameter(names="-Z", description="Number of topics")
		public int z;
		
		@Parameter(names="-C", description="Number of components for supertopic factors")
		public int C = 5;
		
		@Parameter(names="-sigmaDeltaB", description="Stddev for delta bias")
		public double sigmaDeltaBias = 1.0;
		
		@Parameter(names="-sigmaOmegaB", description="Stddev for omega bias")
		public double sigmaOmegaBias = 10.0;
		
		@Parameter(names="-deltaB", description="Initial value for delta bias (on \\widetilde{\\theta})")
		public double deltaBias = -5.0;
		
		@Parameter(names="-omegaB", description="Initial value for omega bias (on \\widetilde{\\phi})")
		public double omegaBias = -5.0;
		
		@Parameter(names="-step", description="Master step size")
		public double step = 0.01;
		
		@Parameter(names="-seed", description="Random seed.  Default is to use clock time")
		public int seed = -1;
		
		@Parameter(names="-likelihoodFreq", description="How often to print out likelihood/perplexity.  Setting less than 1 will disable printing")
		public int likelihoodFreq = 100;
		
		@Parameter(names="-nthreads", description="Number of threads that will sample and take gradient steps in parallel")
		public int numThreads = 1;
		
		@Parameter(names="-iters", description="Number of iterations for training total")
		public int iters = 5000;
		
		@Parameter(names="-samples", description="Number of samples to take for the final estimate")
		public int samples = 100;
		
		@Parameter(names="-logPath", description="Where to log messages.  Defaults to stdout")
		public String logPath = null;
		
		@Parameter(names="-outDir", description="Where to write output.  Defaults to directory where training file is")
		public String outDir = null;
		
		@Parameter(names={"--help", "-help"}, help = true)
		public boolean help;
		
		@Parameter(names="-sigmaAlpha", description="Stddev for component to document assignments")
		public double sigmaAlpha = 1.0;
		
		@Parameter(names="-sigmaDelta", description="Stddev for delta")
		public double sigmaDelta = 1.0;
		
		@Parameter(names="-sigmaOmega", description="Stddev for omega")
		public double sigmaOmega = 1.0;
	}
	
	private static void initRandAndLog(int seed, String logPath) {
		if (seed == -1) {
			MathUtils.initRandomStream();
		}
		else {
			MathUtils.initRandomStream(seed);
		}
		
		if (logPath != null) {
			Log.initFileLogger(logPath);
		}
	}
	
	public static Arguments parseArgs(String[] args) {
		Arguments p = new Arguments();
		JCommander cmdr = new JCommander(p, args);
		
		if (p.help) {
			cmdr.usage();
			return null;
		}
		else {
			initRandAndLog(p.seed, p.logPath);
			return p;
		}
	}
	
}
