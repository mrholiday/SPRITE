package utils;

import utils.MathUtils;

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
		public double deltaBias = -2.0;
		
		@Parameter(names="-omegaB", description="Initial value for omega bias (on \\widetilde{\\phi})")
		public double omegaBias = -4.0;
		
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
		
		@Parameter(names="-sigmaBeta", description="Stddev for beta")
		public double sigmaBeta = 10.0;
		
		@Parameter(names="-sigmaOmega", description="Stddev for omega")
		public double sigmaOmega = 10.0;
		
		@Parameter(names="-initOmegaPath", description="Path to initialize omega from")
		public String initOmegaPath = null;
		
		@Parameter(names="-numFactors", description="How many supertopic factors to build our model with")
		public int numFactors = -1;
		
		@Parameter(names="-priorWeight", description="How heavily to weight the theta/phi priors")
		public double priorWeight = 1.0;
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
			System.exit(0);
			return null;
		}
		else {
			initRandAndLog(p.seed, p.logPath);
			return p;
		}
	}
	
	public static String getArgString(Arguments p) {
		StringBuilder b = new StringBuilder("{");
		
		b.append(String.format("\"C\":%d, ", p.C));
		b.append(String.format("\"deltaBias\":%.2f, ", p.deltaBias));
		b.append(String.format("\"filename\":%s, ", p.filename));
		b.append(String.format("\"initOmegaPath\":%s, ", p.initOmegaPath));
		b.append(String.format("\"iters\":%d, ", p.iters));
		b.append(String.format("\"likelihoodFreq\":%d, ", p.likelihoodFreq));
		b.append(String.format("\"logPath\":%s, ", p.logPath));
		b.append(String.format("\"numFactors\":%d, ", p.numFactors));
		b.append(String.format("\"numThreads\":%d, ", p.numThreads));
		b.append(String.format("\"omegaBias\":%.2f, ", p.omegaBias));
		b.append(String.format("\"outDir\":%s, ", p.outDir));
		b.append(String.format("\"priorWeight\":%1f, ", p.priorWeight));
		b.append(String.format("\"samples\":%d, ", p.samples));
		b.append(String.format("\"seed\":%d, ", p.seed));
		b.append(String.format("\"sigmaAlpha\":%1f, ", p.sigmaAlpha));
		b.append(String.format("\"sigmaBeta\":%1f, ", p.sigmaBeta));
		b.append(String.format("\"sigmaOmega\":%1f, ", p.sigmaOmega));
		b.append(String.format("\"sigmaOmegaBias\":%1f, ", p.sigmaOmegaBias));
		b.append(String.format("\"sigmaDelta\":%1f, ", p.sigmaDelta));
		b.append(String.format("\"sigmaDeltaBias\":%1f, ", p.sigmaDeltaBias));
		b.append(String.format("\"step\":%.3f, ", p.step));
		b.append(String.format("\"z\":%d, ", p.z));
		
		b.append("}");
		
		return b.toString();
	}
	
}
