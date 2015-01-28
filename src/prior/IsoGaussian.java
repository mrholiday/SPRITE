package prior;

import java.util.Random;

/**
 * Isotropic normal distribution.
 * 
 * @author adrianb
 *
 */
public class IsoGaussian extends Prior {
	
	private String name = "GaussianPrior";
	
	private double[] mean;
	private double   variance;
	private double   stdDev;
	
	private double[] adaDeltaMean;
	private double[] gradientMean;
	
	private int N;
	
	public IsoGaussian(double[] mean0, double variance0, boolean isOptimizable0, String name0) {
		mean = mean0;
		variance = variance0;
		stdDev = Math.sqrt(variance);
		isOptimizable = isOptimizable0;
		name = name0;
		init();
	}
	
	public void init() {
		N = mean.length;
		adaDeltaMean = new double[N];
		gradientMean = new double[N];
	}
	
	public double[] gradient(double[] parameters) {
		double[] paramGradient = new double[N];
		for (int i = 0; i < N; i++)
			paramGradient[i] = (mean[i] - parameters[i])/variance;
		
		return paramGradient;
	}
	
	@Override
	public void takeGradientStep(double g, double gPrime) {
		// TODO: figure out division of labor for computing gradients
	}
	
	private double sampleOne(double mean, Random r) {
		return mean + r.nextGaussian() * stdDev;
	}
	
	@Override
	public double[] sample(Random r) {
		double[] paramSample = new double[N];
		for (int i = 0; i < N; i++) {
			paramSample[i] = sampleOne(mean[i], r);
		}
		return paramSample;
	}
	
	@Override
	public double likelihood(double[] parameters) {
		return 0;
	}

}
