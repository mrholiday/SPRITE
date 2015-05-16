package utils;

import java.util.Random;

import org.apache.commons.math3.analysis.function.Gaussian;
import org.apache.commons.math3.special.Gamma;

public class MathUtils {
	
	public static final double eps = 1.0e-6;
	
	public static double logistic(double x) {
		return 1.0 / (1.0 + Math.exp(-1.0*x));
	}
	
	// derivative of logistic function
	public static double dlogistic(double x) {
		return logistic(x) * (1.0 - logistic(x));
	}
	
	// Approximation to the digamma function, from Radford Neal.
	// can also use Gamma.digamma() from commons
	public static double digamma(double x) {
		//return Gamma.digamma(x);
		
		double r = 0.0;
		
		while (x <= 5.0) {
			r -= 1.0 / x;
			x += 1.0;
		}
		
		double f = 1.0 / (x * x);
		double t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0 + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617.0 / 8160.0)))))));
		return r + Math.log(x) - 0.5 / x + t;
	}
	
	public static double normalProb(double x, double mu, double sigma) {
		Gaussian normalDist = new Gaussian(mu, sigma);
		return normalDist.value(x);
	}
	
	public static double normalLogProb(double x, double mu, double sigma) {
		return Math.log(normalProb(x, mu, sigma));
	}
	
	public static double dirichletProb(double[] sample, double[] alpha) {
		double denom = 1.0;
		double alphaSum = 0.0;
		for (int i = 0; i < alpha.length; i++) {
			denom *= Gamma.gamma(alpha[i]);
			alphaSum += alpha[i];
		}
		denom /= Gamma.gamma(alphaSum);
		
		double num = 1.0;
		for (int i = 0; i < sample.length; i++) {
			num *= Math.pow(sample[i], alpha[i] - 1.0);
		}
		
		return num / denom;
	}
	
	public static double dirichletLogProb(double[] sample, double[] alpha) {
		double norm = 0.0;
		double alphaSum = 0.0;
		for (int i = 0; i < alpha.length; i++) {
			norm += Gamma.logGamma(alpha[i]);
			alphaSum += alpha[i];
		}
		norm -= Gamma.logGamma(alphaSum);
		
		double num = 0.0;
		for (int i = 0; i < sample.length; i++) {
			num += (alpha[i] - 1.) * Math.log(sample[i]);
		}
		
		System.out.println("Dirichlet log prob: " + num + ", " + norm);
		return num - norm;
	}
	
	public static double log(double x, double base) {
	    return (Math.log(x) / Math.log(base));
	}
	
	// Singleton source of randomness.  Saves me from passing it around.
	public static Random r;
	
	public static void initRandomStream() { r = new Random(); }
	
	public static void initRandomStream(int seed) { r = new Random(seed); }
	
}
