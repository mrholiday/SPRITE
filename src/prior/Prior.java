package prior;

import java.lang.Math;
import java.util.Random;

import main.Optimizable;

/**
 * Hyperprior over our parameters.  Prior may need to know how to update
 * themselves, but at the very least you should be able to sample from itself,
 * compute the probability of the parameters, and the gradient at a point.
 * 
 * @author adrianb
 *
 */
public abstract class Prior implements Optimizable {
    
	// We may or may not want to allow this prior to vary.
	public boolean isOptimizable = false;
	public double step = 0.01;
	
	public abstract double[] sample(Random r);
	
	// Gradient of probability w.r.t. these parameters.
	public abstract double[] gradient(double[] parameters);
	
    public abstract double likelihood(double[] parameters);
	
	public double logLikelihood(double[] parameters) { return Math.log(likelihood(parameters)); }
	
	@Override
	public void setMasterStepSize(double step0) { step = step0; }
	
}
