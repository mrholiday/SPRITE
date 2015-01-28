package main;

/**
 * 
 * For things we want to optimize with gradient ascent.  If this is a function f,
 * needs to know g' and and g for the current data (for chain rule).
 * 
 * @author adrianb
 */
public interface Optimizable {
	
	// Need to figure out the correct method signature, and how labor is to
	// be divided.  Don't want to rely on autodiff.
	public void takeGradientStep(double g, double gPrime);
	
	public void setMasterStepSize(double step);
	
}
