package cslda;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.Random;
//import org.apache.commons.math.special.Gamma;

import main.TopicModel;

//import org.apache.commons.math3.special.Gamma;

import utils.MathUtils;

public class CSLDA extends TopicModel {

	/**
	 * 
	 */
	private static final long serialVersionUID = -257428547300807172L;
	
	public HashMap<String,Integer> wordMap;
	public HashMap<Integer,String> wordMapInv;
	
	// Fold per document
	public int[] docFolds;
	
	// Doc collection IDs
	public int[] docsC;
	
	// Feature for each collection
	public double[] gold;
	
	// Tokens per document
	public int[][] docs;
	
	public int[][] docsZ;
	public int[][][] docsZZ;
	public int[][] docsX;
	
	public int[][] nDZ;
	public int[] nD;
	public int[][] nZW;
	public int[] nZ;
	public int[][] nCZ;  // topic counts per c label
	public int[] nBW;
	public int nB;
	public int[] nX;
	public int[] nC;  // total number tokens per collection
	
	public int D;
	public int W;
	public int Z;
	
	// Number of collections
	public int C;
	
//	public double beta;
	public double[] deltaBias;
	public double initOmegaBias;
	public double[] omegaBias;
	
	public double[] gradientDeltaBias;
	public double[] adaDeltaBias;
	public double[] gradientOmegaBias;
	public double[] adaOmegaBias;
	
	public double[] lambda;
	public double lambdaB;
	public double[] adaLambda;
	public double adaLambdaB;
//	public double gamma0;
//	public double gamma1;
	
	public double[] priorZW;
	public double   phiNorm;
	public double[] priorDZ;
	public double   thetaNorm;
	
	public double sigma;
	public double sigmaDeltaBias;
	public double sigmaOmegaBias;
	public double stepA;
	
	public boolean computePerplexity;
	
	public CSLDA(int Z0, double deltab0, double omegab0, double s, double st,
			double sigmaDeltaBias0, double sigmaOmegaBias0, boolean computePerplexity0) {
		Z = Z0;
		
		deltaBias = new double[Z];
		for (int i = 0; i < Z; i++) deltaBias[i] = deltab0;
		
//		gamma0 = g0;
//		gamma1 = g1;
		
		sigma = s;
		stepA = st;
		
		initOmegaBias = omegab0;
		
		sigmaDeltaBias = sigmaDeltaBias0;
		sigmaOmegaBias = sigmaOmegaBias0;
		
		computePerplexity = computePerplexity0;
	}
	
	public void initTrain() {
		System.out.println("Initializing...");
		Random r = new Random();

		lambda = new double[Z];
		lambdaB = 0.0; // bias 

		docsZ = new int[D][];
		docsZZ = new int[D][][];
		docsX = new int[D][];

		nDZ = new int[D][Z];
		nD = new int[D];
		nZW = new int[Z][W];
		nZ = new int[Z];
		nCZ = new int[C][Z];
		nBW = new int[W];
		nB = 0;
		nX = new int[2];
		nC = new int[C];
		
		omegaBias = new double[W];
		for (int w = 0; w < W; w++) {
			omegaBias[w] = initOmegaBias;
		}
		
		gradientDeltaBias = new double[Z];
		adaDeltaBias = new double[Z];
		gradientOmegaBias = new double[W];
		adaOmegaBias = new double[W];
		
		adaLambda = new double[Z];
		
		// Since priors are shared across all documents/topics, we don't need to keep separate vectors.
		priorZW = new double[W];
		phiNorm = 0.0;
		priorDZ = new double[Z];
		thetaNorm = 0.0;
		
		updatePriors();
		
		for (int d = 0; d < D; d++) {
			docsZ[d] = new int[docs[d].length];
			docsZZ[d] = new int[docs[d].length][Z];
			docsX[d] = new int[docs[d].length];
			
			int c = docsC[d];

			int N = docs[d].length;
			for (int n = 0; n < N; n++) {
				int w = docs[d][n];
				
				int z = r.nextInt(Z);		// select random z value in {0...Z-1}
				docsZ[d][n] = z;
				
				//int x = r.nextInt(2);		// select x uniformly
//				int x = 0;
//				double u = r.nextDouble();		// select random x value in {0,1}
//				u *= (double)(gamma0+gamma1);		// from distribution given by prior
//				if (u > gamma0) x = 1;
//				x = 1; // disable background
				int x = 1; // disable background
				docsX[d][n] = x;
				
				// update counts
				
				nX[x] += 1;
				
				nDZ[d][z] += 1;
				nD[d] += 1;
				
				if (!computePerplexity || (n % 2 == 0)) {
					nZW[z][w] += 1;	
					nZ[z] += 1;
				}
				
				if (c >= 0) {
					nCZ[c][z] += 1;
					nC[c] += 1;
				}
			}
		}
	}
	
	/*
	 * Update document prior
	 */
	public void updateAlpha() {
		for (int z = 0; z < Z; z++) {
			for (int d = 0; d < D; d++) {
				double dg1  = MathUtils.digamma(thetaNorm + MathUtils.eps);
				double dg2  = MathUtils.digamma(thetaNorm + nD[d] + MathUtils.eps);
				double dgW1 = MathUtils.digamma(priorDZ[z] + nDZ[d][z] + MathUtils.eps);
				double dgW2 = MathUtils.digamma(priorDZ[z] + MathUtils.eps);
				
				double gradientTerm = priorDZ[z] * (dg1-dg2+dgW1-dgW2);
				
				gradientDeltaBias[z] += gradientTerm;
			}
			gradientDeltaBias[z] -= (deltaBias[z]) / Math.pow(sigmaDeltaBias, 2);
			
			adaDeltaBias[z] += Math.pow(gradientDeltaBias[z], 2);
			deltaBias[z] += (stepA / (Math.sqrt(adaDeltaBias[z])+MathUtils.eps)) * gradientDeltaBias[z];
			gradientDeltaBias[z] = 0.;
		}
	}
	
	/**
	 * Update shared document and topic priors
	 */
	public void updatePriors() {
		phiNorm = 0.;
		for (int w = 0; w < W; w++) {
			priorZW[w] = Math.exp(omegaBias[w]);
			phiNorm += priorZW[w];
		}
		
		thetaNorm = 0.;
		for (int z = 0; z < Z; z++) {
			priorDZ[z] = Math.exp(deltaBias[z]);
			thetaNorm += priorDZ[z];
		}
	}
	
	/*
	 * Update topic prior
	 */
	public void updateBeta() {
		double dg1  = MathUtils.digamma(phiNorm + MathUtils.eps);
		for (int z = 0; z < Z; z++) {
			double dg2  = MathUtils.digamma(phiNorm + nZ[z] + MathUtils.eps);
			for (int w = 0; w < W; w++) {
				double dgW1 = MathUtils.digamma(priorZW[w] + nZW[z][w] + MathUtils.eps);
				double dgW2 = MathUtils.digamma(priorZW[w] + MathUtils.eps);
				
				double gradientTerm = priorZW[w] * (dg1-dg2+dgW1-dgW2);
				
				gradientOmegaBias[w] += gradientTerm;
			}
		}
		
		for (int w = 0; w < W; w++) {
			gradientOmegaBias[w] -= (omegaBias[w]) / Math.pow(sigmaOmegaBias, 2);
			adaOmegaBias[w] += Math.pow(gradientOmegaBias[w], 2);
			omegaBias[w] += (stepA / (Math.sqrt(adaOmegaBias[w])+MathUtils.eps)) * gradientOmegaBias[w];
			gradientOmegaBias[w] = 0.; // Clear gradient for the next iteration
		}
		
		/*
		double LLold = 0;
		double LLnew = 0;
		
		Random r = new Random();
        double betaNew = Math.exp(Math.log(beta) + r.nextGaussian());
        double betaSumNew = betaNew * (double)W;
        
		for (int z = 0; z < Z; z++) {
			LLold += Gamma.logGamma(betaSum) - Gamma.logGamma(nZ[z] + betaSum);
			LLnew += Gamma.logGamma(betaSumNew) - Gamma.logGamma(nZ[z] + betaSumNew);
			
			for (int w = 0; w < W; w++) {
				LLold += Gamma.logGamma(nZW[z][w] + beta) - Gamma.logGamma(beta);
				LLnew += Gamma.logGamma(nZW[z][w] + betaNew) - Gamma.logGamma(betaNew);
			}
		}
		
		double ratio = Math.exp(LLnew - LLold);
		
		boolean accept = false;
		if (r.nextDouble() < ratio) accept = true;
		if (betaNew > 0.5) accept = false; // hack
		
		System.out.println("beta: proposed "+betaNew);
		System.out.println(" (ratio = "+ratio);
		if (accept) {
			beta = betaNew;
			System.out.println("Accepted");
		} else {
			System.out.println("Rejected");
		}
		System.out.println("beta: "+beta);
		*/
	}
	
	public void updateLambda() {
		double step = stepA;
		
		double[] grad = new double[Z];
		double gradB = 0.0;
		
		for (int c = 0; c < C; c++) {
			if (nC[c] == 0) continue; // skip empty collections
			
			double[] Zbar = new double[Z];
			for (int z = 0; z < Z; z++) {
				Zbar[z] = (double)nCZ[c][z] / (double)nC[c];
			}
			
			double dot = 0.0;
			for (int z = 0; z < Z; z++) {
				dot += lambda[z] * Zbar[z];
			}
			dot += lambdaB;
			
			double diff = dot - gold[c];
			
			for (int z = 0; z < Z; z++) {
				grad[z] -= (diff * Zbar[z]) / Math.pow(sigma, 2);
			}
			gradB -= (diff) / Math.pow(sigma, 2);
		}
		
		// regularize
		for (int z = 0; z < Z; z++) {
			grad[z] -= lambda[z] / Math.pow(1.0, 2);
			adaLambda[z] += Math.pow(grad[z], 2);
		}
		gradB -= lambdaB / Math.pow(10.0, 2);
		adaLambdaB += Math.pow(gradB, 2);
		
		for (int z = 0; z < Z; z++) {
			lambda[z] += (step  / (Math.sqrt(adaLambda[z]) + MathUtils.eps)) * grad[z];
//			lambda[z] += step * grad[z];
//			System.out.println("lambda["+z+"] = "+lambda[z]);
		}
		lambdaB += (step /(Math.sqrt(adaLambdaB) + MathUtils.eps)) * gradB;
//		lambdaB += step * gradB;
//		System.out.println("lambdaB = "+lambdaB);
	}
	
	public double probResponse(int c) {
		return Math.exp(logProbResponse(c));
	}
	
	public double logProbResponse(int c) {
		double prob;
		
		double score = 0.0;
		for (int z = 0; z < Z; z++) {
			double Zbar = (double)nCZ[c][z] / (double)nC[c];
			score += lambda[z] * Zbar;
		}
		score += lambdaB;
		
		prob = Math.pow(score - gold[c], 2);
		prob /= 2.0 * Math.pow(sigma, 2);
		
		return -prob;
	}
	
	public void doSamplingIteration(int iter) {
		long startTimeS = System.currentTimeMillis();
		
		for (int d = 0; d < D; d++) {
			if (!computePerplexity) {
				for (int n = 0; n < docs[d].length; n++) {
					sample(d, n);
				}
			}
			else {
				for (int n = 0; n < docs[d].length; n+=2) {
					sample(d, n);
				}
			}
		}
		
		if (iter >= 200) {
			updateLambda();
			updateAlpha();
			updateBeta();
			
			updatePriors();
		}
		
		System.out.print("DeltaBias:");
		for (int z = 0; z < Z; z++) {
			System.out.print(" " + deltaBias[z]);
		}
		System.out.println("\nthetaNorm: " + thetaNorm);
		
		System.out.print("OmegaBias:");
		for (int w = 0; w < W; w+=(W/10)) {
			System.out.print(" " + omegaBias[w]);
		}
		System.out.println("\nphiNorm: " + phiNorm);
		
		System.out.print("Lambda:");
		for (int z = 0; z < Z; z++) {
			System.out.print(" " + lambda[z]);
		}
		System.out.println();
		
//		if (iter > 100) updateLambda();
//		if (iter > 20 && iter % 10 == 0) updateAlpha();
//		if (iter >= 100 && iter % 10 == 0) updateBeta();
		
//		System.out.println("Iter % likelihoodFreq: " + iter + " " + likelihoodFreq + " " + iter % likelihoodFreq);
		if ((iter % likelihoodFreq) == 0) {
			System.out.println("LL: " + computeLL());
			
			if (computePerplexity) {
				System.out.println("Train perplexity: " + computePerplexity(false));
				System.out.println("Held-out perplexity: " + computePerplexity(true));
			}
		}
		
		// collect samples (docsZZ) 
		if (burnedIn) {
			for (int d = 0; d < D; d++) {
				for (int n = 0; n < docs[d].length; n++) { 
					int z = docsZ[d][n];
					
					docsZZ[d][n][z] += 1;
				}
			}
		}
		
		long endTimeS = System.currentTimeMillis();
		double sec = (double)(endTimeS-startTimeS)/1000.0;
		System.out.println("time per iter: "+sec);
	}
	
	public void sample(int d, int n) {
		int w = docs[d][n];
		int c = docsC[d];
		int topic = docsZ[d][n];
		int level = docsX[d][n];
		
		// decrement counts
		
		nX[level] -= 1;
		
		nDZ[d][topic] -= 1;
		nD[d] -= 1;
		nZW[topic][w] -= 1;
		nZ[topic] -= 1;
		
		if (c >= 0) {
			nCZ[c][topic] -= 1;
			nC[c] -= 1;
		}
		
//		double betaNorm = W * beta;
		
		// sample new value for level
		
		double pTotal = 0.0;
		double[] p = new double[Z+1];
		
		// this will be p(x=0)	
//		p[Z] = (nX[0] + gamma0) *
//			(nBW[w] + beta) / (nB + betaNorm) * 
//			probResponse(c);
//		pTotal += p[Z];
		p[Z] = 0; //disable
		
		// sample new value for topic and level
		
		for (int z = 0; z < Z; z++) {
			if (c >= 0) {
				nCZ[c][z] += 1;
				nC[c] += 1;
			}
			
			double probResp = c < 0 ? 1.0 : probResponse(c);
			
			p[z] = (nDZ[d][z] + priorDZ[z]) / (nD[d] + thetaNorm) *
				(nZW[z][w] + priorZW[w]) / (nZ[z] + phiNorm) *
				probResp;
//			p[z] = (nX[1] + gamma1) * 
//					(nDZ[d][z] + deltaBias[z]) / (nD[d] + deltaBiasSum) *
//					(nZW[z][w] + beta) / (nZ[z] + betaNorm) *
//					probResponse(c);
			pTotal += p[z];
			
			if (c >= 0) {
				nCZ[c][z] -= 1;
				nC[c] -= 1;
			}
		}
		
		Random r = new Random();
		double u = r.nextDouble() * pTotal;
		
		double v = 0.0;
		for (int z = 0; z < Z; z++) {
			v += p[z];
			
			if (v > u) {
				topic = z;
				break;
			}
		}
		
		nDZ[d][topic] += 1;
		nD[d] += 1;
		nZW[topic][w] += 1;
		nZ[topic] += 1;
		
		if (c >= 0) {
			nCZ[c][topic] += 1;
			nC[c] += 1;
		}
		
		// set new assignments
		
		docsZ[d][n] = topic;
	}
	
	/**
	 * Format of input file:
	 * 
	 * DOC_ID\tFOLD\tCOLLECTION_ID\tFEATURE\tword1 word2 ... wordn
	 */
	public void readDocs(String filename) throws Exception {
		System.out.println("Reading input...");
		
		wordMap = new HashMap<String,Integer>();
		wordMapInv = new HashMap<Integer,String>();
		
		FileReader fr = new FileReader(filename);
		BufferedReader br = new BufferedReader(fr); 
		
		String s;
		
		D = 0;
		C = 0;
		int dj = 0;
		while((s = br.readLine()) != null) {
			//if (dj++ % 20 != 0) continue;
			D++;
			
			String[] cols = s.split("\t");
			int c = Integer.parseInt(cols[2]);
			if (c > C) C = c;
		}
		C++;
		
		docIds   = new BigInteger[D];
		docFolds = new int[D];
		docsC    = new int[D];
		gold     = new double[C];
		docs     = new int[D][];
		
		fr.close();
		fr = new FileReader(filename);
		br = new BufferedReader(fr); 
		
		int d = 0;
		int di = 0;
		while ((s = br.readLine()) != null) {
			//if (di++ % 20 != 0) continue;
			String[] cols = s.split("\t");
			
			BigInteger id = new BigInteger(cols[0]);
			int fold = Integer.parseInt(cols[1]);
			int c = Integer.parseInt(cols[2]);
			
			docIds[d] = id;
			docFolds[d] = fold;
			docsC[d] = c;
			
			if (c >= 0)
				gold[c] = Double.parseDouble(cols[3]);
			
			String[] tokens = null;
			
			try {
				tokens = cols[4].split("\\s+");
			}
			catch (Exception e) {
				System.out.println("Bad line: " + s);
				System.out.print("Not enough columns:");
				for (int i = 0; i < cols.length; i++) {
					System.out.print(" " + cols[i]);
				}
				System.out.println();
				e.printStackTrace();
				
				tokens = new String[] {};
			}
			
			int N = tokens.length;
			docs[d] = new int[N];
			for (int n = 0; n < N; n++) {
				String word = tokens[n];
				
				int key = wordMap.size();
				if (!wordMap.containsKey(word)) {
					wordMap.put(word, new Integer(key));
					wordMapInv.put(new Integer(key), word);
				}
				else {
					key = ((Integer) wordMap.get(word)).intValue();
				}
				
				docs[d][n] = key;
			}
			
			d++;
		}
		
		br.close();
		fr.close();
		
		W = wordMap.size();
		
		System.out.println(D+" documents");
		System.out.println(W+" word types");
		System.out.println(C+" labels");
	}
	
	public void writeOutput(String filename) throws Exception {
		writeOutput(filename, new File(filename).getParent());
	}
	
	@Override
	public void logIteration() { }
	
	@Override
	public void collectSamples() {	}
	
	@Override
	public double computeLL(int[][][] corpus) { return -1.0; }
	
	public double computeLL() {
		double LL = 0;
		
		for (int d = 0; d < D; d++) {
			for (int n = 0; n < docs[d].length; n ++) {
				int w = docs[d][n];
				
				double tokenLL = 0;
				
				// marginalize over z
				
				for (int z = 0; z < Z; z++) {
					tokenLL += (nDZ[d][z] + priorDZ[z]) / (nD[d] + thetaNorm)*
							(nZW[z][w] + priorZW[w]) / (nZ[z] + phiNorm);
				}
				
				LL += MathUtils.log(tokenLL, 2.0);
			}
		}
		
		// Take the probability of the response into account
		for (int c = 0; c < C; c++) {
			LL += logProbResponse(c);
		}
		
		//System.out.println("Denom: " + denom);
		return LL;
	}
	
	public double computePerplexity(boolean heldout) {
		double LL = 0;
		
		int start = heldout ? 1 : 0;
		int N = 0;
		
		for (int d = 0; d < D; d++) {
			for (int n = start; n < docs[d].length; n+=2) {
				int w = docs[d][n];
				
				double tokenLL = 0;
				
				// marginalize over z
				
				for (int z = 0; z < Z; z++) {
					tokenLL += (nDZ[d][z] + priorDZ[z]) / (nD[d] + thetaNorm)*
							(nZW[z][w] + priorZW[w]) / (nZ[z] + phiNorm);
				}
				
				LL += MathUtils.log(tokenLL, 2.0);
				N++;
			}
		}
		
		double perplexity = Math.pow(2.0, -LL/N);
		
		return perplexity;
	}
	
	@Override
	protected void initTest() { }

	@Override
	public void writeOutput(String filename, String outputDir) throws Exception {
		System.out.println("Writing output...");
		
		FileWriter fw = new FileWriter(new File(outputDir, new File(filename).getName() + ".cslda.assign"));
		BufferedWriter bw = new BufferedWriter(fw);
		
		for (int d = 0; d < D; d++) {
			bw.write(docIds[d] + " ");
			
			if (docsC[d] >= 0) {
				bw.write(gold[docsC[d]] + " ");
			}
			else {
				bw.write("0.0 ");
			}
//			bw.write(docsC[d]+" ");
			
			for (int n = 0; n < docs[d].length; n++) {
				String word = wordMapInv.get(docs[d][n]);
				
				//bw.write(word+":"+docsZ[d][n]+" "); // only current sample
				bw.write(word);  // for multiple samples
				for (int zz = 0; zz < Z; zz++) {
					bw.write(":"+docsZZ[d][n][zz]);
				}
				bw.write(" ");
			}
			bw.newLine();
		}
		
		bw.close();
		fw.close();

		fw = new FileWriter(new File(outputDir, new File(filename).getName() + ".cslda.lambda"));
		bw = new BufferedWriter(fw);
		
		for (int z = 0; z < Z; z++) {
			bw.write(""+lambda[z]);
			bw.newLine();
		}
		bw.write("B "+lambdaB);
		bw.newLine();
		
		bw.close();
		fw.close();

		fw = new FileWriter(new File(outputDir, new File(filename).getName() + ".cslda.deltabias"));
		bw = new BufferedWriter(fw); 		
		
		for (int z = 0; z < Z; z++) {
			bw.write(""+deltaBias[z]);
			bw.newLine();
		}
		
		bw.close();
		fw.close();
		
		fw = new FileWriter(new File(outputDir, new File(filename).getName() + ".cslda.omegabias"));
		bw = new BufferedWriter(fw); 		
		
		for (int w = 0; w < W; w++) {
			bw.write(wordMapInv.get(w) + " " + omegaBias[w] + "\n");
		}
		
		bw.close();
		fw.close();
	}
	
	@Override
	public void cleanUp() throws Exception { }
}
