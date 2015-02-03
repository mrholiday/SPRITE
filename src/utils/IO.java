package utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigInteger;
import java.util.HashMap;
import java.util.Map;



/**
 * Methods to read an input document and write to log file.
 * 
 * @author adrianb
 *
 */
public class IO {
	
	/**
	 * Loads documents, along with factor scores/values for each, and alphabet.  Fields are
	 * separated by tabs, scores for each component in a factor and words are separated
	 * by spaces.  Factor component scores should each be floats.  I could have inferred
	 * the number of views and components for each factor by reading the file, but I think
	 * this may introduce subtle bugs if the input file is incorrectly formatted.
	 * 
	 * @param inputPath   Where your data is sitting.
	 * @param observedFactorSizes Array containing the number of components in each factor.
	 * @param numViews    Number of document views to read in.
	 * 
	 * @return (Alphabet, InverseAlphabet, ObservedFactorValues, DocumentTextByView)
	 */
	public static Tup5<BigInteger[], Map<String, Integer>[], Map<Integer, String>[], double[][][], int[][][]> readTrainInput(String inputPath,
			                                                                                                   int[] observedFactorSizes,
			                                                                                                   int numViews) {
		Log.info("io", "Reading training input...");
		
		int numFactorsObserved = observedFactorSizes.length;
		
		BigInteger[] docIds = null;
		int[][][] docs = null;
		double[][][] observedFactors = null;
		
		@SuppressWarnings("unchecked")
		Map<String, Integer>[] wordMaps    = new HashMap[numViews];
		
		@SuppressWarnings("unchecked")
		Map<Integer, String>[] wordMapInvs = new HashMap[numViews];
		
		try {
			FileReader fr = new FileReader(inputPath);
			BufferedReader br = new BufferedReader(fr); 
			
			String s;
			
			int D = 0;
			while((s = br.readLine()) != null) {
				D++;
			}
			fr.close();
			
			docIds = new BigInteger[D];
			docs = new int[D][numViews][];
			observedFactors = new double[numFactorsObserved][D][];

			fr = new FileReader(inputPath);
			br = new BufferedReader(fr); 

			int d = 0;
			while ((s = br.readLine()) != null) {
				String[] fields = s.split("\t+");

				for (int i = 0; i < numFactorsObserved; i++) {
					observedFactors[i][d] = new double[observedFactorSizes[i]];

					String[] factorValues = fields[1+i].split(" ");
					for (int j = 0; j < observedFactorSizes[i]; j++) {
						observedFactors[i][d][j] = Double.parseDouble(factorValues[j]);
					}
				}
				
				for (int v = 0; v < numViews; v++) {
					String[] tokens = fields[ v + 1 + numFactorsObserved ].split(" ");
					docs[d][v] = new int[tokens.length];
					
					for (int n = 0; n < tokens.length; n++) {
						String t = tokens[n];
						
						int key = wordMaps[v].size();
						if (!wordMaps[v].containsKey(t)) {
							wordMaps[v].put(t, key);
							wordMapInvs[v].put(key, t);
						}
						else {
							key = wordMaps[v].get(t).intValue();
						}

						docs[d][v][n] = key;
					}
				}

				d++;
			}
			
			br.close();
			fr.close();
			
			Log.info("io", D + " documents loaded for training");
			
			int[] Ws = new int[numViews];
			
			for (int v = 0; v < numViews; v++) {
				Ws[v] = wordMaps[v].size();
				Log.info("io", Ws[v] + " word types in view " + v);
			}
			
		} catch (FileNotFoundException e) {
			Log.error("io", "Could not find input file: " + inputPath, e);
		} catch (IOException e) {
			Log.error("io", "Problem reading/closing file: " + inputPath, e);
		}
		
		return new Tup5<BigInteger[], Map<String, Integer>[],
				        Map<Integer, String>[],
				        double[][][],
				        int[][][]>( docIds, wordMaps, wordMapInvs, observedFactors, docs );
	}
	
	/**
	 * Called when reading documents for prediction/inference.  Strings not in wordMap are ignored.
	 * 
	 * @param inputPath   Where your data is sitting.
	 * @param factorSizes Array containing the number of components in each factor.
	 * @param numViews    Number of document views to read in.
	 * @param wordMap     Mapping form word to index.
	 * 
	 * @return (ObservedFactorValues, DocumentTextByView)
	 */
	public static Tup3<BigInteger[], double[][][], int[][][]> readPredictionInput(String inputPath,
			                                                        int[] factorSizes,
			                                                        int numViews,
			                                                        Map<String, Integer>[] wordMaps) {
		Log.debug("io", "Reading prediction input...");
		
		int numFactorsObserved = factorSizes.length;
		
		BigInteger[] docIds = null;
		int[][][] docs = null;
		double[][][] observedFactors = null;
		
		try {
			FileReader fr = new FileReader(inputPath);
			BufferedReader br = new BufferedReader(fr); 

			String s;
			
			int D = 0;
			while((s = br.readLine()) != null) {
				D++;
			}
			fr.close();
			
			docIds = new BigInteger[D];
			docs = new int[D][numViews][];
			observedFactors = new double[numFactorsObserved][D][];
			
			fr = new FileReader(inputPath);
			br = new BufferedReader(fr); 
			
			int d = 0;
			while ((s = br.readLine()) != null) {
				String[] fields = s.split("\t+");
				
				for (int i = 0; i < numFactorsObserved; i++) {
					observedFactors[i][d] = new double[factorSizes[i]];

					String[] factorValues = fields[1+i].split(" ");
					for (int j = 0; j < factorSizes[i]; j++) {
						observedFactors[i][d][j] = Double.parseDouble(factorValues[j]);
					}
				}
				
				for (int v = 0; v < numViews; v++) {
					String[] tokens = fields[ v + 1 + numFactorsObserved ].split(" ");
					int tokenIndex = 0;
					for (String t : tokens) {
						if (wordMaps[v].containsKey(t)) {
							tokenIndex++;
						}
					}
					
					docs[d][v] = new int[tokenIndex];
					
					tokenIndex = 0;
					for (int n = 0; n < tokens.length; n++) {
						String t = tokens[n];
						
						if (wordMaps[v].containsKey(t)) {
							docs[d][v][n] = wordMaps[v].get(t);
							tokenIndex++;
						}
					}
				}

				d++;
			}
			
			Log.info("io", D + " documents loaded for prediction");
			
			br.close();
			fr.close();
		} catch (FileNotFoundException e) {
			Log.error("io", "Could not find input file: " + inputPath, e);
		} catch (IOException e) {
			Log.error("io", "Problem reading/closing file: " + inputPath, e);
		}
		
		return new Tup3<BigInteger[], double[][][], int[][][]>(docIds, observedFactors, docs);
	}
	
}
