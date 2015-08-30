package utils;

import java.lang.reflect.Array;

public class Utils {

	private Utils() { }
	
	public static String mkStr(int[] arr0) {
		Object[] arr = new Object[arr0.length];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = Array.get(arr0, i);
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(double[] arr0) {
		Object[] arr = new Object[arr0.length];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = Array.get(arr0, i);
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(float[] arr0) {
		Object[] arr = new Object[arr0.length];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = Array.get(arr0, i);
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(boolean[] arr0) {
		Object[] arr = new Object[arr0.length];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = Array.get(arr0, i);
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(char[] arr0) {
		Object[] arr = new Object[arr0.length];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = Array.get(arr0, i);
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(Object[] arr) {
		if (arr.length == 0) {
			return "[]";
		}
		
		StringBuilder b = new StringBuilder();
		
		b.append("[" + arr[0].toString());
		for (int i = 1; i < arr.length; i++) {
			String v = arr[i] == null ? "null" : arr[i].toString();
			b.append("," + v);
		}
		b.append("]");
		
		return b.toString();
	}
	
	public static String mkStr(int[][] arr0) {
		Object[][] arr = new Object[arr0.length][];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = new Object[arr0[i].length];
			for (int j = 0; j < arr0[i].length; j++) {
				arr[i][j] = Array.get(arr0[i], j);
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(double[][] arr0) {
		Object[][] arr = new Object[arr0.length][];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = new Object[arr0[i].length];
			for (int j = 0; j < arr0[i].length; j++) {
				arr[i][j] = Array.get(arr0[i], j);
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(float[][] arr0) {
		Object[][] arr = new Object[arr0.length][];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = new Object[arr0[i].length];
			for (int j = 0; j < arr0[i].length; j++) {
				arr[i][j] = Array.get(arr0[i], j);
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(boolean[][] arr0) {
		Object[][] arr = new Object[arr0.length][];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = new Object[arr0[i].length];
			for (int j = 0; j < arr0[i].length; j++) {
				arr[i][j] = Array.get(arr0[i], j);
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(char[][] arr0) {
		Object[][] arr = new Object[arr0.length][];
		for (int i = 0; i < arr0.length; i++) {
			arr[i] = new Object[arr0[i].length];
			for (int j = 0; j < arr0[i].length; j++) {
				arr[i][j] = Array.get(arr0[i], j);
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(Object[][] arr) {
		if (arr.length == 0) {
			return "[]";
		}
		
		StringBuilder b = new StringBuilder();
		
		b.append("[");
		
		for (int i = 0; i < arr.length; i++) {
			if (arr[i].length == 0) {
			    b.append("[]");
				
			    if ((i+1) < arr.length) {
			    	b.append(",");
			    }
			    
			    continue;
			}
			
			b.append("[" + arr[i][0].toString());
			
			for (int j = 1; j < arr[i].length; j++) {
				String v = arr[i][j] == null ? "null" : arr[i][j].toString();
				b.append("," + v);
			}
			
			b.append("]");
			
			if ((i+1) < arr.length) {
				b.append(",");
			}
		}
		
		b.append("]");
		
		return b.toString();
	}
	
	public static String mkStr(char[][][] arr0) {
		Object[][][] arr = new Object[arr0.length][][];
		for (int k = 0; k < arr0.length; k++) {
			arr[k] = new Object[arr0[k].length][];
			
			for (int i = 0; i < arr0[k].length; i++) {
				arr[k][i] = new Object[arr0[k][i].length];
				for (int j = 0; j < arr0[k][i].length; j++) {
					arr[k][i][j] = Array.get(arr0[k][i], j);
				}
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(double[][][] arr0) {
		Object[][][] arr = new Object[arr0.length][][];
		for (int k = 0; k < arr0.length; k++) {
			arr[k] = new Object[arr0[k].length][];
			
			for (int i = 0; i < arr0[k].length; i++) {
				arr[k][i] = new Object[arr0[k][i].length];
				for (int j = 0; j < arr0[k][i].length; j++) {
					arr[k][i][j] = Array.get(arr0[k][i], j);
				}
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(int[][][] arr0) {
		Object[][][] arr = new Object[arr0.length][][];
		for (int k = 0; k < arr0.length; k++) {
			arr[k] = new Object[arr0[k].length][];
			
			for (int i = 0; i < arr0[k].length; i++) {
				arr[k][i] = new Object[arr0[k][i].length];
				for (int j = 0; j < arr0[k][i].length; j++) {
					arr[k][i][j] = Array.get(arr0[k][i], j);
				}
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(float[][][] arr0) {
		Object[][][] arr = new Object[arr0.length][][];
		for (int k = 0; k < arr0.length; k++) {
			arr[k] = new Object[arr0[k].length][];
			
			for (int i = 0; i < arr0[k].length; i++) {
				arr[k][i] = new Object[arr0[k][i].length];
				for (int j = 0; j < arr0[k][i].length; j++) {
					arr[k][i][j] = Array.get(arr0[k][i], j);
				}
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(boolean[][][] arr0) {
		Object[][][] arr = new Object[arr0.length][][];
		for (int k = 0; k < arr0.length; k++) {
			arr[k] = new Object[arr0[k].length][];
			
			for (int i = 0; i < arr0[k].length; i++) {
				arr[k][i] = new Object[arr0[k][i].length];
				for (int j = 0; j < arr0[k][i].length; j++) {
					arr[k][i][j] = Array.get(arr0[k][i], j);
				}
			}
		}
		
		return mkStr(arr);
	}
	
	public static String mkStr(Object[][][] arr) {
		if (arr.length == 0) {
			return "[]";
		}
		
		StringBuilder b = new StringBuilder();
		
		b.append("[");
		
		for (int k = 0; k < arr.length; k++) {
			
			if (arr[k].length == 0) {
				b.append("[]");
				
				if ((k+1) < arr.length) {
					b.append(",");
				}
				
				continue;
			}
			
			b.append("[");
			
			for (int i = 0; i < arr[k].length; i++) {
				if (arr[k][i].length == 0) {
					b.append("[]");
					
					if ((i+1) < arr[k][i].length) {
						b.append(",");
					}
					
					continue;
				}
				
				b.append("[" + arr[k][i][0].toString());
				
				for (int j = 1; j < arr[k][i].length; j++) {
					String v = arr[k][i][j] == null ? "null" : arr[k][i][j].toString();
					b.append("," + v);
				}
				
				b.append("]");
				
				if ((i+1) < arr[k].length) {
					b.append(",");
				}
			}
			
			b.append("]");
			
			if ((k+1) < arr.length) {
				b.append(",");
			}
		}

		b.append("]");

		return b.toString();
	}
	
	public static void main(String[] args) {
		int[] arr0 = new int[] {0, 5, 6, 10, -11};
		double[][] arr1 = new double[][] {{1.0, -5.0, 3.3}, {1.0, 10.0}, {-1.0, 1.0, -2.0, 2.0, -5.0, 5.0}};
		boolean[][][] arr2 = new boolean[][][] {{{true}, {false}}, {{false, false}, {true, true}, {false, true}}, {{true, false, true}}};
		
		System.out.println("arr0: " + mkStr(arr0));
		System.out.println("arr1: " + mkStr(arr1));
		System.out.println("arr2: " + mkStr(arr2));
	}
	
}
