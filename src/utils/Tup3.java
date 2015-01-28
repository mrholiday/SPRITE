package utils;

public class Tup3<V1, V2, V3> {
	
	private V1 v1;
	private V2 v2;
	private V3 v3;
	
	public Tup3(V1 v10, V2 v20, V3 v30) {
		v1 = v10;
		v2 = v20;
		v3 = v30;
	}
	
	public V1 _1() { return v1; };
	public V2 _2() { return v2; };
	public V3 _3() { return v3; };
	
}
