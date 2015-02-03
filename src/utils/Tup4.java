package utils;

public class Tup4<V1, V2, V3, V4> {
	
	private V1 v1;
	private V2 v2;
	private V3 v3;
	private V4 v4;
	
	public Tup4(V1 v10, V2 v20, V3 v30, V4 v40) {
		v1 = v10;
		v2 = v20;
		v3 = v30;
		v4 = v40;
	}
	
	public V1 _1() { return v1; };
	public V2 _2() { return v2; };
	public V3 _3() { return v3; };
	public V4 _4() { return v4; };
	
}
