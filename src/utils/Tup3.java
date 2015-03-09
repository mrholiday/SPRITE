package utils;

public class Tup3<V1, V2, V3> implements Comparable<Tup3<V1, V2, V3>> {
	
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
	
	@Override
	@SuppressWarnings("unchecked")
	public int compareTo(Tup3<V1, V2, V3> o) {
		Comparable<V1> this1 = (Comparable<V1>)v1;
		Comparable<V2> this2 = (Comparable<V2>)v2;
		Comparable<V3> this3 = (Comparable<V3>)v3;
		
		int lt1 = this1.compareTo(o.v1);
		if (lt1 == 0) {
			int lt2 = this2.compareTo(o.v2);
			
			if (lt2 == 0) {
				return this3.compareTo(o.v3);
			}
			else {
				return lt2;
			}
		}
		else {
			return lt1;
		}
	};

}
