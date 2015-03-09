package utils;

public class Tup2<V1, V2> implements Comparable<Tup2<V1, V2>> {
	
	private V1 v1;
	private V2 v2;
	
	public Tup2(V1 v10, V2 v20) {
		v1 = v10;
		v2 = v20;
	}
	
	public V1 _1() { return v1; };
	public V2 _2() { return v2; }
	
	@Override
	@SuppressWarnings("unchecked")
	public int compareTo(Tup2<V1, V2> o) {
		Comparable<V1> this1 = (Comparable<V1>)v1;
		Comparable<V2> this2 = (Comparable<V2>)v2;
		
		int lt1 = this1.compareTo(o.v1);
		if (lt1 == 0) {
			return this2.compareTo(o.v2);
		}
		else {
			return lt1;
		}
	};
	
}
