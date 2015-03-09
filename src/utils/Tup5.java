package utils;

public class Tup5<V1, V2, V3, V4, V5> implements Comparable<Tup5<V1, V2, V3, V4, V5>>{
	
	private V1 v1;
	private V2 v2;
	private V3 v3;
	private V4 v4;
	private V5 v5;
	
	public Tup5(V1 v10, V2 v20, V3 v30, V4 v40, V5 v50) {
		v1 = v10;
		v2 = v20;
		v3 = v30;
		v4 = v40;
		v5 = v50;
	}
	
	public V1 _1() { return v1; };
	public V2 _2() { return v2; };
	public V3 _3() { return v3; };
	public V4 _4() { return v4; };
	public V5 _5() { return v5; };
	
	@Override
	@SuppressWarnings("unchecked")
	public int compareTo(Tup5<V1, V2, V3, V4, V5> o) {
		Comparable<V1> this1 = (Comparable<V1>)v1;
		Comparable<V2> this2 = (Comparable<V2>)v2;
		Comparable<V3> this3 = (Comparable<V3>)v3;
		Comparable<V4> this4 = (Comparable<V4>)v4;
		Comparable<V5> this5 = (Comparable<V5>)v5;
		
		int lt1 = this1.compareTo(o.v1);
		if (lt1 == 0) {
			int lt2 = this2.compareTo(o.v2);
			
			if (lt2 == 0) {
				int lt3 = this3.compareTo(o.v3);
				
				if (lt3 == 0) {
					int lt4 = this4.compareTo(o.v4);
					
					if (lt4 == 0) {
						return this5.compareTo(o.v5);
					}
					else {
						return lt4;
					}
				}
				else {
					return lt3;
				}
			}
			else {
				return lt2;
			}
		}
		else {
			return lt1;
		}
	}
	
}
