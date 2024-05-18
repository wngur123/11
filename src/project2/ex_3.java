package project2;

public class ex_3 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int sum=0;
		for(int i=1;i<=100;i++) {
			if(i%3==0)
				continue;
			sum+=i;
		}
		System.out.printf("1~100까지의 합(3의 배수 제외): %d\n", sum);

	}

}
