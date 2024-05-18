package project2;

public class ex_4 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int sum=0;
		for(int i=1;i<=100;i++) {
			sum+=i;
		}
		System.out.printf("1부터 100까지의 합은 %d입니다. \n", sum);
		if(sum>5000)
			return;
		System.out.printf("프로그램의 끝입니다.");

	}

}
