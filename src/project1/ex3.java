package project1;
import java.util.Random;
public class ex3 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		int r[]=new int[10];
		Random R =new Random();
		int sum=0;
		System.out.printf("random number: \n");
		for(int i=0;i<10;i++) {
			r[i]=R.nextInt(100)+1;
			for(int j=0;j<i;j++) {
				if(r[i]==r[j]) {
					i--;
				}
			}
			sum+=r[i];
			System.out.printf("%d\n", r[i]);
		}
		System.out.printf("random number sum: %d", sum);

	}

}
