package project1;
import java.util.Scanner;
public class ex1 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner s=new Scanner(System.in);
		int score;
		System.out.printf("enter your score: ");
		score=s.nextInt();
		if(score>=90) {
			System.out.printf("A grade");
		}
		else if(score>=80) {
			System.out.printf("B grade");
		}
		else if(score>=70) {
			System.out.printf("C grade");
		}
		else if(score>=60) {
			System.out.printf("D grade");
		}
		else{
			System.out.printf("F grade");
		}





	}

}
