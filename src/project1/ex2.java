package project1;
import java.util.Scanner;
public class ex2 {
	static int comparision(int score) {
		if(score>=90) {
			return 5;
		}
		else if(score>=80) {
			return 4;
		}
		else if(score>=70) {
			return 3;
		}
		else if(score>=60) {
			return 2;
		}
		else  {
			return 1;
		}
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner s=new Scanner(System.in);
		int score;
		int a;
		System.out.printf("enter your score: ");
		score=s.nextInt();
		a=comparision(score);
		switch(a) {
		case 5:
			System.out.printf("A grade");
			break;
		case 4:
			System.out.printf("B grade");
			break;
		case 3:
			System.out.printf("C grade");
			break;
		case 2:
			System.out.printf("D grade");
			break;
		case 1:
			System.out.printf("F grade");
			break;
		}

	}

}
