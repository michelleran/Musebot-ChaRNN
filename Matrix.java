package com.mran.charnn;

import java.util.Random;


/******************************************************************************
 *  Compilation:  javac Matrix.java
 *  Execution:    java Matrix
 *
 *  A bare-bones collection of static methods for manipulating
 *  matrices.
 *
 ******************************************************************************/

public class Matrix {

    // return a random m-by-n matrix with values between 0 and 1
    public static double[][] random(int m, int n) {
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                a[i][j] = StdRandom.uniform(0.0, 1.0);
        return a;
    }

    public static double[] random(int m) {
        double[] a = new double[m];
        for (int i = 0; i < m; i++)
             a[i] = StdRandom.uniform(0.0, 1.0);
        return a;
    }
    
    // return n-by-n identity matrix I
    public static double[][] identity(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
            a[i][i] = 1;
        return a;
    }

    // return x^T y
    public static double dot(double[] x, double[] y) {
        if (x.length != y.length) throw new RuntimeException("Illegal vector dimensions.");
        double sum = 0.0;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    // return B = A^T
    public static double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] b = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                b[j][i] = a[i][j];
        return b;
    }

    // return c = a + b
    public static double[][] add(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c[i][j] = a[i][j] + b[i][j];
        return c;
    }
    
    public static double[] add(double[] a, double[] b) {
    	double[] sum = new double[a.length];
    	for (int i = 0; i < a.length; i++) {
    		sum[i] = a[i] + b[i];
    	}
    	return sum;
    }
    
    public static double[] add(double[] a, double b) {
    	double[] sum = new double[a.length];
    	for (int i = 0; i < a.length; i++) {
    		sum[i] = a[i] + b;
    	}
    	return sum;
    }
    
    public static double[][] add(double[][] a, double b) {
    	double[][] sum = new double[a.length][a[0].length];
    	for (int i = 0; i < a.length; i++) {
    		for (int j = 0; j < a[0].length; j++) {
    			sum[i][j] = a[i][j] + b;
    		}
    	}
    	return sum;
    }

    // return c = a - b
    public static double[][] subtract(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c[i][j] = a[i][j] - b[i][j];
        return c;
    }
    
    public static double[] subtract(double a, double[] b) {
    	double[] result = new double[b.length];
    	for (int i = 0; i < b.length; i++) {
    		result[i] = a-b[i];
    	}
    	return result;
    }

    // return c = a * b
    public static double[][] multiply(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) throw new RuntimeException("Illegal matrix dimensions.");
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n2; j++)
                for (int k = 0; k < n1; k++)
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }

    // matrix-vector multiplication (y = A * x)
    public static double[] multiply(double[][] a, double[] x) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                y[i] += a[i][j] * x[j];
        return y;
    }


    // vector-matrix multiplication (y = x^T A)
    public static double[] multiply(double[] x, double[][] a) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != m) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                y[j] += a[i][j] * x[i];
        return y;
    }
    
    // <a> is mx1, <b> is 1xn
    public static double[][] multiply(double[] a, double[] b) {
    	int m = a.length;
    	int n = b.length;
    	double[][] c = new double[m][n];
    	for (int i = 0; i < m; i++) {
    		for (int j = 0; j < n; j++) {
    			c[i][j] = a[i]*b[j];
    		}
    	}
    	return c;
    }
    
    // matrix scalar multiplication
    public static double[][] multiply(double[][] matrix, double scale){
    	double[][] result = new double[matrix.length][matrix[0].length];
    	for(int r = 0; r < matrix.length; r++) {
    		for(int c = 0; c < matrix[0].length; c++) {
    			result[r][c] = matrix[r][c] * scale;
    		}
    	}
    	return result;
    }
    
    // vector scalar multiplication
    public static double[] multiply(double[] vector, double scale){
    	double[] result = new double[vector.length];
    	for(int i = 0; i < vector.length; i++) {
    		result[i] = vector[i] * scale;
    	}
    	return result;
    }
    
    // element-wise multiplication; a and b must have same shape
    public static double[] elementMult(double[] a, double[] b) {
    	double[] result = new double[a.length];
    	for (int i = 0; i < a.length; i++) {
    		result[i] = a[i]*b[i];
    	}
    	return result;
    }
    
    public static double[][] elementMult(double[][] a, double[][] b) {
    	double[][] result = new double[a.length][a[0].length];
    	for (int i = 0; i < a.length; i++) {
    		for (int j = 0; j < a[0].length; j++) {
    			result[i][j] = a[i][j] * b[i][j];
    		}
    	}
    	return result;
    }
    
    // element-wise vector division: divide <vector> by <by>
    public static double[] divide(double[] vector, double by) {
    	double[] result = new double[vector.length];
    	for(int i = 0; i < vector.length; i++) {
			result[i] = vector[i] / by;
		}
    	return result;
    }
    
    public static double[] divide(double[] a, double[] b) {
    	double[] result = new double[a.length];
    	for(int i = 0; i < a.length; i++) {
			result[i] = a[i] / b[i];
		}
    	return result;
    }
    
    // requirements: a and b are same shape
    public static double[][] divide(double[][] a, double[][] b) {
    	double[][] result = new double[a.length][a[0].length];
    	for (int i = 0; i < a.length; i++) {
    		for (int j = 0; j < a[0].length; j++) {
    			result[i][j] = a[i][j] / b[i][j];
    		}
    	}
    	return result;
    }
    
    // element-wise sqrt
    public static double[][] sqrt(double[][] a) {
    	double[][] result = new double[a.length][a[0].length];
    	for (int i = 0; i < a.length; i++) {
    		for (int j = 0; j < a[0].length; j++) {
    			result[i][j] = Math.sqrt(a[i][j]);
    		}
    	}
    	return result;
    }
    
    public static double[] sqrt(double[] a) {
    	double[] result = new double[a.length];
    	for (int i = 0; i < a.length; i++) {
    		result[i] = Math.sqrt(a[i]);
    	}
    	return result;
    }
    
    public static double[] exp(double[] vector) {
    	double[] result = new double[vector.length];
    	for (int i = 0; i < vector.length; i++) {
    		result[i] = Math.exp(vector[i]);
    	}
    	return result;
    }
    
    public static double[] tanh(double[] vector) {
    	double[] result = new double[vector.length];
    	for (int i = 0; i < vector.length; i++) {
    		result[i] = Math.tanh(vector[i]);
    	}
    	return result;
    }
    
    public static double sum(double[] vector) {
    	double result = 0;
    	for (double d : vector) { result += d; }
    	//if (result <= 0.000001) { return 0.000001; }
    	return result;
    }
    
    public static double[] softmax(double[] vector) {
    	return divide(exp(vector), sum(exp(vector)));
    }
    
    public static double sum(double[][] m) {
    	double result = 0;
    	for(int i = 0; i < m.length; i++) {
    		double[] vector = m[i];
    		for (double d : vector) { result += d; }
    	}
    	return result;
    }
    
    public static double[][] clip(double[][] a, double lower, double upper) {
    	double[][] b = a.clone();
    	for (int i = 0; i < a.length; i++) {
    		for (int j = 0; j < a[0].length; j++) {
    			if (b[i][j] < lower) { b[i][j] = lower; }
    			if (b[i][j] > upper) { b[i][j] = upper; }
    		}
    	}
    	return b;
    }
    
    public static double[] clip(double[] a, double lower, double upper) {
    	double[] b = a.clone();
    	for (int i = 0; i < a.length; i++) {
    		if (b[i] < lower) { b[i] = lower; }
			if (b[i] > upper) { b[i] = upper; }
    	}
    	return b;
    }
    
    public static double[] ravel(double[][] a) {
    	double[] b = new double[a.length*a[0].length];
    	int index = 0;
    	for (double[] row : a) {
    		for (double item : row) {
    			b[index] = item;
    			index++;
    		}
    	}
    	return b;
    }
    
    public static void zero(double[][] a) {
     	for (int i = 0; i < a.length; i++) {
    		for (int j = 0; j < a[0].length; j++) {
    			a[i][j] = 0.0;
    		}
    	}    	
    }
 
    public static void zero(double[] a) {
     	for (int i = 0; i < a.length; i++) {
    			a[i] = 0.0;
    	}    	
    }
    
    public static int randomChoice(int range, double[] probabilities) {
    	// assuming range is one by one, so range should == probabilities.length
    	double completeWeight = 0.0;
        for (double p : probabilities)
            completeWeight += p;
        double r = Math.random() * completeWeight;
        double countWeight = 0.0;
        for (int i = 0; i < range; i++) {
            countWeight += probabilities[i];
            if (countWeight >= r)
                return i;
        }
        throw new RuntimeException("Should never be shown.");
    }
    
	// debug utilities
	public static void print(double[][] matrix) {
		String string = "[ ";
		for (double[] r : matrix) {
			string += "[ ";
			for (double i : r) {
				string += i + " ";
			}
			string += "]";
		}
		string += "]";
		System.out.println(string);
	}
	
	public static void print(double[] vector) {
		String string = "[ ";
		for (double n : vector) { string += n + " "; }
		string += "]";
		System.out.println(string);
	}
}
