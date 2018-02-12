public class Loss {
	public double loss;
	public double[][] dWxh, dWhh, dWhy;
	public double[] dbh, dby, state;
	
	public Loss(double loss, double[][] dWxh, double[][] dWhh, double[][] dWhy, double[] dbh, double[] dby, double[] state) {
		this.loss = loss;
		this.dWxh = dWxh;
		this.dWhh = dWhh;
		this.dWhy = dWhy;
		this.dbh = dbh;
		this.dby = dby;
		this.state = state;
	}
}
