package com.mran.charnn;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;

import org.json.simple.JSONObject;

import com.google.gson.*;

public class ChaRNN {
	// Data
	private String data;
	private int vocabSize;
	private String inputPath;
	private int dataSize = 0;
	
	// Chars
	private ArrayList<String> chars;
	private HashMap<String, Integer> charToIndex;
	private HashMap<Integer, String> indexToChar;
	
	// Hyperparameters
	private int hiddenSize;
	private int seqLength;
	private double learningRate;
	private double global_reg = -0.001;
	
	// Weights & biases
	private double[][] Wxh;
	private double[][] Whh;
	private double[][] Why;
	private double[] bh;
	private double[] by;
	
	// Previous state
	private double[] hprev = new double[hiddenSize];
	
	// Memory variables for Adagrad
	private double[][] mWxh;
	private double[][] mWhh;
	private double[][] mWhy;
	private double[] mbh;
	private double[] mby;

	public ChaRNN(String inPath, int hSize, int sLength, double lRate) {
		this.inputPath = inPath;
		this.hiddenSize = hSize;
		this.seqLength = sLength;
		this.learningRate = lRate;
		
		try {
			prepareInput();
			
			double wxh[][] = Matrix.add(Matrix.random(hiddenSize, vocabSize), -0.5);
			Wxh = Matrix.multiply(wxh, 0.01);
			
			double[][] whh = Matrix.add(Matrix.random(hiddenSize, hiddenSize), -0.5);
			Whh = Matrix.multiply(whh, 0.01);
			
			double[][] why = Matrix.add(Matrix.random(vocabSize, hiddenSize), -0.5);
			Why = Matrix.multiply(why, 0.01);
			
			bh = new double[hiddenSize];
			by = new double[vocabSize];
			
			mWxh = new double[Wxh.length][Wxh[0].length];
			mWhh = new double[Whh.length][Whh[0].length];
			mWhy = new double[Why.length][Why[0].length];
			mbh = new double[bh.length];
			mby = new double[by.length];
			
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}	
	}
	
	public ChaRNN(String path) {
		JsonParser parser = new JsonParser();

		try {
            		Gson gson = new Gson();
            		JsonElement element = parser.parse(new FileReader(path));
            		Map<String, Object> obj = gson.fromJson(element, Map.class);
            		inputPath = obj.get("inputPath").toString();
           		vocabSize = (int) Double.parseDouble(obj.get("vocabSize").toString());
            		hiddenSize = (int) Double.parseDouble(obj.get("hiddenSize").toString());
            		seqLength = (int) Double.parseDouble(obj.get("seqLength").toString());
            		learningRate = Double.parseDouble(obj.get("learningRate").toString());
            
            		chars = gson.fromJson(obj.get("chars").toString(), ArrayList.class);
            
            		Wxh = gson.fromJson(obj.get("Wxh").toString(), double[][].class);
            		Whh = gson.fromJson(obj.get("Whh").toString(), double[][].class);
            		Why = gson.fromJson(obj.get("Why").toString(), double[][].class);
            		bh = gson.fromJson(obj.get("bh").toString(), double[].class);
            		by = gson.fromJson(obj.get("by").toString(), double[].class);
            
            		hprev = gson.fromJson(obj.get("hprev").toString(), double[].class);
            
            		prepareInput();
            
            		mWxh = new double[Wxh.length][Wxh[0].length];
			mWhh = new double[Whh.length][Whh[0].length];
			mWhy = new double[Why.length][Why[0].length];
			mbh = new double[bh.length];
			mby = new double[by.length];
        	} catch (IOException e) {
            		e.printStackTrace();
        	}
	}
	
	public void prepareInput() throws IOException {
		try {
			data = readFile(inputPath, StandardCharsets.UTF_8);
		} catch (IOException e) { throw e; }

		chars = listFromSetFromString(data);
		dataSize = data.length();
		vocabSize = chars.size();
		
		charToIndex = new HashMap<String, Integer>();
		indexToChar = new HashMap<Integer, String>();
		ListIterator<String> iterator = chars.listIterator();
		while (iterator.hasNext()) {
			int index = iterator.nextIndex();
			String next = iterator.next();
			charToIndex.put(next, index);
			indexToChar.put(index, next);
		}
	}
	
	public void prepareChars(ArrayList<String> chars) {
		charToIndex = new HashMap<String, Integer>();
		indexToChar = new HashMap<Integer, String>();
		ListIterator<String> iterator = chars.listIterator();
		while (iterator.hasNext()) {
			int index = iterator.nextIndex();
			String next = iterator.next();
			charToIndex.put(next, index);
			indexToChar.put(index, next);
		}
	}
	
	public Loss loss(int[] inputs, int[] targets, double[] hprev) {
		HashMap<Integer, Double[]> xs = new HashMap<Integer, Double[]>();
		HashMap<Integer, Double[]> hs = new HashMap<Integer, Double[]>();
		HashMap<Integer, Double[]> ys = new HashMap<Integer, Double[]>();
		HashMap<Integer, Double[]> ps = new HashMap<Integer, Double[]>();
		hs.put(new Integer(-1), toObject(hprev));
		double loss = 0;

		// Forward pass
		for (int i = 0; i < inputs.length; i++) {
			// One-hot encoding for xs
			double[] encoding = new double[vocabSize];
			encoding[inputs[i]] = 1;
			xs.put(new Integer(i), toObject(encoding));
			
			// Hidden state
			double[] Wxh_dot_xs = Matrix.multiply(Wxh, toRaw(xs.get(i)));
			double[] Whh_dot_hs = Matrix.multiply(Whh, toRaw(hs.get(new Integer(i-1))));
			
			double[] state = Matrix.add(Matrix.add(Wxh_dot_xs, Whh_dot_hs), bh);
			
			state = Matrix.tanh(state);
			
			hs.put(i, toObject(state));
			
			// Unnormalized log probabilities for next characters
			double[] Why_dot_hs = Matrix.multiply(Why, toRaw(hs.get(i)));
			ys.put(i, toObject(Matrix.add(Why_dot_hs, by)));
			
			// Probabilities for next characters (logits)
			double[] probabilities = Matrix.softmax(toRaw(ys.get(i)));
			ps.put(i, toObject(probabilities));
			
			// Cross-entropy loss
			loss += -1 * Math.log(toRaw(ps.get(i))[targets[i]]);
		}
		
		// Backward pass
		double[][] dWxh = new double[Wxh.length][Wxh[0].length];
		double[][] dWhh = new double[Whh.length][Whh[0].length];
		double[][] dWhy = new double[Why.length][Why[0].length];
		double[] dbh = new double[bh.length];
		double[] dby = new double[by.length];
		double[] dhnext = new double[hs.get(0).length];

		for (int n = inputs.length-1; n >= 0; n--) {
			double[] dy = toRaw(ps.get(n).clone());
			dy[targets[n]] -= 1; // basically one-hot encoding minus the scores/probability, so delta y
			dWhy = Matrix.add(dWhy, Matrix.multiply(dy, toRaw(hs.get(n))));
			dby = Matrix.add(dby, dy);
			
			double[] dh = Matrix.multiply(Matrix.transpose(Why), dy);
			dh = Matrix.add(dh, dhnext); // backprop into h
			double[] state = toRaw(hs.get(n));
			double[] statesq = Matrix.elementMult(state, state);
			double[] dhraw = Matrix.elementMult(Matrix.subtract(1.0, statesq), dh);
			dbh = Matrix.add(dbh, dhraw);
			
			double[][] dhraw_dot_xsT = Matrix.multiply(dhraw, toRaw(xs.get(n)));
			dWxh = Matrix.add(dWxh, dhraw_dot_xsT);
			
			double[][] dhraw_dot_hsT = Matrix.multiply(dhraw, toRaw(hs.get(n-1)));
			dWhh = Matrix.add(dWhh, dhraw_dot_hsT);
			
			double[] WhhT_dot_dhraw = Matrix.multiply(Matrix.transpose(Whh), dhraw);
			dhnext = WhhT_dot_dhraw;

		}
		
		dWxh = Matrix.clip(dWxh, -5, 5);
		dWhh = Matrix.clip(dWhh, -5, 5);
		dWhy = Matrix.clip(dWhy, -5, 5);
		dbh = Matrix.clip(dbh, -5, 5);
		dby = Matrix.clip(dby, -5, 5);
		
		return new Loss(loss, dWxh, dWhh, dWhy, dbh, dby, toRaw(hs.get(inputs.length-1)));
	}
	
	public void train(int maxstep) {
		int n = 0; int p = 0;
		
		// Loss at iteration 0
		double smoothLoss = -1 * Math.log(1.0/vocabSize) * seqLength;
		
		while (true) { 
			
			if ((p + seqLength + 1) >= data.length() || n == 0) {
				hprev = new double[hiddenSize]; // reset RNN memory...
				p = 0; // go from start of data
			}
			
			int[] inputs = new int[seqLength];
			for (int a = 0; a < seqLength; a++) {
				String ch = Character.toString(data.charAt(p+a));
				inputs[a] = charToIndex.get(ch);		
			}
			
			int[] targets = new int[seqLength];
			for (int b = 0; b < seqLength; b++) {
				String ch = Character.toString(data.charAt(p+b+1));
				targets[b] = charToIndex.get(ch);
			}
			
			// Sample from the model now and then
			if (n % 100 == 0) {
				int[] sampleIndices = sample(hprev, inputs[0], 200);
				String text = indicesToChars(sampleIndices);
				System.out.println("----\n" + text + "\n----");
			}
			
			// Forward seqLength characters thru the net and fetch gradient
			Loss loss = loss(inputs, targets, hprev);
			hprev = loss.state;
						
			double reg_t = global_reg / Math.sqrt(n+100);
			
			// Perform parameter update with Adagrad
			mWxh = Matrix.add(mWxh, Matrix.elementMult(loss.dWxh, loss.dWxh));
			double[][] adagrad = Matrix.multiply(loss.dWxh, (-1.0*learningRate));
			adagrad = Matrix.divide(adagrad, Matrix.sqrt(Matrix.add(mWxh, 0.00000001)));			
			adagrad = Matrix.add(adagrad, Matrix.multiply(Wxh, reg_t));
			Wxh = Matrix.add(Wxh, adagrad);
			
			mWhh = Matrix.add(mWhh, Matrix.elementMult(loss.dWhh, loss.dWhh));
			adagrad = Matrix.multiply(loss.dWhh, (-1.0*learningRate));
			adagrad = Matrix.divide(adagrad, Matrix.sqrt(Matrix.add(mWhh, 0.00000001)));
			adagrad = Matrix.add(adagrad, Matrix.multiply(Whh, reg_t));
			Whh = Matrix.add(Whh, adagrad);
			
			mWhy = Matrix.add(mWhy, Matrix.elementMult(loss.dWhy, loss.dWhy));
			adagrad = Matrix.multiply(loss.dWhy, (-1.0*learningRate));
			adagrad = Matrix.divide(adagrad, Matrix.sqrt(Matrix.add(mWhy, 0.00000001)));
			adagrad = Matrix.add(adagrad, Matrix.multiply(Why, reg_t));
			Why = Matrix.add(Why, adagrad);
			
			mbh = Matrix.add(mbh, Matrix.elementMult(loss.dbh, loss.dbh));
			double[] adagrad2 = Matrix.multiply(loss.dbh, (-1.0*learningRate));
			adagrad2 = Matrix.divide(adagrad2, Matrix.sqrt(Matrix.add(mbh, 0.00000001)));
			adagrad2 = Matrix.add(adagrad2, Matrix.multiply(bh, reg_t));
			bh = Matrix.add(bh, adagrad2);
			
			mby = Matrix.add(mby, Matrix.elementMult(loss.dby, loss.dby));
			adagrad2 = Matrix.multiply(loss.dby, (-1.0*learningRate));
			//adagrad2 = Matrix.add(adagrad2, Matrix.multiply(by, global_reg));			
			adagrad2 = Matrix.divide(adagrad2, Matrix.sqrt(Matrix.add(mby, 0.00000001)));

			by = Matrix.add(by, adagrad2);
			
			smoothLoss = (smoothLoss * 0.999) + (loss.loss * 0.001);
			if (n % 100 == 0) { System.out.println("iter " + n + ", loss: " + smoothLoss); }

			p += seqLength;
			n++;
			
			if(n>maxstep) break;
		}
	}
	
	
	public int[] sample(double[] h, int seedIndex, int n) {
		double[] state = h.clone();
		double[] x = new double[vocabSize];
		x[seedIndex] = 1;
		int[] indices = new int[n];
		for (int t = 0; t < n; t++) {
			double[] Wxh_dot_x = Matrix.multiply(Wxh, x);
			double[] Whh_dot_h = Matrix.multiply(Whh, state);
			
			state = Matrix.add(Matrix.add(Wxh_dot_x, Whh_dot_h), bh);
			
			state = Matrix.tanh(state);
			
			double[] y = Matrix.add(Matrix.multiply(Why, state), by);
			double[] p = Matrix.softmax(y);
			
			int index = Matrix.randomChoice(vocabSize, p);
			x = new double[vocabSize];
			x[index] = 1;
			indices[t] = index;
		}
				
		return indices;
	}
	
	public int[] forward(double[] h, int[] seedIndexs, int n) {
		double[] state = h.clone();
		double[] x = new double[vocabSize];
		int[] indices = new int[n+seedIndexs.length];
		
		int seedIndex = seedIndexs[0];
		x[seedIndex] = 1;
		
		for (int t = 0; t < n+seedIndexs.length; t++) {
			double[] Wxh_dot_x = Matrix.multiply(Wxh, x);
			double[] Whh_dot_h = Matrix.multiply(Whh, state);
			
			state = Matrix.add(Matrix.add(Wxh_dot_x, Whh_dot_h), bh);
			state = Matrix.tanh(state);
			
			double[] y = Matrix.add(Matrix.multiply(Why, state), by);
			double[] p = Matrix.softmax(y);
			
			int index = Matrix.randomChoice(vocabSize, p);
			x = new double[vocabSize];
			
			if(t<seedIndexs.length) {
				indices[t] = seedIndexs[t];
				x[seedIndexs[t]] = 1;		
			} else {
				indices[t] = index;
				x[index] = 1;
			}
		}
				
		return indices;
	}
	
	public String forward(String seeds, int n) {
		String str = "";
		
		double[] h = new double[hiddenSize]; 
		
		int [] seedIndexs = stringToIndices(seeds);
		int[] result = forward(h, seedIndexs, n);
		str = indicesToChars(result);
		return str;
	}
	
	public void save() throws IOException {
		System.out.println("Saving checkpoint...");
		
		Gson gson = new Gson();
		JSONObject obj = new JSONObject();
		obj.put("inputPath", inputPath);
		obj.put("vocabSize", vocabSize);
		obj.put("hiddenSize", hiddenSize);
		obj.put("seqLength", seqLength);
		obj.put("learningRate", learningRate);
		obj.put("chars", gson.toJson(chars));
		obj.put("Wxh", gson.toJson(Wxh));
		obj.put("Whh", gson.toJson(Whh));
		obj.put("Why", gson.toJson(Why));
		obj.put("bh", gson.toJson(bh));
		obj.put("by", gson.toJson(by));
		obj.put("hprev", gson.toJson(by));
		
		System.out.println(Wxh[0][0]);
		
		String timestamp = new Date().toString();
		try (FileWriter writer = new FileWriter("checkpoint_" + timestamp + ".txt")) {
			writer.write(obj.toJSONString());
			System.out.println("Save complete.");
		} catch (IOException e) {
			throw e;
		}
	}
	
	//--- Helper functions ---//
	private String readFile(String path, Charset encoding) throws IOException {
		byte[] encoded = Files.readAllBytes(Paths.get(path));
		return new String(encoded, encoding);
	}
	
	private ArrayList<String> listFromSetFromString(String string) {
		Set<String> set = new HashSet<String>(Arrays.asList(string.split("")));
		set.remove("");
		ArrayList<String> list = new ArrayList<String>();
		list.addAll(set);
		return list;
	}
	
	private String indicesToChars(int[] indices) {
		String string = "";
		for (int index : indices) { string += indexToChar.get(index); }
		return string;
	}
	
	private int[] stringToIndices(String str) {
		int[] indices = new int[str.length()];
		for(int n = 0; n<str.length(); n++) {
			String c = Character.toString(str.charAt(n));
			Integer i = charToIndex.get(c);
			if(i!=null) indices[n] = i.valueOf(i);
		}
		return indices;
	}
	
	private Double[] toObject(double[] array) {
		Double[] result = new Double[array.length];
		for (int i = 0; i < array.length; i++) {
			result[i] = new Double(array[i]);
		}
		return result;
	}
	
	private double[] toRaw(Double[] array) {
		double[] result = new double[array.length];
		for (int i = 0; i < array.length; i++) {
			result[i] = array[i].doubleValue();
		}
		return result;
	}
	
	//--- Debug utilities ---//
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
