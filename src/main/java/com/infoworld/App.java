package com.infoworld;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;

import java.text.DecimalFormat;

import java.util.stream.Collectors;

public class App {
  private static final DecimalFormat df = new DecimalFormat("0.0000000000");
  public static void main( String[] args ) {
    App app = new App();
    app.train();
  }

  public void train () {
    Network network = new Network();
    List<List<Double>> data = new ArrayList<List<Double>>();
    data.add(Arrays.asList(-1.0, -5.5)); //-2.0, -1.0));
    data.add(Arrays.asList(-3.5, -2.0)); //25.0, 6.0));
    data.add(Arrays.asList(5.0, 6.5)); //17.0, 4.0));
    data.add(Arrays.asList(3.0, 1.5)); //-15.0, -6.0));
    List<Double> answers = Arrays.asList(.98,.95,0.01,0.2);  
    network.train(data, answers);
  }

  class Network {
    double learnRate = .15;
    int epochs = 1000;
    Neuron nHidden1 = new Neuron();
    Neuron nHidden2 = new Neuron();
    Neuron nOutput = new Neuron();

    public Double predict(Double input1, Double input2){
      return nOutput.compute(nHidden1.compute(input1, input2), nHidden2.compute(input1, input2));
    }
    public void train(List<List<Double>> data, List<Double> answers){          
      double learnRate = .1;
      for (int epoch = 0; epoch < epochs; epoch++){
        for (int i = 0; i < data.size(); i++){
	  double in1 = data.get(i).get(0); double in2 = data.get(i).get(1);
	  double loss = -2 * (answers.get(i) - this.predict(in1, in2)); // Derivative of the loss for this answer
          this.adjust(loss, in1, in2);
	}
	
	if (epoch % 10 == 0){
  	  List<Double> predictions = data.stream().map( item -> this.predict(item.get(0), item.get(1)) ).collect( Collectors.toList() );
          Double loss = Util.meanSquareLoss(answers, predictions);
          System.out.println("     Epoch " + epoch + "    pred: " + predictions + "     Loss: "+ loss);
	}
	
      }
    }
    public void adjust(Double loss, Double in1, Double in2){
      Double o1W1 = nOutput.getWeight1();  Double o1W2 = nOutput.getWeight2();
 
      Double h1Output = nHidden1.compute(in1, in2); 
      Double h2Output = nHidden2.compute(in1, in2);
      
      Double derivedOutput = nOutput.getDerivedOutput(h1Output, h2Output);
      
      Double derivedH1 = nHidden1.getDerivedOutput(in1, in2);
      Double derivedH2 = nHidden2.getDerivedOutput(in1, in2);

      nHidden1.adjust( 
	learnRate * loss * (o1W1 * derivedOutput) * (in1 * derivedH1),
        learnRate * loss * (o1W1 * derivedOutput) * (in2 * derivedH1),
        learnRate * loss * (o1W1 * derivedOutput) * derivedH1);
         
      nHidden2.adjust(
	learnRate * loss * (o1W2 * derivedOutput) * (in1 * derivedH2),
        learnRate * loss * (o1W2 * derivedOutput) * (in2 * derivedH2),
        learnRate * loss * (o1W2 * derivedOutput) * derivedH2);

      nOutput.adjust(
	learnRate * loss * h1Output * derivedOutput,
        learnRate * loss * h2Output * derivedOutput,
        learnRate * loss * derivedOutput);
    }
  }

  class Neuron {
    Random random = new Random();
    private Double bias = random.nextGaussian(); 
    private Double weight1 = random.nextGaussian(); 
    private Double weight2 = random.nextGaussian();
    private Double preActivation = null; private Double input1, input2;
    public double compute(double input1, double input2){
      return Util.sigmoid(this.getSum(input1, input2));
    }
    public String toString(){ return "w1: " + this.weight1 + " w2: " + this.weight2 + " b: " + this.bias; } 
    public Double getWeight1() { return this.weight1; }
    public Double getWeight2() { return this.weight2; }
    public Double getBias() { return this.bias; }
    public void setWeight1(Double w){ this.weight1 = w; }
    public void setWeight2(Double w){ this.weight2 = w; }
    public void setBias(Double b) { this.bias = b; }
    public Double getSum(double input1, double input2){ return (this.weight1 * input1) + (this.weight2 * input2) + this.bias; }
    public Double getDerivedOutput(double input1, double input2){ return Util.sigmoidDeriv(this.getSum(input1, input2)); }
    public void adjust(Double w1, Double w2, Double b){
      this.weight1 -= w1; this.weight2 -= w2; this.bias -= b;
    }
  }

  class Util {
    public static double sigmoid(double in){
      return 1 / (1 + Math.exp(-in));
    }
    public static double sigmoidDeriv(double in){
      double sigmoid = Util.sigmoid(in);
      return sigmoid * (1 - sigmoid);
    }
    /** Assumes array args are same length */
    public static Double meanSquareLoss(List<Double> correctAnswers, List<Double> predictedAnswers){
      double sumSquare = 0;
      for (int i = 0; i < correctAnswers.size(); i++){
        double error = correctAnswers.get(i) - predictedAnswers.get(i);
	sumSquare += (error * error);
      }
      return sumSquare / (correctAnswers.size());
    }
  }
}
