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
    List<List<Integer>> data = new ArrayList<List<Integer>>();
    data.add(Arrays.asList(-2, -1));
    data.add(Arrays.asList(25, 6));
    data.add(Arrays.asList(17, 4));
    data.add(Arrays.asList(-15, -6));
    List<Double> answers = Arrays.asList(1.0,0.0,0.0,1.0);  
    network.train(data, answers);
  }

  class Network {
    double learnRate = .1;
    int epochs = 1000;
    Neuron nHidden1 = new Neuron();
    Neuron nHidden2 = new Neuron();
    Neuron nOutput = new Neuron();

    Random rand = new Random();
    Double w1 = 1.0; //rand.nextGaussian();
    Double w2 = 1.0; //rand.nextGaussian();
    Double w3 = 1.0; //rand.nextGaussian();
    Double w4 = 1.0; //rand.nextGaussian();
    Double w5 = 1.0; //rand.nextGaussian();
    Double w6 = 1.0; //rand.nextGaussian();
    Double b1 = 1.0; //rand.nextGaussian();
    Double b2 = 1.0; //rand.nextGaussian();
    Double b3 = 1.0; // rand.nextGaussian();

    public Double predict(Integer input1, Integer input2){
      return nOutput.compute(nHidden1.compute(input1, input2), nHidden2.compute(input1, input2));
    }
    public void train(List<List<Integer>> data, List<Double> answers){      
      //System.out.println("nHidden1: " + nHidden1);
      double learnRate = .1;
      for (int epoch = 0; epoch < epochs; epoch++){
        for (int i = 0; i < data.size(); i++){
	  double x0 = data.get(i).get(0); double x1 = data.get(i).get(1);
          double sumH1 = this.w1 * x0 + this.w2 * x1 + this.b1;
	  //System.out.println("!!! sumH!: " + sumH1 + " = " + this.w1 + " * " + x0 + " + " + this.w2 + " * " + x1 + " + "+  this.b1);
	  double h1 = Util.sigmoid(sumH1);
	  double sumH2 = this.w3 * x0 + this.w4 * x1 + this.b2;
	  double h2 = Util.sigmoid(sumH2);
	  double sumO1 = this.w5 * h1 + this.w6 * h2 + this.b3; 
	  double o1 = Util.sigmoid(sumO1);
          double yPred = o1;

	  double d_L_d_ypred = -2 * (answers.get(i) - yPred);
	  //if(epoch % 10 ==0) System.out.println("d_L_d_ypred = " + d_L_d_ypred + " answers.get("+i+") = " + answers.get(i) + " - " + yPred);

	  double d_ypred_d_w5 = h1 * Util.sigmoidDeriv(sumO1);
	  double d_ypred_d_w6 = h2 * Util.sigmoidDeriv(sumO1);
	  double d_ypred_d_b3 = Util.sigmoidDeriv(sumO1);

	  double d_ypred_d_h1 = this.w5 * Util.sigmoidDeriv(sumO1);
	  double d_ypred_d_h2 = this.w6 * Util.sigmoidDeriv(sumO1);

	  double d_h1_d_w1 = x0 * Util.sigmoidDeriv(sumH1);
	  double d_h1_d_w2 = x1 * Util.sigmoidDeriv(sumH1);
	  double d_h1_d_b1 = Util.sigmoidDeriv(sumH1);

	  double d_h2_d_w3 = x0 * Util.sigmoidDeriv(sumH2);
	  double d_h2_d_w4 = x1 * Util.sigmoidDeriv(sumH2);
	  double d_h2_d_b2 = Util.sigmoidDeriv(sumH2);

	  this.w1 -= learnRate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1;
	  this.w2 -= learnRate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2;
	  this.b1 -= learnRate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1;

	  this.w3 -= learnRate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3;
	  this.w4 -= learnRate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4;
	  this.b2 -= learnRate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2;

	  this.w5 -= learnRate * d_L_d_ypred * d_ypred_d_w5;
	  this.w6 -= learnRate * d_L_d_ypred * d_ypred_d_w6;
	  this.b3 -= learnRate * d_L_d_ypred * d_ypred_d_b3;

	  //System.out.println("this.b3 = " + this.b3 + " learnRate = " + learnRate + " d_L_d_ypred = " + d_L_d_ypred + " d_ypred_d_b3 = " + d_ypred_d_b3);
	}
        if (epoch % 10 == 0){
          //List<Double> predictions = data.stream().map( item -> this.predict(item.get(0), item.get(1)) ).collect( Collectors.toList() );
	  // feedforward
	  List<Double> predictions = new ArrayList<Double>(data.size());
	  for (int ff = 0; ff < data.size(); ff++){
	    double ffX0 = data.get(ff).get(0); double ffX1 = data.get(ff).get(1);
	    double ffH1 = Util.sigmoid(this.w1 * ffX0 + this.w2 * ffX1 + this.b1);
	    double ffH2 = Util.sigmoid(this.w3 * ffX0 + this.w4 * ffX1 + this.b2);
	    double ffO1 = Util.sigmoid(this.w5 * ffH1 + this.w6 * ffH2 + this.b3);
            predictions.add(ff, ffO1);
	  }
          Double loss = Util.meanSquareLoss(answers, predictions);
          System.out.println("     Epoch " + epoch + "    Loss: "+ loss + " | answers: " + answers + " | predictions: " + predictions);
        }
		/*
          //nHidden1.compute(data.get(i).get(0), data.get(i).get(1));
          //nHidden2.compute(data.get(i).get(0), data.get(i).get(1));
          //nOutput.compute(nHidden1.getActivatedOutput(), nHidden2.getActivatedOutput());	
	  //this.predict(data.get(i).get(0), data.get(i).get(1));
          if (epoch % 10 == 0){
	    //System.out.println("o: " + nOutput);
//  System.out.println("target: " + answers.get(i) + " | output: " + df.format(nOutput.compute(nHidden1.getActivatedOutput(), nHidden2.getActivatedOutput())) + " | nHidden1.w1: " + nHidden1.getWeight1());
	  }

	  double overallLoss = -2 * (answers.get(i) - this.predict(data.get(i).get(0), data.get(i).get(1))); // Derivative of the loss for this answer
//          if (epoch % 10 == 0) System.out.println("overallLoss: " + overallLoss + " = " + answers.get(i) +" - "+ nOutput.getActivatedOutput());

	  nHidden1.adjust(learnRate * overallLoss * (nOutput.getWeight1() * nOutput.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))) * (data.get(i).get(0) * nHidden1.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))),
			  learnRate * overallLoss * (nOutput.getWeight1() * nOutput.getDerivedOutput(data.get(i).get(0), data.get(i).get(1)  )) * (data.get(i).get(1) * nHidden1.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))),
			  learnRate * overallLoss * (nOutput.getWeight1() * nOutput.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))) * nHidden1.getDerivedOutput(data.get(i).get(0), data.get(i).get(1)));

	  if (epoch % 10 == 0) System.out.println("nHidden1.getDerivedOutput(): " + nHidden1.getDerivedOutput(data.get(i).get(0), data.get(i).get(1)));
         
	  nHidden2.adjust(learnRate * overallLoss * (nOutput.getWeight2() * nOutput.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))) * (data.get(i).get(0) * nHidden2.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))),
		          learnRate * overallLoss * (nOutput.getWeight2() * nOutput.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))) * (data.get(i).get(1) * nHidden2.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))),
			  learnRate * overallLoss * (nOutput.getWeight2() * nOutput.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))) * nHidden2.getDerivedOutput(data.get(i).get(0), data.get(i).get(1)));

	  nOutput.adjust(learnRate * overallLoss * (nHidden1.compute(data.get(i).get(0), data.get(i).get(1)) * nOutput.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))),
		         learnRate * overallLoss * (nHidden2.compute(data.get(i).get(0), data.get(i).get(1)) * nOutput.getDerivedOutput(data.get(i).get(0), data.get(i).get(1))),
			 learnRate * overallLoss * nOutput.getDerivedOutput(data.get(i).get(0), data.get(i).get(1)));
//	  if (epoch % 10 == 0) System.out.println("n1.w1 loss: " + learnRate * overallLoss * (nOutput.getWeight1() * nOutput.getDerivedOutput()) * (nHidden1.getInput1() * nHidden1.getDerivedOutput()));
	}
	
	if (epoch % 10 == 0){
  	  List<Double> predictions = data.stream().map( item -> this.predict(item.get(0), item.get(1)) ).collect( Collectors.toList() );
	  System.out.println("predictions: " + predictions);
          Double loss = Util.meanSquareLoss(answers, predictions);
          System.out.println("     Epoch " + epoch + "    Loss: "+ loss);
	}
	*/
      }
      System.out.println("sigmoid: " + Util.sigmoid(.25));
      System.out.println("deriv_sigmoid: " + Util.sigmoidDeriv(.25));
    }
  }

  class Neuron {
    Random random = new Random();
    private Double bias = random.nextGaussian(); private Double weight1 = random.nextGaussian(); private Double weight2 = random.nextGaussian();
    private Double preActivation = null; private Double output = null; private Double input1, input2;
    public double compute(double input1, double input2){
      return Util.sigmoid(this.getSum(input1, input2));
    }
    public String toString(){ return "w1: " + this.weight1 + " w2: " + this.weight2 + " b: " + this.bias + " = " + this.output; } 
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
