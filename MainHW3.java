package HomeWork3;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;

public class MainHW3 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void createFoldTrainingData(Instances instances, int firstIndex, int numOfInstancesToRemove){
		for (int i = 0; i < numOfInstancesToRemove ; i++) {
			// when the firstIndex is removed, the others "shift" left
			instances.delete(firstIndex);
		}
	}

	public static void main(String[] args) throws Exception {
		String[] weightingScheme = {"uniform", "weighted"};
		FeatureScaler scaler = new FeatureScaler();
		Instances unScaled = loadData("src/auto_price.txt");
		// shuffling the data
		Random random = new Random();
		unScaled.randomize(random);
		// scaling the data
		Instances scaled = scaler.scaleData(unScaled);

		double scaledMinError = Double.MAX_VALUE;
		double unScaledMinError = Double.MAX_VALUE;
		double scaledCrossValidationError;
		double unscaledCrossValidationError;
		// arrays to save the p,k and weightingScheme values
		int[] scaledCombination = new int[3];
		int[] unScaledCombination = new int[3];
		// part 3
		int[] numOfFolds = {scaled.numInstances(), 50, 10, 5, 3};
		double regularCrossValidation;
		double efficientCrossValidation;
		String unscaledLpString;
		String scaledLpString;


		// for every k value
		for (int k = 1; k < 21; k++) {
			// if j = 0 lpvalue is infinity
			for (int p = 0; p < 4; p++) {
				// if l = 0 -> uniform if l=1 -> weighted
				for (int w = 0; w < 2; w++) {
					// unScaled
					Knn unScaledKnn = new Knn();
					// scaled
					Knn scaledKnn = new Knn();
					// uniform
					if (w == 0) {
						unScaledKnn.setCombination(p, k, true);
						scaledKnn.setCombination(p, k, true);
					} else {
						// weighted
						unScaledKnn.setCombination(p, k, false);
						scaledKnn.setCombination(p, k, false);
					}
					// calculate the cross validation error for the scaled data
					scaledCrossValidationError = scaledKnn.crossValidationError(scaled,10);
					// calculate the cross validation error for the unscaled data
					unscaledCrossValidationError = unScaledKnn.crossValidationError(unScaled,10);

					// current cross validation error is smaller than the minimum found so far
					if (scaledCrossValidationError < scaledMinError) {
						// current error is min error so far
						scaledMinError = scaledCrossValidationError;
						// updating to the current combination
						scaledCombination[0] = k;
						scaledCombination[1] = p;
						scaledCombination[2] = w;
					}
					// current cross validation error is smaller than the minimum found so far
					if (unscaledCrossValidationError < unScaledMinError) {
						// current error is min error so far
						unScaledMinError = unscaledCrossValidationError;
						// updating to the current combination
						unScaledCombination[0] = k;
						unScaledCombination[1] = p;
						unScaledCombination[2] = w;
					}
				}
			}
		}

		unscaledLpString = (unScaledCombination[1] == 0) ? "infinity" : ("" + unScaledCombination[1]);
		System.out.println("-------------------------------------");
		System.out.println("Results for original dataset: ");
		System.out.println("-------------------------------------");
		System.out.println("Cross validation error with K = " + unScaledCombination[0]
		+ ", lp = " + unscaledLpString + ", majority function = " + weightingScheme[unScaledCombination[2]] + " for auto_price data is: " + unScaledMinError);

		scaledLpString = (scaledCombination[1] == 0) ? "infinity" : ("" + scaledCombination[1]);
		System.out.println("-------------------------------------");
		System.out.println("Results for scaled dataset: ");
		System.out.println("-------------------------------------");
		System.out.println("Cross validation error with K = " + scaledCombination[0]
				+ ", lp = " + scaledLpString + ", majority function = " + weightingScheme[scaledCombination[2]] + " for auto_price data is: " + scaledMinError);

		for (int i = 0; i < numOfFolds.length; i++) {
			Knn efficientKnn = new Knn();
			Knn regularKnn = new Knn();

			if (scaledCombination[2] == 0) {
				// uniform
				efficientKnn.setCombination(scaledCombination[1], scaledCombination[0], true, Knn.DistanceCheck.Efficient);
				regularKnn.setCombination(scaledCombination[1], scaledCombination[0], true, Knn.DistanceCheck.Regular);
			} else {
				// weighted
				efficientKnn.setCombination(scaledCombination[1], scaledCombination[0], false, Knn.DistanceCheck.Efficient);
				regularKnn.setCombination(scaledCombination[1], scaledCombination[0], false, Knn.DistanceCheck.Regular);
			}
			// calculate the cross validation error for the scaled data
			long startReg = System.nanoTime();
			regularCrossValidation = regularKnn.crossValidationError(scaled, numOfFolds[i]);
			long endReg = System.nanoTime();
			// calculate the cross validation error for the scaled data
			long startEff = System.nanoTime();
			efficientCrossValidation = efficientKnn.crossValidationError(scaled, numOfFolds[i]);
			long endEff = System.nanoTime();

			System.out.println("-------------------------------------");
			System.out.println("Results for " + numOfFolds[i] + " number of folds: ");
			System.out.println("-------------------------------------");
			System.out.println("Cross validation error of regular knn on auto_price dataset is " + regularCrossValidation + " and the average elapsed time is " + ((endReg-startReg)/numOfFolds[i]));
			System.out.println("The total elapsed time is: " + (endReg-startReg));
			System.out.println();
			System.out.println("Cross validation error of efficient knn on auto_price dataset is " + efficientCrossValidation + " and the average elapsed time is " + ((endEff-startEff)/numOfFolds[i]));
			System.out.println("The total elapsed time is: " + (endEff-startEff));
		}
	}


}


