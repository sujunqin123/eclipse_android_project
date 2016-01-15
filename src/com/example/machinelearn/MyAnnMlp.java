package com.example.machinelearn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.TermCriteria;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Ml;

import android.R.integer;

import com.example.dataprocess.TextDataRead;
import com.example.utils.LogUtil;

public class MyAnnMlp {
	private static final int ROW = TextDataRead.ROW;
	private static final int COL = TextDataRead.COL;
	private static final int COUNT_FEATURE = COL - 1;

	private ANN_MLP ann_MLP;
	private int numCharacter = 10;

	private Mat dataSetMat;
	private int[] classSet = new int[ROW];
	private Mat train_set, sample_set;
	private List<Integer> train_labels, sample_labels;

	public MyAnnMlp(float[][] dataset, ANN_MLP ann_MLP) {

		this.ann_MLP = ann_MLP;
		dataSetMat = new Mat(ROW, COUNT_FEATURE, CvType.CV_32FC1);
		train_set = new Mat();
		sample_set = new Mat();
		train_labels = new ArrayList<Integer>();
		sample_labels = new ArrayList<Integer>();

		for (int i = 0; i < dataSetMat.rows(); i++) {
			for (int j = 0; j < COL; j++) {
				if (j < COUNT_FEATURE) {
					dataSetMat.put(i, j, dataset[i][j]);
				} else {
					classSet[i] = (int) dataset[i][j];
				}

			}
		}

		int markSample[] = new int[ROW];
		generateRandom(0, 500, 0, ROW - 1, markSample);

		for (int i = 0; i < dataSetMat.rows(); i++) {
			if (markSample[i] == 1) {
				sample_set.push_back(dataSetMat.row(i));
				sample_labels.add(classSet[i]);
			} else {
				train_set.push_back(dataSetMat.row(i));
				train_labels.add(classSet[i]);
			}
		}

	}

	void generateRandom(int n, int test_num, int min, int max, int[] mark_samples) {
		Random random = new Random();
		int range = max - min;
		int index = random.nextInt(range) % range + min;
		if (mark_samples[index] == 0) {
			mark_samples[index] = 1;
			n++;
		}

		if (n < test_num)
			generateRandom(n, test_num, min, max, mark_samples);
	}

	public void annTrain(Mat train_set, List<Integer> train_labels, int nNeruns) {
		LogUtil.info("annTrain->step1");
		// Prepare trainClases
		// Create a mat with n trained data by m classes

		Mat trainClasses = new Mat(train_set.rows(), numCharacter, CvType.CV_32FC1);
		for (int i = 0; i < trainClasses.rows(); i++) {
			for (int k = 0; k < trainClasses.cols(); k++) {
				// If class of data i is same than a k class
				if (k == train_labels.get(i))
					trainClasses.put(i, k, 1);
				else
					trainClasses.put(i, k, 0);
			}

		}

		LogUtil.info("annTrain->step2");
		// setting the NN layer size
		int ar[] = { train_set.cols(), nNeruns, numCharacter };
		Mat layerSizes = new Mat(1, 3, CvType.CV_32SC1);

		for (int i = 0; i < ar.length; i++) {
			layerSizes.put(0, i, ar[i]);
		}

		ann_MLP.setLayerSizes(layerSizes);
		ann_MLP.setActivationFunction(ANN_MLP.SIGMOID_SYM);
		ann_MLP.setTrainMethod(ANN_MLP.BACKPROP);
		ann_MLP.setBackpropMomentumScale(0.3);
		ann_MLP.setBackpropWeightScale(0.3);
		ann_MLP.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, (int) 5000, 0.01));
		// ann_MLP.setRpropDW0(0.1);
		// ann_MLP.setRpropDWPlus(1.2);
		// ann_MLP.setRpropDWMinus(0.5);
		// ann_MLP.setRpropDWMin(FLT_EPSILON);
		// ann_MLP.setRpropDWMax(50.0);
		ann_MLP.train(train_set, Ml.ROW_SAMPLE, trainClasses);
	}

	public float ann_test(Mat testSet, List<Integer> testLabels) {
		int correctNum = 0;
		float accurate = 0;

		// Prepare trainClases
		// Create a mat with n trained data by m classes
		Mat testClasses = new Mat(testSet.rows(), numCharacter, CvType.CV_32FC1);
		for (int i = 0; i < testClasses.rows(); i++) {
			for (int k = 0; k < testClasses.cols(); k++) {
				// If class of data i is same than a k class
				if (k == testLabels.get(i))
					testClasses.put(i, k, 1);
				else
					testClasses.put(i, k, 0);
			}

		}

		for (int i = 0; i < testSet.rows(); i++) {
			int result = ann_classify(testSet.row(i));
			if (result == testLabels.get(i))
				correctNum++;
		}
		accurate = (float) correctNum / testSet.rows();
		return accurate;
	}

	public String testOneSample() {
		int classLabel[] = new int[1];
		int result = -1;
		Mat features = new Mat();
		features = getOneSample(classLabel);
		result = ann_classify(features);
		return "classLabel:" + classLabel[0] + ",result:" + result;
	}

	public Mat getOneSample(int[] reuslt) {
		Mat sample = new Mat();
		Random random = new Random();
		int range = dataSetMat.rows() - 0;
		int index = random.nextInt(range) % range + 0;
		reuslt[0] = classSet[index];
		sample.push_back(dataSetMat.row(index));
		return sample;
	}

	public int ann_classify(Mat features) {
		int result = -1;
		float max = 0;
		Mat predictResult = new Mat(1, numCharacter, CvType.CV_32FC1);
		ann_MLP.predict(features, predictResult, 0);

		for (int i = 0; i < predictResult.cols(); i++) {
			float data[] = new float[1];
			predictResult.get(0, i, data);
			if (max < data[0]) {
				max = data[0];
				result = i;
			}
		}
		return result;
	}

	public Mat getTrain_set() {
		return train_set;
	}

	public void setTrain_set(Mat train_set) {
		this.train_set = train_set;
	}

	public Mat getSample_set() {
		return sample_set;
	}

	public void setSample_set(Mat sample_set) {
		this.sample_set = sample_set;
	}

	public List<Integer> getTrain_labels() {
		return train_labels;
	}

	public void setTrain_labels(List<Integer> train_labels) {
		this.train_labels = train_labels;
	}

	public List<Integer> getSample_labels() {
		return sample_labels;
	}

	public void setSample_labels(List<Integer> sample_labels) {
		this.sample_labels = sample_labels;
	}

}
