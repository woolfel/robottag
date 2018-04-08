package org.woolfel.robottag;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SignSaved {

	protected static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
	protected static int height = 75;
	protected static int width = 100;
	protected static int channels = 1;
	protected static int outputNum = 2;
	protected static final long seed = 123;
	protected static double rate = 0.006;
	protected static int epochs = 10;
	public static final Random randNumGen = new Random();
	private static Logger log = LoggerFactory.getLogger(SignSaved.class);
	
	public static void main(String[] args) {
		
		File parentDir = new File("./data/signs3");
		String modelfile = "./data/sign-gm-C.model";
		System.out.println(modelfile);

		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
		FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

		// Split the image files into train and test.
		InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter);
		InputSplit testData = filesInDirSplit[0];
		ImageRecordReader testReader = new ImageRecordReader(height, width, channels, labelMaker);
		System.out.println("Number of records in Test: " + testData.length());
		try {
			testReader.initialize(testData);
			long loadStart = System.currentTimeMillis();
			MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(modelfile));
			long loadend = System.currentTimeMillis();
			System.out.println("Model loaded. It took " + (loadend- loadStart)/1000 + " secs");
			DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, 20, 1, outputNum);
			Evaluation eval = new Evaluation(outputNum);
			long start = System.nanoTime();
			while (testIter.hasNext()) {
				DataSet next = testIter.next();
				INDArray output = model.output(next.getFeatureMatrix(), false);
				eval.eval(next.getLabels(), output);
			}
			long end = System.nanoTime();
			System.out.println(eval.stats());
			log.info(eval.stats());
			long elnano = end - start;
			System.out.println("Nanoseconds total: " + elnano + " ns - " + elnano/1000000 + " ms");
			System.out.println("Nanoseconds per test: " + elnano/testData.length() + " ns - " + (elnano/testData.length())/1000000 + " ms");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
