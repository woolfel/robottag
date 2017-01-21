package org.woolfel.robottag;

import java.io.File;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Text2Training {

	protected static final String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
	protected static int height = 20;
	protected static int width = 20;
	protected static int channels = 1;
	protected static int outputNum = 2;
	protected static final long seed = 123;
	protected static double rate = 0.006;
	protected static int epochs = 5;

	public static final Random randNumGen = new Random();
	private static Logger log = LoggerFactory.getLogger(Text2Training.class);

	public Text2Training() {
	}

	public static void main(String[] args) {
		try {
			File parentDir = new File("./data/text2");

			ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
			BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
			FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

			// Split the image files into train and test.
			InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 50, 50);
			InputSplit trainData = filesInDirSplit[0];
			InputSplit testData = filesInDirSplit[1];
			System.out.println("training set: " + trainData.length());
			System.out.println("testomg set: " + testData.length());

			ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
			ImageRecordReader testReader = new ImageRecordReader(height, width, channels, labelMaker);

			recordReader.initialize(trainData);
			testReader.initialize(testData);

			boolean stop = false;
			for (int i=0; i < 5000; i++) {
				MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()

						.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
						.iterations(1)
						.activation("relu")
						.weightInit(WeightInit.XAVIER)
						.learningRate(rate)
						.updater(Updater.NESTEROVS)
						.momentum(0.9)
						.regularization(true).l2(1e-4).list()
						.layer(0,
								new DenseLayer.Builder()
								.nIn(height * width)
								.nOut(8000)
								.activation("relu")
								.weightInit(WeightInit.XAVIER)
								.build())
						.layer(1,
								new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
								.nIn(8000)
								.nOut(outputNum)
								.activation("softmax")
								.weightInit(WeightInit.XAVIER)
								.build())
						.pretrain(false)
						.setInputType(InputType.convolutional(height, width, channels))
						.backprop(true)
						.build();

				MultiLayerNetwork model = new MultiLayerNetwork(conf);
				System.out.println(" --- start training ---");
				long start = System.currentTimeMillis();
				model.init();
				model.setListeners(new ScoreIterationListener(1));

				for (int e = 0; e < epochs; e++) {
					DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 20, 1, outputNum);
					model.fit(dataIter);
					System.out.println("iterations: " + i + " / epoch " + e + " done");

					DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, 20, 1, outputNum);
					Evaluation eval = new Evaluation(outputNum);
					while (testIter.hasNext()) {
						DataSet next = testIter.next();
						INDArray output = model.output(next.getFeatureMatrix(), false);
						eval.eval(next.getLabels(), output);
					}
					System.out.println(eval.stats());
					log.info(eval.stats());
					if (eval.accuracy() > 0.98) {
						 ModelSerializer.writeModel(model, new File("./data/text2-gm-" + i + ".model"), true);
						 //stop = true;
						 break;
					}
					recordReader.reset();
					testReader.reset();
				}
				long end = System.currentTimeMillis();
				System.out.println(" --- end training ---");

				System.out.println("****************Example finished********************");
				long duration = end - start;
				System.out.println(" training duration in MS: " + duration);
				System.out.println(" training duration in Min: " + (duration / 1000) / 60);
				System.out.println(" training duration in hours: " + (duration / 1000) / 3600);
				if (stop) {
					System.out.println("iterations to get good model: " + i);
					break;
				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
