package org.woolfel.robottag;

import java.io.File;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RobotRecognizerTraining {

	protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static int height = 240;
    protected static int width = 320;
    protected static int channels = 3;
    protected static int outputNum = 4;
    protected static final long seed = 1234; 
    protected static double rate = 0.0006;
    protected static int epochs = 4000;

    public static final Random randNumGen = new Random(seed);
    private static Logger log = LoggerFactory.getLogger(RobotRecognizerTraining.class);
    
	public RobotRecognizerTraining() {
	}

	public static void main(String[] args) {
		try {
			File parentDir = new File("./data/robot_clearcase");
			
			ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
	        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
	        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

	        int oc = 62;
	        // 1000, 68 - 48% P
	        // 1000, 61 - 29% P, 30% A
	        // 1000, 62 - 35% P, 35% A
	        // 1000, 65 - 63% P, 28% A

        	System.out.println(" ---------- The input to OutputLayer: " + oc + " ------------");
	        //Split the image files into train and test.
	        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 10, 90);
	        InputSplit trainData = filesInDirSplit[0];
	        InputSplit testData = filesInDirSplit[1];
	        
	        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

	        recordReader.initialize(trainData);

	        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .seed(seed)
	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	        .iterations(1)
	        .activation("relu")
	        .weightInit(WeightInit.XAVIER)
	        .learningRate(rate)
	        .updater(Updater.NESTEROVS).momentum(0.92)
	        .regularization(true).l2(1e-6)
	        .list()
	        .layer(0, new DenseLayer.Builder()
	        		.nIn(height * width * channels)
	                .nOut(1000)
	                .weightInit(WeightInit.XAVIER)
	                .build())
	        .layer(1,  new DenseLayer.Builder()
	                .nIn(1000)
	                .nOut(oc)
	                .weightInit(WeightInit.XAVIER)
	                .build())
	        .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
	                .activation("softmax")
	                .nIn(oc)
	                .nOut(outputNum)
	                .build())
	        .pretrain(true)
	        .setInputType(InputType.convolutional(height,width,channels))
	        .backprop(true)
	        .build();

	        MultiLayerNetwork model = new MultiLayerNetwork(conf);
	        System.out.println(" --- start training ---");
	        long start = System.currentTimeMillis();
	        model.init();
	        model.setListeners(new ScoreIterationListener(5));

	        for (int i=0; i < epochs; i++) {
		        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 20, 1, outputNum);
		        while( dataIter.hasNext()) {
		        	DataSet nxt = dataIter.next();
		        	model.fit(nxt.getFeatureMatrix());
		        }
	        }
	        long end = System.currentTimeMillis();
	        System.out.println(" --- end training ---");
	        
	        System.out.println(" --- start TEST ---");
	        // test phase
	        recordReader.reset();
	        recordReader.initialize(testData);
	        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, 20, 1, outputNum);
	        Evaluation eval = new Evaluation(outputNum);
	        while(testIter.hasNext()){
	            DataSet next = testIter.next();
	            INDArray output = model.output(next.getFeatureMatrix(), false);
	            eval.eval(next.getLabels(), output);
	        }
	        System.out.println(eval.stats());
	        System.out.println("****************Example finished********************");
	        long duration = end - start;
	        System.out.println(" training duration in MS: " + duration);
	        System.out.println(" training duration in Min: " + (duration/1000)/60);

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
