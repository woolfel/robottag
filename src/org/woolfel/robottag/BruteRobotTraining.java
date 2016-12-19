package org.woolfel.robottag;

import java.io.File;
import java.util.Random;

import org.apache.log4j.Logger;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class BruteRobotTraining {

	protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static int height = 240;
    protected static int width = 320;
    protected static int channels = 1;
    protected static int outputNum = 4;

    protected static double rate = 0.0006;
    protected static int epochs = 5;

    public static final Random randNumGen = new Random();
    private static Logger log = Logger.getLogger(BruteRobotTraining.class);
    
	public BruteRobotTraining() {
	}

	public static void main(String[] args) {
		try {
			File parentDir = new File("./data/robot_clearcase_bw");
			
			ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
	        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
	        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

	        //Split the image files into train and test.
	        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 50, 50);
	        InputSplit trainData = filesInDirSplit[0];
	        InputSplit testData = filesInDirSplit[1];
	        
	        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

	        recordReader.initialize(trainData);

	        
	        for (int l1=500; l1 < 1000; l1++) {
	        	// inner loop for second layer
	        	for (int l2=50; l2 < 200; l2++) {
	        		for (int s=4450; s < 4900; s++) {
		    	        
		    	        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		    	        .seed(s)
		    	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		    	        .iterations(1)
		    	        .activation("relu")
		    	        .weightInit(WeightInit.XAVIER)
		    	        .learningRate(rate)
		    	        .updater(Updater.NESTEROVS).momentum(0.96)
		    	        .regularization(true).l2(1e-6)
		    	        .list()
		    	        .layer(0, new DenseLayer.Builder()
		    	        		.nIn(height * width * channels)
		    	                .nOut(l1)
		    	                .weightInit(WeightInit.XAVIER)
		    	                .build())
		    	        .layer(1,  new DenseLayer.Builder()
		    	                .nIn(l1)
		    	                .nOut(l2)
		    	                .weightInit(WeightInit.XAVIER)
		    	                .build())
		    	        .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
		    	                .activation("softmax")
		    	                .nIn(l2)
		    	                .nOut(outputNum)
		    	                .build())
		    	        .pretrain(true)
		    	        .setInputType(InputType.convolutional(height,width,channels))
		    	        .backprop(true)
		    	        .build();

		    	        MultiLayerNetwork model = new MultiLayerNetwork(conf);
		    	        long start = System.currentTimeMillis();
		    	        model.init();
		    	        model.setListeners(new ScoreIterationListener(2));

		    	        for (int i=0; i < epochs; i++) {
		    		        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 20, 1, outputNum);
		    		        while( dataIter.hasNext()) {
		    		        	DataSet nxt = dataIter.next();
		    		        	model.fit(nxt.getFeatureMatrix());
		    		        }
		    	        }
		    	        long end = System.currentTimeMillis();

		    	        // test phase
		    	        recordReader.reset();
		    	        recordReader.initialize(testData);
		    	        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, 20, 1, outputNum);
		    	        Evaluation eval = new Evaluation(outputNum);
		    	        while(testIter.hasNext()){
		    	            DataSet next = testIter.next();
		    	            INDArray output = model.output(next.getFeatureMatrix(), true);
		    	            eval.eval(next.getLabels(), output);
		    	        }
				        if (eval.accuracy() > 0.5 || eval.precision() > 0.5) {
			    	        long duration = end - start;
			    	        System.out.println("****************Example finished********************");
			    	        log.info("****************Example finished********************");
				        	log.info(eval.stats());
				        	log.info(" ------------------ Seed: " + s);
				        	log.info(" ------------------ inputs: " + l1 + ", " + l2);
				        	log.info(" training duration in MS: " + duration);
			    	        System.out.println(eval.stats());
			    	        System.out.println(" training duration in MS: " + duration);
			    	        System.out.println(" training duration in Min: " + (duration/1000)/60);
				        }
	        		}
	        		java.lang.Runtime.getRuntime().gc();
	        	}
	        }
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
