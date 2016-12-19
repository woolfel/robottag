package org.woolfel.robottag;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
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
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class SymbolTrainingIterations {

	protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static int height = 50;
    protected static int width = 50;
    protected static int channels = 3;
    protected static int outputNum = 4;
    protected static double rate = 0.0006;
    protected static int epochs = 4; //4000;

    public static final Random randNumGen = new Random();
    private static Logger log = Logger.getLogger(SymbolTrainingIterations.class);
    private static List<Integer> goodSeeds = new ArrayList<Integer>();
    
	public SymbolTrainingIterations() {
	}

	public static void main(String[] args) {
		try {
			File parentDir = new File("./data/symbols3");
			
			ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
	        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
	        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

	        //Split the image files into train and test.
	        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 50, 50);
	        InputSplit trainData = filesInDirSplit[0];
	        InputSplit testData = filesInDirSplit[1];
	        
	        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        
	        int l1out = 900;
	        int outputIn = 500;

	        System.out.println(" --------- # of input for Output Layer: " + outputIn + " -----------");

	        for (int i=10; i < 100; i++) {
	        	//System.out.println(" ------ seed: " + s + " ---------- ");
		        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
		        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		        .iterations(i)
		        .activation("relu")
		        .weightInit(WeightInit.XAVIER)
		        .learningRate(rate)
		        .updater(Updater.NESTEROVS).momentum(0.98)
		        .regularization(true).l2(1e-4)
		        .list()
		        .layer(0, new DenseLayer.Builder()
		        		.nIn(height * width * channels)
		                .nOut(l1out)
		                .weightInit(WeightInit.XAVIER)
		                .activation("relu")
		                .build())
			    .layer(1,  new DenseLayer.Builder()
	            		.nIn(l1out)
	            		.nOut(outputIn)
	            		.weightInit(WeightInit.XAVIER)
	            		.activation("relu")
	            		.build())
			    .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
			            .activation("softmax")
			            .nIn(outputIn)
			            .nOut(outputNum)
			            .build())
		        .pretrain(false)
		        .setInputType(InputType.convolutional(height,width,channels))
		        .backprop(true)
		        .build();

		        MultiLayerNetwork model = new MultiLayerNetwork(conf);
		        //System.out.println(" --- start training ---");
		        long start = System.currentTimeMillis();
		        model.init();
		        model.setListeners(new ScoreIterationListener(1));

		        recordReader.initialize(trainData);
		        for (int e=0; e < epochs; e++) {
			        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 1, 1, outputNum);
			        model.fit(dataIter);
		        }
		        long end = System.currentTimeMillis();
		        //System.out.println(" --- end training ---");
		        
		        //System.out.println(" --- start TEST ---");
		        // test phase
		        recordReader.reset();
		        recordReader.initialize(testData);
		        Evaluation eval = new Evaluation(outputNum);
		        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, 1, 1, outputNum);
		        while(testIter.hasNext()){
		            DataSet next = testIter.next();
		            INDArray output = model.output(next.getFeatureMatrix(), true);
		            eval.eval(next.getLabels(), output);
		        }
		        //System.out.println(" --- end TEST ---");
		        
		        //if (eval.accuracy() > 0.5 || eval.precision() > 0.5) {
		        	log.info(eval.stats());
		        	log.info(" ------------------ iterations: " + i);
		        	//goodSeeds.add(s);
			        System.out.println("****************Example finished********************");
			        System.out.println(eval.stats());
			        long duration = end - start;
			        System.out.println(" ------------------ iterations: " + i);
			        System.out.println(" training duration in MS: " + duration);
			        System.out.println(" training duration in Min: " + (duration/1000)/60);
		        //}
	        }

	        System.exit(0);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
