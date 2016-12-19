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
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SymbolTraining3 {

	protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static int height = 50;
    protected static int width = 50;
    protected static int channels = 3;
    protected static int outputNum = 4;
    protected static final long seed = 4878;
    protected static double rate = 0.0006;
    protected static int epochs = 5; //4000;

    public static final Random randNumGen = new Random(seed);
    private static Logger log = LoggerFactory.getLogger(SymbolTraining3.class);
    
	public SymbolTraining3() {
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
	        ImageRecordReader testReader = new ImageRecordReader(height,width,channels,labelMaker);

	        // 900, 71 - 45% A, 45.8% P, 45% R, 45.4% F1
	        // 900,500 - 40% A, 53.3% P, 40% R, 45.7% F1
	        
	        // seed values
	        // 4464
	        
	        int l1out = 900;
	        int outputIn = 500;

	        System.out.println(" --------- # of input the Layers: " + l1out + ", " + outputIn + " -----------");
	        //System.out.println(" --------- new seed: " + seed + " -----------");
	        
	        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	        .iterations(50)
	        .activation("relu")
	        .weightInit(WeightInit.XAVIER)
	        .learningRate(rate)
	        .updater(Updater.NESTEROVS).momentum(0.98)
	        .regularization(true).l2(1e-6)
	        .list()
	        .layer(0, new DenseLayer.Builder()
	        		.nIn(height * width * channels)
	                .nOut(l1out)
	                .weightInit(WeightInit.XAVIER)
	                .build())
		    .layer(1,  new DenseLayer.Builder()
            		.nIn(l1out)
            		.nOut(outputIn)
            		.weightInit(WeightInit.XAVIER)
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
	        System.out.println(" --- start training ---");
	        long start = System.currentTimeMillis();
	        model.init();
	        model.setListeners(new ScoreIterationListener(10));

	        recordReader.initialize(trainData);
	        for (int i=0; i < epochs; i++) {
		        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 40, 1, outputNum);
		        model.fit(dataIter);
		        // test phase
		        testReader.initialize(testData);
		        Evaluation eval = new Evaluation(outputNum);
		        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, 20, 1, outputNum);
		        while(testIter.hasNext()){
		            DataSet next = testIter.next();
		            INDArray output = model.output(next.getFeatureMatrix(), false);
		            eval.eval(next.getLabels(), output);
		        }
		        
		        System.out.println(eval.stats());
	        }
	        long end = System.currentTimeMillis();
	        System.out.println("**************** Run finished********************");
	        recordReader.reset();

	        long duration = end - start;
	        System.out.println(" training duration in MS: " + duration);
	        System.out.println(" training duration in Min: " + (duration/1000)/60);
	        System.exit(0);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
