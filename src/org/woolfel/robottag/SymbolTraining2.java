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

public class SymbolTraining2 {

	protected static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static int height = 240;
    protected static int width = 320;
    protected static int channels = 3;
    protected static int outputNum = 4;
    protected static final long seed = 1234; // seed of 1234 gets 30% accuracy
    protected static double rate = 0.0006;
    protected static int epochs = 5; //4000;

    public static final Random randNumGen = new Random(seed);
    private static Logger log = LoggerFactory.getLogger(SymbolTraining2.class);
    
	public SymbolTraining2() {
	}

	public static void main(String[] args) {
		try {
			File parentDir = new File("./data/symbols2");
			
			ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
	        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
	        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

	        //Split the image files into train and test.
	        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 50, 50);
	        InputSplit trainData = filesInDirSplit[0];
	        InputSplit testData = filesInDirSplit[1];
	        
	        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

	        // 447, 71 - 35% A, 44.0% P, 35% R
	        // 453, 71 - 35% A, 63.8% P, 35% R, 45.2% F1
	        // 753, 71 - 40% A, 53.3% P, 40% R
	        // 800, 71 - 45% A, 62.8% P, 45% R
	        
	        // layer 0 softsign activation
	        // 453, 71 - 40% A, 43.7% P, 40% R, 41.7% F1
	        
	        int l1out = 453;
	        int outputIn = 71;
	        System.out.println(" --------- # of input for Output Layer: " + outputIn + " -----------");
	        
	        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .seed(seed)
	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	        .iterations(1)
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
	                .activation("softsign")
	                .build())
		    .layer(1,  new DenseLayer.Builder()
            		.nIn(l1out)
            		.nOut(outputIn)
            		.weightInit(WeightInit.XAVIER)
            		.activation("relu")
            		.build())
		    .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
		            .activation("relu")
		            .nIn(outputIn)
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
	        model.setListeners(new ScoreIterationListener(1));

	        recordReader.initialize(trainData);
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
	        Evaluation eval = new Evaluation(outputNum);
	        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, 20, 1, outputNum);
	        while(testIter.hasNext()){
	            DataSet next = testIter.next();
	            INDArray output = model.output(next.getFeatureMatrix(), true);
	            eval.eval(next.getLabels(), output);
	        }
	        System.out.println(" --- end TEST ---");
	        
	        System.out.println(eval.stats());
	        System.out.println("****************Example finished********************");
	        long duration = end - start;
	        System.out.println(" training duration in MS: " + duration);
	        System.out.println(" training duration in Min: " + (duration/1000)/60);
	        System.exit(0);
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
}
