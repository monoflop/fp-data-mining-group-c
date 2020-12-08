package com.fp_data_mining_group_c.datasetfix;

import com.fp_data_mining_group_c.datasetfix.scheme.DataTestConverter;
import com.fp_data_mining_group_c.datasetfix.scheme.DataTestRow;
import com.fp_data_mining_group_c.datasetfix.scheme.DataTrainConverter;
import com.fp_data_mining_group_c.datasetfix.scheme.DataTrainRow;

import java.io.File;
import java.io.IOException;

public class DataSetFix {

    public static void main(String[] args) throws IOException {
        System.out.println("-----------------------------------------");
        System.out.println("TU-Dortmund fp-data-mining-group-c 2020");
        System.out.println("DataSetFix Util");
        System.out.println("-----------------------------------------");

        if(args.length != 1) {
            System.out.println("Usage:");
            System.out.println("DataSetFix [DATA_DIR]");
            return;
        }

        //Create paths and check if paths are valid
        File dataRootDirectory = new File(args[0]);
        File dataTrainDirectory = new File(dataRootDirectory, "train");
        File dataTestDirectory = new File(dataRootDirectory, "test");
        if(!dataTrainDirectory.exists() || !dataTrainDirectory.isDirectory()
                || !dataTestDirectory.exists() || !dataTestDirectory.isDirectory()) {
            System.out.println("Invalid directory " + args[0]);
        }

        //Run for training data
        File[] trainingDataFiles = dataTrainDirectory.listFiles();
        if(trainingDataFiles == null || trainingDataFiles.length != 23) {
            System.out.println("Directory has invalid files");
            return;
        }
        DataSetReader<DataTrainRow> trainReader = new DataSetReader<>(new DataTrainConverter());
        for(File trainingDataFile : trainingDataFiles) {
            System.out.println("Fixing dataSet " + trainingDataFile.getName());
            trainReader.fix(trainingDataFile);
        }

        //Run for test data
        File[] testDataFiles = dataTestDirectory.listFiles();
        if(testDataFiles == null || testDataFiles.length != 23) {
            System.out.println("Directory has invalid files");
            return;
        }
        DataSetReader<DataTestRow> testReader = new DataSetReader<>(new DataTestConverter());
        for(File testDataFile : testDataFiles) {
            System.out.println("Fixing dataSet " + testDataFile.getName());
            testReader.fix(testDataFile);
        }
    }
}
