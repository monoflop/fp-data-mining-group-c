package com.fp_data_mining_group_c.datasetfix;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class DataSetReader<T> {
    private final DataSetConverter<T> dataSetConverter;

    public DataSetReader(DataSetConverter<T> dataSetConverter) {
        this.dataSetConverter = dataSetConverter;
    }

    public void fix(File dataFile) throws IOException {
        //Read csv file
        BufferedReader reader = new BufferedReader(new FileReader(dataFile));
        Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader);

        //Read header line
        reader = new BufferedReader(new FileReader(dataFile));
        String headerString = reader.readLine();
        String[] header = headerString.split(",");

        //Create writer for fixed csv file
        File tempFile = new File(dataFile.getParent(), dataFile.getName() + "_tmp");
        BufferedWriter writer = Files.newBufferedWriter(tempFile.toPath());
        CSVPrinter csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT
                .withHeader(header));

        //Collect frames
        int currentFrame = 0;
        List<CSVRecord> currentFrameList = new ArrayList<>();
        for (CSVRecord record : records) {
            int frameNumber = Integer.parseInt(record.get("frame_number"));

            //Add to current frame
            if (frameNumber != currentFrame) {
                List<T> dataRowList = createFixedFrame(currentFrameList);

                //Write
                for (T t : dataRowList) {
                    csvPrinter.printRecord(dataSetConverter.toValueString(t));
                }

                currentFrame = frameNumber;
                currentFrameList = new ArrayList<>();
            }
            currentFrameList.add(record);
        }

        List<T> dataRowList = createFixedFrame(currentFrameList);
        for (T t : dataRowList) {
            csvPrinter.printRecord(dataSetConverter.toValueString(t));
        }

        csvPrinter.flush();

        //Delete original file
        if(!dataFile.delete() || !tempFile.renameTo(dataFile)) {
            throw new IOException("Failed to delete or rename file");
        }
    }

    private List<T> createFixedFrame(List<CSVRecord> frame) {
        List<T> dataRowList = new ArrayList<>();
        for(CSVRecord record : frame) {
            dataRowList.add(dataSetConverter.fromCSVRecord(record));
        }
        dataSetConverter.fixList(dataRowList);
        return dataRowList;
    }

    public interface DataSetConverter<T> {
        T fromCSVRecord(CSVRecord csvRecord);
        void fixList(List<T> dataList);
        List<String> toValueString(T t);
    }
}
