package com.fp_data_mining_group_c.datasetfix.scheme;

import com.fp_data_mining_group_c.datasetfix.DataSetReader;
import org.apache.commons.csv.CSVRecord;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.ToDoubleFunction;

public class DataTrainConverter implements DataSetReader.DataSetConverter<DataTrainRow> {
    private static final int DATASET_SENSOR_COUNT = 15;
    private static final int DATASET_SENSOR_SUM = (int)((Math.pow(DATASET_SENSOR_COUNT, 2) + DATASET_SENSOR_COUNT) / 2.0);

    @Override
    public DataTrainRow fromCSVRecord(CSVRecord csvRecord) {
        DataTrainRow dataRow = new DataTrainRow();
        dataRow.setFrameNumber(Integer.parseInt(csvRecord.get("frame_number")));
        dataRow.setStripId(Integer.parseInt(csvRecord.get("strip_id")));
        dataRow.setNodeId(Integer.parseInt(csvRecord.get("node_id")));
        dataRow.setTimestamp(csvRecord.get("timestamp"));
        dataRow.setRunNumber(Integer.parseInt(csvRecord.get("run_number")));

        dataRow.setAx(Double.parseDouble(csvRecord.get("ax")));
        dataRow.setAy(Double.parseDouble(csvRecord.get("ay")));
        dataRow.setAz(Double.parseDouble(csvRecord.get("az")));

        dataRow.setGx(Double.parseDouble(csvRecord.get("gx")));
        dataRow.setGy(Double.parseDouble(csvRecord.get("gy")));
        dataRow.setGz(Double.parseDouble(csvRecord.get("gz")));

        dataRow.setMx(Double.parseDouble(csvRecord.get("mx")));
        dataRow.setMy(Double.parseDouble(csvRecord.get("my")));
        dataRow.setMz(Double.parseDouble(csvRecord.get("mz")));

        //R values is maybe empty...
        dataRow.setR(csvRecord.get("r").isEmpty() ? Double.NaN : Double.parseDouble(csvRecord.get("r")));

        dataRow.setNear(Double.parseDouble(csvRecord.get("near")));
        dataRow.setViconX(Double.parseDouble(csvRecord.get("vicon_x")));
        dataRow.setViconY(Double.parseDouble(csvRecord.get("vicon_y")));

        return dataRow;
    }

    @Override
    public void fixList(List<DataTrainRow> dataList) {
        //Fix NaN values in r column
        double rMean = dataList.stream()
                .filter(row -> !Double.isNaN(row.getR()))
                .mapToDouble(DataTrainRow::getR)
                .average()
                .orElse(Double.NaN);
        //System.out.println("R mean " + rMean);
        dataList.stream()
                .filter(row -> Double.isNaN(row.getR()))
                .forEach(row -> row.setR(rMean));

        //We assume that only one node is missing
        //Check if a node is missing
        if(dataList.size() != DATASET_SENSOR_COUNT) {
            //Find missing node id
            int nodeId = DATASET_SENSOR_SUM - dataList.stream().mapToInt(DataTrainRow::getNodeId).sum();
            //System.out.println("Fixing node id #" + nodeId);

            //Add row with mean values
            DataTrainRow dataRow = new DataTrainRow();
            //frame_number,strip_id,node_id,timestamp,run_number,ax,ay,az,gx,gy,gz,mx,my,mz,r,near,vicon_x,vicon_y
            dataRow.setFrameNumber(dataList.get(0).getFrameNumber());//FrameNumber is for every element identical
            dataRow.setStripId(dataList.get(0).getStripId());//StripId is for every element identical
            dataRow.setNodeId(nodeId);
            dataRow.setTimestamp(dataList.get(0).getTimestamp());//Does not matter much
            dataRow.setRunNumber(dataList.get(0).getRunNumber());
            dataRow.setNear(dataList.get(0).getNear());
            dataRow.setViconX(dataList.get(0).getViconX());
            dataRow.setViconY(dataList.get(0).getViconY());

            //Sensor data
            //Create mean values
            dataRow.setAx(dataList.stream().mapToDouble(DataTrainRow::getAx).average().orElse(Double.NaN));
            dataRow.setAy(dataList.stream().mapToDouble(DataTrainRow::getAy).average().orElse(Double.NaN));
            dataRow.setAz(dataList.stream().mapToDouble(DataTrainRow::getAz).average().orElse(Double.NaN));
            dataRow.setGx(dataList.stream().mapToDouble(DataTrainRow::getGx).average().orElse(Double.NaN));
            dataRow.setGy(dataList.stream().mapToDouble(DataTrainRow::getGy).average().orElse(Double.NaN));
            dataRow.setGz(dataList.stream().mapToDouble(DataTrainRow::getGz).average().orElse(Double.NaN));
            dataRow.setMx(dataList.stream().mapToDouble(DataTrainRow::getMx).average().orElse(Double.NaN));
            dataRow.setMy(dataList.stream().mapToDouble(DataTrainRow::getMy).average().orElse(Double.NaN));
            dataRow.setMz(dataList.stream().mapToDouble(DataTrainRow::getMz).average().orElse(Double.NaN));
            dataRow.setR(dataList.stream().mapToDouble(DataTrainRow::getR).average().orElse(Double.NaN));

            //Add
            dataList.add(nodeId - 1, dataRow);
        }
    }

    @Override
    public List<String> toValueString(DataTrainRow dataTrainRow) {
        List<String> valueList = new ArrayList<>();
        valueList.add(String.valueOf(dataTrainRow.getFrameNumber()));
        valueList.add(String.valueOf(dataTrainRow.getStripId()));
        valueList.add(String.valueOf(dataTrainRow.getNodeId()));
        valueList.add(dataTrainRow.getTimestamp());
        valueList.add(String.valueOf(dataTrainRow.getRunNumber()));
        valueList.add(String.valueOf(dataTrainRow.getAx()));
        valueList.add(String.valueOf(dataTrainRow.getAy()));
        valueList.add(String.valueOf(dataTrainRow.getAz()));
        valueList.add(String.valueOf(dataTrainRow.getGx()));
        valueList.add(String.valueOf(dataTrainRow.getGy()));
        valueList.add(String.valueOf(dataTrainRow.getGz()));
        valueList.add(String.valueOf(dataTrainRow.getMx()));
        valueList.add(String.valueOf(dataTrainRow.getMy()));
        valueList.add(String.valueOf(dataTrainRow.getMz()));
        valueList.add(String.valueOf(dataTrainRow.getR()));
        valueList.add(String.valueOf(dataTrainRow.getNear()));
        valueList.add(String.valueOf(dataTrainRow.getViconX()));
        valueList.add(String.valueOf(dataTrainRow.getViconY()));
        return valueList;
    }

    private void replaceDoubleNanByMean(List<DataTrainRow> dataList,
                                        Predicate<? super DataTrainRow> filter,
                                        ToDoubleFunction<? super DataTrainRow> toDoubleFunction,
                                        Consumer<? super DataTrainRow> action) {
        double rMean = dataList.stream()
                .filter(filter.negate())
                .mapToDouble(toDoubleFunction)
                .average()
                .orElse(Double.NaN);
        dataList.stream()
                .filter(filter)
                .forEach(row -> row.setR(rMean));
    }
}
