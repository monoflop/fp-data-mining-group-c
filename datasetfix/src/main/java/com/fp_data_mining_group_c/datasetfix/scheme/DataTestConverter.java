package com.fp_data_mining_group_c.datasetfix.scheme;

import com.fp_data_mining_group_c.datasetfix.DataSetReader;
import org.apache.commons.csv.CSVRecord;

import java.util.ArrayList;
import java.util.List;

public class DataTestConverter implements DataSetReader.DataSetConverter<DataTestRow> {

    private static final int DATASET_SENSOR_COUNT = 15;
    private static final int DATASET_SENSOR_SUM = (int)((Math.pow(DATASET_SENSOR_COUNT, 2) + DATASET_SENSOR_COUNT) / 2.0);

    @Override
    public DataTestRow fromCSVRecord(CSVRecord csvRecord) {
        DataTestRow dataRow = new DataTestRow();
        dataRow.setFrameNumber(Integer.parseInt(csvRecord.get("frame_number")));
        dataRow.setStripId(Integer.parseInt(csvRecord.get("strip_id")));
        dataRow.setNodeId(Integer.parseInt(csvRecord.get("node_id")));
        dataRow.setTimestamp(csvRecord.get("timestamp"));

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

        return dataRow;
    }

    @Override
    public void fixList(List<DataTestRow> dataList) {
        //Fix NaN values in r column
        double rMean = dataList.stream()
                .filter(row -> !Double.isNaN(row.getR()))
                .mapToDouble(DataTestRow::getR)
                .average()
                .orElse(Double.NaN);
        dataList.stream()
                .filter(row -> Double.isNaN(row.getR()))
                .forEach(row -> row.setR(rMean));

        //We assume that only one node is missing
        //Check if a node is missing
        if(dataList.size() != DATASET_SENSOR_COUNT) {
            //Find missing node id
            int nodeId = DATASET_SENSOR_SUM - dataList.stream().mapToInt(DataTestRow::getNodeId).sum();
            //System.out.println("Fixing node id #" + nodeId);

            //Add row with mean values
            DataTestRow dataRow = new DataTestRow();
            //frame_number,strip_id,node_id,timestamp,run_number,ax,ay,az,gx,gy,gz,mx,my,mz,r,near,vicon_x,vicon_y
            dataRow.setFrameNumber(dataList.get(0).getFrameNumber());//FrameNumber is for every element identical
            dataRow.setStripId(dataList.get(0).getStripId());//StripId is for every element identical
            dataRow.setNodeId(nodeId);
            dataRow.setTimestamp(dataList.get(0).getTimestamp());//Does not matter much

            //Sensor data
            //Create mean values
            dataRow.setAx(dataList.stream().mapToDouble(DataTestRow::getAx).average().orElse(Double.NaN));
            dataRow.setAy(dataList.stream().mapToDouble(DataTestRow::getAy).average().orElse(Double.NaN));
            dataRow.setAz(dataList.stream().mapToDouble(DataTestRow::getAz).average().orElse(Double.NaN));
            dataRow.setGx(dataList.stream().mapToDouble(DataTestRow::getGx).average().orElse(Double.NaN));
            dataRow.setGy(dataList.stream().mapToDouble(DataTestRow::getGy).average().orElse(Double.NaN));
            dataRow.setGz(dataList.stream().mapToDouble(DataTestRow::getGz).average().orElse(Double.NaN));
            dataRow.setMx(dataList.stream().mapToDouble(DataTestRow::getMx).average().orElse(Double.NaN));
            dataRow.setMy(dataList.stream().mapToDouble(DataTestRow::getMy).average().orElse(Double.NaN));
            dataRow.setMz(dataList.stream().mapToDouble(DataTestRow::getMz).average().orElse(Double.NaN));
            dataRow.setR(dataList.stream().mapToDouble(DataTestRow::getR).average().orElse(Double.NaN));

            //Add
            dataList.add(nodeId - 1, dataRow);
        }
    }

    @Override
    public List<String> toValueString(DataTestRow dataTestRow) {
        List<String> valueList = new ArrayList<>();
        valueList.add(String.valueOf(dataTestRow.getFrameNumber()));
        valueList.add(String.valueOf(dataTestRow.getStripId()));
        valueList.add(String.valueOf(dataTestRow.getNodeId()));
        valueList.add(dataTestRow.getTimestamp());
        valueList.add(String.valueOf(dataTestRow.getAx()));
        valueList.add(String.valueOf(dataTestRow.getAy()));
        valueList.add(String.valueOf(dataTestRow.getAz()));
        valueList.add(String.valueOf(dataTestRow.getGx()));
        valueList.add(String.valueOf(dataTestRow.getGy()));
        valueList.add(String.valueOf(dataTestRow.getGz()));
        valueList.add(String.valueOf(dataTestRow.getMx()));
        valueList.add(String.valueOf(dataTestRow.getMy()));
        valueList.add(String.valueOf(dataTestRow.getMz()));
        valueList.add(String.valueOf(dataTestRow.getR()));
        return valueList;
    }
}
