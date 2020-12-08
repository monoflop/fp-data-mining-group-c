package com.fp_data_mining_group_c.datasetfix.scheme;

public class DataTrainRow {
    private int frameNumber;
    private int stripId;
    private int nodeId;
    private String timestamp;//Maybe LocalDateTime, but we do not use this value anyway
    private int runNumber;
    private double ax;
    private double ay;
    private double az;
    private double gx;
    private double gy;
    private double gz;
    private double mx;
    private double my;
    private double mz;
    private double r;
    private double near;
    private double viconX;
    private double viconY;

    public int getFrameNumber() {
        return frameNumber;
    }

    public void setFrameNumber(int frameNumber) {
        this.frameNumber = frameNumber;
    }

    public int getStripId() {
        return stripId;
    }

    public void setStripId(int stripId) {
        this.stripId = stripId;
    }

    public int getNodeId() {
        return nodeId;
    }

    public void setNodeId(int nodeId) {
        this.nodeId = nodeId;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public int getRunNumber() {
        return runNumber;
    }

    public void setRunNumber(int runNumber) {
        this.runNumber = runNumber;
    }

    public double getAx() {
        return ax;
    }

    public void setAx(double ax) {
        this.ax = ax;
    }

    public double getAy() {
        return ay;
    }

    public void setAy(double ay) {
        this.ay = ay;
    }

    public double getAz() {
        return az;
    }

    public void setAz(double az) {
        this.az = az;
    }

    public double getGx() {
        return gx;
    }

    public void setGx(double gx) {
        this.gx = gx;
    }

    public double getGy() {
        return gy;
    }

    public void setGy(double gy) {
        this.gy = gy;
    }

    public double getGz() {
        return gz;
    }

    public void setGz(double gz) {
        this.gz = gz;
    }

    public double getMx() {
        return mx;
    }

    public void setMx(double mx) {
        this.mx = mx;
    }

    public double getMy() {
        return my;
    }

    public void setMy(double my) {
        this.my = my;
    }

    public double getMz() {
        return mz;
    }

    public void setMz(double mz) {
        this.mz = mz;
    }

    public double getR() {
        return r;
    }

    public void setR(double r) {
        this.r = r;
    }

    public double getNear() {
        return near;
    }

    public void setNear(double near) {
        this.near = near;
    }

    public double getViconX() {
        return viconX;
    }

    public void setViconX(double viconX) {
        this.viconX = viconX;
    }

    public double getViconY() {
        return viconY;
    }

    public void setViconY(double viconY) {
        this.viconY = viconY;
    }
}
