package com.samsung.onert;

public class TestCommon {
    static TensorInfo newTensorInfo() {
        TensorInfo.Type type = TensorInfo.Type.FLOAT32;
        int[] shape = new int[]{2,2};
        int rank = shape.length;
        return new TensorInfo(type, rank, shape);
    }

    static TensorInfo newTensorInfo2() {
        TensorInfo.Type type = TensorInfo.Type.UINT8;
        int[] shape = new int[]{1024, 128, 3};
        int rank = shape.length;
        return new TensorInfo(type, rank, shape);
    }
}