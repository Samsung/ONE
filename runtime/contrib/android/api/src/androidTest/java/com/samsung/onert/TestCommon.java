package com.samsung.onert;

import java.io.File;

import android.os.Environment;

public class TestCommon {
    static boolean exist(String path) {
        File f = new File(path);
        return (f.exists() && f.isDirectory());
    }

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