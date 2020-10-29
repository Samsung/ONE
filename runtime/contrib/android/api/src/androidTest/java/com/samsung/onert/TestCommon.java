package com.samsung.onert;

import java.io.File;

import android.os.Environment;

public class TestCommon {
    // input: [1, 299, 299, 3], FLOAT32
    // output: [1, 149, 149, 32], FLOAT32
    static String test_nnpkg = Environment.getExternalStorageDirectory().getAbsolutePath() +
        "/nnpkg/convolution_test/";
    //static String test_nnpkg = "/data/local/tmp/nnpkg/convolution_test";

    static String getNnpkgPath() {
        return test_nnpkg;
    }

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