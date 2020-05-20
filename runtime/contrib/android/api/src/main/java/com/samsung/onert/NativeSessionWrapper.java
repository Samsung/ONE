package com.samsung.onert;

import java.nio.ByteBuffer;

import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;

final class NativeSessionWrapper implements AutoCloseable {

    static final String TAG = "ONERT_NATIVE";

    static
    {
        System.loadLibrary("onert-native-api");
    }

    static final String DEFAULT_BACKENDS = "cpu";

    NativeSessionWrapper(@NonNull String nnpkg_path) {
        this(nnpkg_path, DEFAULT_BACKENDS);
    }

    NativeSessionWrapper(@NonNull String nnpkg_path, @NonNull String backends) {
        _handle = nativeCreateSession();
        nativeLoadModelFromFile(_handle, nnpkg_path);
        _backends = backends;
    }

    // TODO Layout
    boolean setInputs(Tensor[] inputs) {
        final int input_size = nativeGetInputSize(_handle);
        if (input_size != inputs.length)
            return false;

        for (int i = 0; i < inputs.length; ++i) {
            Tensor t = inputs[i];
            if (!t.validate() ||
                !nativeSetInput(_handle, i, convertTensorType(t.type()),
                                t.buffer(), t.getByteSize()) ||
                !nativeSetInputLayout(_handle, i, 1)) // CHANNELS_LAST
                return false;
        }
        return true;
    }

    // TODO Layout
    boolean setOutputs(Tensor[] outputs) {
        final int output_size = nativeGetOutputSize(_handle);
        if (output_size != outputs.length)
            return false;

        for (int i = 0; i < outputs.length; ++i) {
            Tensor t = outputs[i];
            if (!t.validate() ||
                !nativeSetOutput(_handle, i, convertTensorType(t.type()),
                                 t.buffer(), t.getByteSize()) ||
                !nativeSetOutputLayout(_handle, i, 1)) // CHANNELS_LAST
                return false;
        }
        return true;

    }

    boolean prepare() {
        return (nativeSetAvailableBackends(_handle, _backends) &&
                nativePrepare(_handle));
    }

    boolean run() {
        return nativeRun(_handle);
    }

    int getInputSize() {
        return nativeGetInputSize(_handle);
    }

    int getOutputSize() {
        return nativeGetOutputSize(_handle);
    }

    TensorInfo getInputTensorInfo(int index) {
        InternalTensorInfo info = new InternalTensorInfo();
        nativeGetInputTensorInfo(_handle, index, info);
        return newTensorInfo(info);
    }

    TensorInfo getOutputTensorInfo(int index) {
        InternalTensorInfo info = new InternalTensorInfo();
        nativeGetOutputTensorInfo(_handle, index, info);
        return newTensorInfo(info);
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }

    @Override
    public void close() {
        nativeCloseSession(_handle);
    }

    private long _handle = 0l;
    private String _backends = null;

    private int convertTensorType(TensorInfo.Type type) {
        int ret = 0;
        switch (type) {
            case FLOAT32: 
                ret = 0; break;
            case INT32:
                ret = 1; break;
            case QUANT8_ASYMM:
                ret = 2; break;
            case BOOL:
                ret = 3; break;
            case UINT8:
                ret = 4; break;
            default:
                ret = -1; break;
        }
        return ret;
    }

    private TensorInfo.Type convertOneRTTensorType(int type) {
        TensorInfo.Type ret;
        switch (type) {
            case 0:
                ret = TensorInfo.Type.FLOAT32; break;
            case 1:
                ret = TensorInfo.Type.INT32; break;
            case 2:
                ret = TensorInfo.Type.QUANT8_ASYMM; break;
            case 3:
                ret = TensorInfo.Type.BOOL; break;
            case 4:
                ret = TensorInfo.Type.UINT8; break;
            default:
                ret = TensorInfo.Type.UNKNOWN; break;
        }
        return ret;
    }

    // TODO How to handle enum in jni properly
    class InternalTensorInfo {
        int type;
        int rank;
        int[] shape;
    };

    private TensorInfo newTensorInfo(InternalTensorInfo info) {
        TensorInfo.Type type = convertOneRTTensorType(info.type);
        int rank = info.rank;
        int[] shape = info.shape;
        return new TensorInfo(type, rank, shape);
    }

    // onert-native-api
    private native long nativeCreateSession();
    private native void nativeCloseSession(long handle);
    private native boolean nativeLoadModelFromFile(long handle, String nnpkg_path);
    private native boolean nativePrepare(long handle);
    private native boolean nativeRun(long handle);
    private native boolean nativeSetInput(long handle, int index, int type, ByteBuffer buffer, int byteSize);
    private native boolean nativeSetOutput(long handle, int index, int type, ByteBuffer buffer, int byteSize);
    private native boolean nativeSetInputLayout(long handle, int index, int layout);
    private native boolean nativeSetOutputLayout(long handle, int index, int layout);
    private native int nativeGetInputSize(long handle);
    private native int nativeGetOutputSize(long handle);
    private native boolean nativeGetInputTensorInfo(long handle, int index, InternalTensorInfo info);
    private native boolean nativeGetOutputTensorInfo(long handle, int index, InternalTensorInfo info);
    private native boolean nativeSetAvailableBackends(long handle, String backends);
}
