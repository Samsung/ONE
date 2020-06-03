package com.samsung.onert;

// java
import java.util.Arrays;
import java.util.HashMap;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.FloatBuffer;
import java.nio.ByteOrder;

// android
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;

final class NativeSessionWrapper implements AutoCloseable {

    static final String TAG = "ONERT_NATIVE";

    static final String LIB_NAME = "onert-native-api";

    static
    {
        System.loadLibrary(LIB_NAME);
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

    boolean prepare() {
        return (nativeSetAvailableBackends(_handle, _backends) &&
                nativePrepare(_handle));
    }

    Tensor[] prepareInputs() {
        final int count = nativeGetInputSize(_handle);
        Tensor[] inputs = new Tensor[count];
        for (int i = 0; i < count; ++i) {
            TensorInfo info = getInputTensorInfo(i);
            Tensor t = new Tensor(i, info);
            // allocate
            ByteBuffer bb = ByteBuffer.allocateDirect(t.getByteSize())
                            .order(ByteOrder.nativeOrder());
            t.buffer(bb);
            inputs[i] = t;
        }
        return inputs;
    }

    Tensor[] prepareOutputs() {
        final int count = nativeGetOutputSize(_handle);
        Tensor[] outputs = new Tensor[count];
        for (int i = 0; i < count; ++i) {
            TensorInfo info = getOutputTensorInfo(i);
            Tensor t = new Tensor(i, info);
            outputs[i] = t;

            // allocate temp buffers on jni level
            if (!nativeNewTempOutputBuf(_handle, i)) {
                Log.e(TAG, String.format("%s] nativeAllocTempOutputBuf for %d output failed", "prepareOutputs", i));
                return null;
            }
        }
        return outputs;
    }

    // TODO Layout
    boolean setInputs(Tensor[] inputs) {
        final int count = inputs.length;
        for (int i = 0; i < count; ++i) {
            Tensor t = inputs[i];
            if (!nativeSetInput(_handle, i, Helper.convertTensorType(t.type()), t.buffer(),
                                t.getByteSize())) {
                Log.e(TAG, String.format("%s] nativeSetInput failed", "setInputs"));
                return false;
            }
            if (!nativeSetInputLayout(_handle, i, 1)) { // CHANNELS_LAST
                Log.e(TAG, String.format("%s] nativeSetInputLayout failed", "setInputs"));
                return false;
            }
        }
        return true;
    }

    // TODO Layout
    boolean setOutputs(Tensor[] outputs) {
        final int count = outputs.length;
        for (int i = 0; i < count; ++i) {
            Tensor t = outputs[i];
            if (!nativeSetOutput(_handle, i, Helper.convertTensorType(t.type()))) {
                Log.e(TAG, String.format("%s] nativeSetOutput failed", "setOutputs"));
                return false;
            }
            if (!nativeSetOutputLayout(_handle, i, 1)) { // CHANNELS_LAST
                Log.e(TAG, String.format("%s] nativeSetOutputLayout failed", "setOutputs"));
                return false;
            }
        }
        return true;
    }

    // NOTE input/output buffer pointer should not be changed
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
        return Helper.newTensorInfo(info);
    }

    TensorInfo getOutputTensorInfo(int index) {
        InternalTensorInfo info = new InternalTensorInfo();
        nativeGetOutputTensorInfo(_handle, index, info);
        return Helper.newTensorInfo(info);
    }

    // TODO Hidden this method in setOutputs
    public ByteBuffer getOutputByteBuffer(int index) {
        ByteBuffer bb = nativeGetOutputBuf(_handle, index); // from jni
        bb.order(ByteOrder.nativeOrder());
        return bb;
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
        nativeDeleteAllTempOutputBuf(_handle);
        _handle = 0;
    }

    private long _handle = 0;
    private String _backends = null;

    // TODO How to handle enum in jni properly
    class InternalTensorInfo {
        int type;
        int rank;
        int[] shape;
    };

    // onert-native-api
    private native long nativeCreateSession();
    private native void nativeCloseSession(long handle);
    private native boolean nativeLoadModelFromFile(long handle, String nnpkg_path);
    private native boolean nativePrepare(long handle);
    private native boolean nativeRun(long handle);
    private native boolean nativeSetInput(long handle, int index, int type, ByteBuffer buffer, int byteSize);
    private native boolean nativeSetOutput(long handle, int index, int type);
    private native boolean nativeSetInputLayout(long handle, int index, int layout);
    private native boolean nativeSetOutputLayout(long handle, int index, int layout);
    private native int nativeGetInputSize(long handle);
    private native int nativeGetOutputSize(long handle);
    private native boolean nativeGetInputTensorInfo(long handle, int index, InternalTensorInfo info);
    private native boolean nativeGetOutputTensorInfo(long handle, int index, InternalTensorInfo info);
    private native boolean nativeSetAvailableBackends(long handle, String backends);
    private native boolean nativeNewTempOutputBuf(long handle, int index);
    private native boolean nativeDeleteTempOutputBuf(long handle, int index);
    private native ByteBuffer nativeGetOutputBuf(long handle, int index);
    private native void nativeDeleteAllTempOutputBuf(long handle);
}
