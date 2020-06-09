package com.samsung.onert;

// android
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;

public final class Session implements AutoCloseable {

    // TODO backends -> enum & bit
    public Session(@NonNull String nnpkg_path, @Nullable String backends) {
        _native_sess = new NativeSessionWrapper(nnpkg_path, backends);
    }

    public boolean prepare() {
        return _native_sess.prepare();
    }

    public void setInputs(Tensor[] inputs) {
        _native_sess.setInputs(inputs);
    }

    public void setOutputs(Tensor[] outputs) {
        _native_sess.setOutputs(outputs);
    }

    public int getInputSize() {
        return _native_sess.getInputSize();
    }

    public int getOutputSize() {
        return _native_sess.getOutputSize();
    }

    public TensorInfo getInputTensorInfo(int index) {
        return _native_sess.getInputTensorInfo(index);
    }

    public TensorInfo getOutputTensorInfo(int index) {
        return _native_sess.getOutputTensorInfo(index);
    }

    public boolean run() {
        return _native_sess.run();
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
        _native_sess = null;
    }

    private NativeSessionWrapper _native_sess = null;
}
