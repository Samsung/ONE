package com.samsung.onert;

// java
import java.util.HashMap;
import java.nio.ByteBuffer;

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

    public Tensor[] prepareInputs() {
        return _native_sess.prepareInputs();
    }

    public void setInputs(Tensor[] inputs) {
        _native_sess.setInputs(inputs);
    }

    public Tensor[] prepareOutputs() {
        return _native_sess.prepareOutputs();
    }

    public void setOutputs(Tensor[] outputs) {
        _native_sess.setOutputs(outputs);
    }

    // TODO Hidden this method in setOutputs
    public ByteBuffer getOutputByteBuffer(int index) {
        return _native_sess.getOutputByteBuffer(index);
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
