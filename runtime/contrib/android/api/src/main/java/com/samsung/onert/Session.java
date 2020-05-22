package com.samsung.onert;

// java
import java.util.HashMap;

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

    public void prepareInputs() {
        _native_sess.prepareInputs();
        _native_sess.setInputs();
    }

    public void prepareOutputs() {
        _native_sess.prepareOutputs();
        _native_sess.setOutputs();
    }

    public void source(Object[] inputs) {
        _native_sess.source(inputs);
    }

    public void sink(HashMap<Integer, Object> outputs) {
        _native_sess.sink(outputs);
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
