package com.samsung.onert;

import android.support.annotation.NonNull;
import android.support.annotation.Nullable;

public final class Session implements AutoCloseable {

    private NativeSessionWrapper _session = null;

    // TODO backends -> enum & bit
    public Session(@NonNull String nnpkg_path, @Nullable String backends) {
        _session = new NativeSessionWrapper(nnpkg_path, backends);
    }

    public boolean prepare() {
        return _session.prepare();
    }

    public boolean run(Tensor[] inputs, Tensor[] outputs) {
        _session.setInputs(inputs);
        _session.setOutputs(outputs);
        return _session.run();
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
        _session = null;
    }
}
