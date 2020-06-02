package com.samsung.onert;

import java.nio.ByteBuffer;

import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;

// TODO LAYOUT
public final class Tensor implements AutoCloseable {

    static final String TAG = "ONERT_NATIVE";

    public static boolean validateBuffer(ByteBuffer buffer) {
        if (buffer == null)
            return false;
        if (!buffer.isDirect())
            return false;
        return true;
    }

    public Tensor(int index, @NonNull TensorInfo info) {
        _index = index;
        _info = info;
    }

    public Tensor(int index, @NonNull int[] shape, @NonNull TensorInfo.Type type) {
        _index = index;
        _info = new TensorInfo(type, shape.length, shape);
    }

    public int index() {
        return _index;
    }

    public int[] shape() {
        return _info.shape;
    }

    public TensorInfo.Type type() {
        return _info.type;
    }

    public void buffer(ByteBuffer buffer) {
        _buffer = buffer;
    }

    public ByteBuffer buffer() {
        return _buffer;
    }

    public int getByteSize() {
        return TensorInfo.getByteSize(_info);
    }

    public int getSize() {
        return TensorInfo.getSize(_info);
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
        _buffer = null;
    }

    private final int _index;
    private TensorInfo _info = null;
    private ByteBuffer _buffer = null;
}
