package com.samsung.onert;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

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

    public Tensor(@NonNull TensorInfo info) {
        _info = info;
        _buffer = ByteBuffer.allocateDirect(getByteSize())
                  .order(ByteOrder.nativeOrder());
    }

    public Tensor(@NonNull int[] shape, @NonNull TensorInfo.Type type) {
        this(new TensorInfo(type, shape.length, shape));
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

    private TensorInfo _info = null;
    private ByteBuffer _buffer = null;
}
