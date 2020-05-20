package com.samsung.onert;

import java.nio.ByteBuffer;

import android.support.annotation.NonNull;
import android.support.annotation.Nullable;

// TODO LAYOUT
public final class Tensor implements AutoCloseable {

    public Tensor(@NonNull TensorInfo info) {
        _info = info;
    }

    public Tensor(@NonNull int[] shape, @NonNull TensorInfo.Type type) {
        _info = new TensorInfo(type, shape.length, shape);
    }

    public int[] shape() {
        return _info.shape;
    }

    public TensorInfo.Type type() {
        return _info.type;
    }

    public ByteBuffer buffer() {
        return _buffer;
    }

    // ByteBuffer Should be done by ByteBuffer.allocateDirect
    public void buffer(ByteBuffer buffer) {
        _buffer = buffer;
    }

    public int getByteSize() {
        int size = TensorInfo.getTypeSize(_info.type);
        int[] shape = _info.shape;
        for (int i = 0; i < shape.length; ++i) {
            size *= shape[i];
        }
        return size;
    }

    public boolean validate() {
        if (_buffer == null)
            return false;
        if (!_buffer.isDirect())
            return false;
        if (_buffer.capacity() != getByteSize())
            return false;
        return true;
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
