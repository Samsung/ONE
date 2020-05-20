package com.samsung.onert;

import java.nio.ByteBuffer;

import android.support.annotation.NonNull;
import android.support.annotation.Nullable;

// TODO LAYOUT
public final class Tensor implements AutoCloseable {
    public enum Type {
        FLOAT32,
        INT32,
        QUANT8_ASYMM,
        BOOL,
        UINT8
    }

    public static int getTypeSize(Type type) {
        int size = 0;
        switch (type) {
            case FLOAT32:
            case INT32:
                size = 4;
                break;
            case QUANT8_ASYMM:
            case BOOL: // Note. different from java's one
            case UINT8:
                size = 1;
                break;
            default:
                size = -1;
                break;
        }
        return size;
    }

    public Tensor(@NonNull int[] shape, @NonNull Type type) {
        _shape = shape;
        _type = type;
    }

    public int[] shape() {
        return _shape;
    }

    public Type type() {
        return _type;
    }

    public ByteBuffer buffer() {
        return _buffer;
    }

    // ByteBuffer Should be done by allocateDirect
    public void buffer(ByteBuffer buffer) {
        _buffer = buffer;
    }

    public int getByteSize() {
        int size = getTypeSize(_type);
        for (int i = 0; i < _shape.length; ++i) {
            size *= _shape[i];
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

    private int[] _shape = null;
    private Type _type = null;
    private ByteBuffer _buffer = null;
}
