package com.samsung.onert;

public final class TensorInfo implements AutoCloseable {
    public enum Type {
        FLOAT32,
        INT32,
        QUANT8_ASYMM,
        BOOL,
        UINT8,
        UNKNOWN
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

    public static int getByteSize(TensorInfo info) {
        int size = TensorInfo.getTypeSize(info.type);
        int[] shape = info.shape;
        for (int i = 0; i < shape.length; ++i) {
            size *= shape[i];
        }
        return size;
    }

    public static int getSize(TensorInfo info) {
        int size = 1;
        int[] shape = info.shape;
        for (int i = 0; i < shape.length; ++i) {
            size *= shape[i];
        }
        return size;
    }

    public TensorInfo() {}

    public TensorInfo(Type t, int r, int[] s) {
        type = t;
        rank = r;
        shape = s;
    }
    public Type type;
    public int rank;
    public int[] shape;

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
        shape = null;
    }
}

