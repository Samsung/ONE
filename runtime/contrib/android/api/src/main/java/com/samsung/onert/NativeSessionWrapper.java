package com.samsung.onert;

// java
import java.util.Arrays;
import java.util.HashMap;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.ByteOrder;

// android
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.util.Log;

final class NativeSessionWrapper implements AutoCloseable {

    static final String TAG = "ONERT_NATIVE";

    static
    {
        System.loadLibrary("onert-native-api");
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

    void prepareInputs() {
        final int count = nativeGetInputSize(_handle);
        _inputs = new Tensor[count];
        for (int i = 0; i < count; ++i) {
            TensorInfo info = getInputTensorInfo(i);
            Tensor t = new Tensor(i, info);
            // allocate
            ByteBuffer bb = ByteBuffer.allocateDirect(t.getByteSize())
                            .order(ByteOrder.nativeOrder());
            t.buffer(bb);
            _inputs[i] = t;
        }
    }

    void prepareOutputs() {
        final int count = nativeGetOutputSize(_handle);
        _outputs = new Tensor[count];
        for (int i = 0; i < count; ++i) {
            TensorInfo info = getOutputTensorInfo(i);
            Tensor t = new Tensor(i, info);
            _outputs[i] = t;

            // allocate temp buffers on jni level
            if (!nativeNewTempOutputBuf(_handle, i, TensorInfo.getByteSize(info))) {
                Log.e(TAG, String.format("%s] nativeAllocTempOutputBuf for %d output failed", "prepareOutputs", i));
                return;
            }
        }
    }

    // TODO Support other types of input
    void source(Object[] inputs) {
        Log.d(TAG, "##### source()");
        final int count = inputs.length;
        for (int i = 0; i < count; ++i) {
            Tensor t = _inputs[i];
            ByteBuffer tb = t.buffer(); // from out of onert such as sample apps
            Log.d(TAG, String.format("#%d Tensor's ByteBuffer", i));
            if (isByteBuffer(inputs[i])) {
                Log.d(TAG, String.format("ByteBuffer tb")); printByteBufferForDebug(tb);
                tb.clear();
                Log.d(TAG, String.format("ByteBuffer tb after clear()")); printByteBufferForDebug(tb);
                ByteBuffer bb = (ByteBuffer)inputs[i];
                Log.d(TAG, String.format("ByteBuffer bb")); printByteBufferForDebug(bb);
                tb.put(bb); // copied
                Log.d(TAG, String.format("ByteBuffer tb after put()")); printByteBufferForDebug(tb);
                Log.d(TAG, String.format("ByteBuffer bb after put()")); printByteBufferForDebug(bb);
            }
            else { // if not, handle it as int[]
                Log.d(TAG, String.format("ByteBuffer tb")); printByteBufferForDebug(tb);
                tb.clear();
                Log.d(TAG, String.format("ByteBuffer tb after clear()")); printByteBufferForDebug(tb);
                ByteBuffer bb = asByteBuffer((int[])inputs[i]);
                Log.d(TAG, String.format("ByteBuffer bb")); printByteBufferForDebug(bb);
                tb.put(bb);
                Log.d(TAG, String.format("ByteBuffer tb after put()")); printByteBufferForDebug(tb);
                Log.d(TAG, String.format("ByteBuffer bb after put()")); printByteBufferForDebug(bb);
            }
        }
    }

    // TODO Support other types of output
    void sink(HashMap<Integer, Object> outputs) {
        Log.d(TAG, "##### sink()");
        final int count = outputs.size();
        for (int i = 0; i < count; ++i) {
            Log.d(TAG, String.format("#%d Tensor's ByteBuffer", i));
            ByteBuffer bb = nativeGetOutputBuf(_handle, i); // from jni
            Log.d(TAG, String.format("ByteBuffer bb")); printByteBufferForDebug(bb);
            //bb.flip();
            //Log.d(TAG, String.format("ByteBuffer bb after flip()")); printByteBufferForDebug(bb);
            bb.rewind();
            Log.d(TAG, String.format("ByteBuffer bb after rewind()")); printByteBufferForDebug(bb);
            if (isByteBuffer(outputs.get(i))) {
                ByteBuffer obb = (ByteBuffer)outputs.get(i); // to out of onert such as sample apps
                Log.d(TAG, String.format("ByteBuffer obb")); printByteBufferForDebug(obb);
                obb.clear();
                Log.d(TAG, String.format("ByteBuffer obb after clear()")); printByteBufferForDebug(obb);
                obb.put(bb); // copied
                Log.d(TAG, String.format("ByteBuffer obb after put()")); printByteBufferForDebug(obb);
            }
            else { // if not, handle it as int[]
                int[] ia = asIntArray(bb);
                Log.d(TAG, String.format("int[] ia")); printIntArrayForDebug(ia);
                int[] oia = (int[])outputs.get(i);
                Log.d(TAG, String.format("int[] oia")); printIntArrayForDebug(oia);
                System.arraycopy(ia, 0, oia, 0, ia.length);
                Log.d(TAG, String.format("int[] oia after arraycopy")); printIntArrayForDebug(oia);
            }
        }
    }

    // TODO Layout
    boolean setInputs() {
        final int count = _inputs.length;
        for (int i = 0; i < count; ++i) {
            Tensor t = _inputs[i];
            if (!nativeSetInput(_handle, i, convertTensorType(t.type()), t.buffer(),
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
    boolean setOutputs() {
        final int count = _outputs.length;
        for (int i = 0; i < count; ++i) {
            Tensor t = _outputs[i];
            if (!nativeSetOutput(_handle, i, convertTensorType(t.type()))) {
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
        return newTensorInfo(info);
    }

    TensorInfo getOutputTensorInfo(int index) {
        InternalTensorInfo info = new InternalTensorInfo();
        nativeGetOutputTensorInfo(_handle, index, info);
        return newTensorInfo(info);
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
        final int count = _outputs.length;
        for (int i = 0; i < count; ++i) {
            nativeDeleteTempOutputBuf(_handle, i);
        }
        _outputs = null;
        _inputs = null;
        _handle = 0;
    }

    private long _handle = 0;
    private String _backends = null;
    private Tensor[] _inputs;
    private Tensor[] _outputs;

    private static int convertTensorType(TensorInfo.Type type) {
        int ret = 0;
        switch (type) {
            case FLOAT32: 
                ret = 0; break;
            case INT32:
                ret = 1; break;
            case QUANT8_ASYMM:
                ret = 2; break;
            case BOOL:
                ret = 3; break;
            case UINT8:
                ret = 4; break;
            default:
                ret = -1; break;
        }
        return ret;
    }

    private static TensorInfo.Type convertOneRTTensorType(int type) {
        TensorInfo.Type ret;
        switch (type) {
            case 0:
                ret = TensorInfo.Type.FLOAT32; break;
            case 1:
                ret = TensorInfo.Type.INT32; break;
            case 2:
                ret = TensorInfo.Type.QUANT8_ASYMM; break;
            case 3:
                ret = TensorInfo.Type.BOOL; break;
            case 4:
                ret = TensorInfo.Type.UINT8; break;
            default:
                ret = TensorInfo.Type.UNKNOWN; break;
        }
        return ret;
    }

    // TODO How to handle enum in jni properly
    class InternalTensorInfo {
        int type;
        int rank;
        int[] shape;
    };

    private TensorInfo newTensorInfo(InternalTensorInfo info) {
        TensorInfo.Type type = convertOneRTTensorType(info.type);
        int rank = info.rank;
        int[] shape = info.shape;
        return new TensorInfo(type, rank, shape);
    }

    private static boolean isByteBuffer(Object o) {
        return o instanceof ByteBuffer;
    }

    private static ByteBuffer asByteBuffer(int[] int_arr) {
        int size = int_arr.length * Integer.BYTES;
        assert size > 0;

        ByteBuffer bb = ByteBuffer.allocateDirect(size).
               order(ByteOrder.nativeOrder());
        bb.clear();

        bb.asIntBuffer().put(int_arr);
        return bb;
    }

    private static int[] asIntArray(ByteBuffer bb) {
        bb.flip();
        int size = (bb.limit()) / Integer.BYTES +
                   ((bb.limit() % Integer.BYTES == 0) ? 0 : 1);
        assert size > 0;

        int[] int_arr = new int[size];

        IntBuffer ib = bb.asIntBuffer();
        ib.get(int_arr);
        return int_arr;
    }

    private static void printByteBufferForDebug(ByteBuffer bb) {
        Log.d(TAG,
              String.format("ByteBuffer cap(%d) limit(%d) pos(%d) remaining(%d)",
              bb.capacity(), bb.limit(), bb.position(), bb.remaining()));
    }

    private static void printIntArrayForDebug(int[] ia) {
        Log.d(TAG,
              String.format("IntArray(%d) %s", ia.length, Arrays.toString(ia)));
    }

    // onert-native-api
    private native long nativeCreateSession();
    private native void nativeCloseSession(long handle);
    private native boolean nativeLoadModelFromFile(long handle, String nnpkg_path);
    private native boolean nativePrepare(long handle);
    private native boolean nativeRun(long handle);
    private native boolean nativeSetInput(long handle, int index, int type, ByteBuffer buffer, int byteSize);
    //private native boolean nativeSetOutput(long handle, int index, int type, ByteBuffer buffer, int byteSize);
    private native boolean nativeSetOutput(long handle, int index, int type);
    private native boolean nativeSetInputLayout(long handle, int index, int layout);
    private native boolean nativeSetOutputLayout(long handle, int index, int layout);
    private native int nativeGetInputSize(long handle);
    private native int nativeGetOutputSize(long handle);
    private native boolean nativeGetInputTensorInfo(long handle, int index, InternalTensorInfo info);
    private native boolean nativeGetOutputTensorInfo(long handle, int index, InternalTensorInfo info);
    private native boolean nativeSetAvailableBackends(long handle, String backends);
    private native boolean nativeNewTempOutputBuf(long handle, int index, int byteSize);
    private native boolean nativeDeleteTempOutputBuf(long handle, int index);
    private native ByteBuffer nativeGetOutputBuf(long handle, int index);
}
