package com.samsung.onert;

// java
import java.util.Arrays;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.FloatBuffer;
import java.nio.ByteOrder;

// android
import android.util.Log;

final class Helper {
    static final String TAG = "ONERT_NATIVE";

    static boolean isByteBuffer(Object o) {
        return o instanceof ByteBuffer;
    }

    static ByteBuffer asByteBuffer(int[] int_arr) {
        int size = int_arr.length * Integer.BYTES;
        assert size > 0;

        ByteBuffer bb = ByteBuffer.allocateDirect(size).
               order(ByteOrder.nativeOrder());
        bb.clear();

        bb.asIntBuffer().put(int_arr);
        return bb;
    }

    static int[] asIntArray(ByteBuffer bb) {
        bb.rewind();
        int size = (bb.limit()) / Integer.BYTES +
                   ((bb.limit() % Integer.BYTES == 0) ? 0 : 1);
        assert size > 0;

        int[] int_arr = new int[size];

        IntBuffer ib = bb.asIntBuffer();
        ib.get(int_arr);
        return int_arr;
    }

    static int[] asIntArray(byte[] ba) {
        ByteBuffer bb = ByteBuffer.wrap(ba).order(ByteOrder.nativeOrder());
        return asIntArray(bb);
    }

    static void printByteBufferForDebug(ByteBuffer bb) {
        Log.d(TAG,
              String.format("ByteBuffer cap(%d) limit(%d) pos(%d) remaining(%d)",
              bb.capacity(), bb.limit(), bb.position(), bb.remaining()));
    }

    static void printIntArrayForDebug(int[] ia) {
        Log.d(TAG,
              String.format("IntArray(%d) %s", ia.length, Arrays.toString(ia)));
    }
    
    static void printArray(int[] array) {
        printIntArrayForDebug(array);
        Log.d(TAG, "PRINT_DATA_size: " + array.length);
    }

    static void printFloatBuffer(ByteBuffer bb, int size) {
        try {
            printByteBufferForDebug(bb);
            Log.d(TAG, "PRINT_DATA_size: " + size + ", capacity: " + bb.capacity());
            bb.position(0);
            FloatBuffer fb = bb.asFloatBuffer();

            int print_size = Math.min(size, 30);
            float[] data = new float[print_size];
            fb.get(data, 0, print_size);
            Log.d(TAG, "PRINT_DATA :" + Arrays.toString(data));
            bb.position(0);
        } catch (Exception e) {
            Log.d(TAG, "PRINT_DATA_EXCEPTION : " + e.toString());
        }
    }

    static int convertTensorType(TensorInfo.Type type) {
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

    static TensorInfo.Type convertOneRTTensorType(int type) {
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
}
