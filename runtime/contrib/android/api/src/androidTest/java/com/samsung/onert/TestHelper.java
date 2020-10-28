package com.samsung.onert;

import androidx.test.filters.SmallTest;
import androidx.test.runner.AndroidJUnit4;
import org.junit.Before;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import static com.google.common.truth.Truth.*;

import java.nio.ByteBuffer;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.ByteOrder;

@RunWith(AndroidJUnit4.class)
@SmallTest
public class TestHelper {
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    @Test
    public void test_isByteBuffer_p() {
        ByteBuffer bb = ByteBuffer.allocateDirect(16);
        assertThat(Helper.isByteBuffer(bb)).isTrue();
    }

    @Test
    public void test_asByteBuffer_p() {
        int[] ia = new int[]{1,2,3,4};
        ByteBuffer byte_buf = Helper.asByteBuffer(ia);
        assertThat(byte_buf.getInt()).isEqualTo(1);
        assertThat(byte_buf.getInt()).isEqualTo(2);
        assertThat(byte_buf.getInt()).isEqualTo(3);
        assertThat(byte_buf.getInt()).isEqualTo(4);
    }

    @Test
    public void test_asIntArray_p() {
        ByteBuffer bb = ByteBuffer.allocateDirect(16).
                            order(ByteOrder.nativeOrder());
        bb.putInt(1);
        bb.putInt(2);
        bb.putInt(3);
        bb.putInt(4);
        int[] ia = Helper.asIntArray(bb);
        assertThat(ia[0]).isEqualTo(1);
        assertThat(ia[1]).isEqualTo(2);
        assertThat(ia[2]).isEqualTo(3);
        assertThat(ia[3]).isEqualTo(4);
    }

    @Test
    public void test_asIntArray_p2() {
        byte[] ba = null;
        if (ByteOrder.nativeOrder() == ByteOrder.BIG_ENDIAN) {
            ba = new byte[]{0x00, 0x00, 0x00, 0x01,
                            0x00, 0x00, 0x00, 0x02,
                            0x00, 0x00, 0x00, 0x03,
                            0x00, 0x00, 0x00, 0x04};
        } else { // LITTLE_ENDIEN
            ba = new byte[]{0x01, 0x00, 0x00, 0x00,
                            0x02, 0x00, 0x00, 0x00,
                            0x03, 0x00, 0x00, 0x00,
                            0x04, 0x00, 0x00, 0x00};
        }
        int[] ia = Helper.asIntArray(ba);
        assertThat(ia[0]).isEqualTo(1);
        assertThat(ia[1]).isEqualTo(2);
        assertThat(ia[2]).isEqualTo(3);
        assertThat(ia[3]).isEqualTo(4);
    }

    @Test
    public void test_convertTensorType_p() {
        assertThat(Helper.convertTensorType(TensorInfo.Type.FLOAT32)).isEqualTo(0);
        assertThat(Helper.convertTensorType(TensorInfo.Type.INT32)).isEqualTo(1);
        assertThat(Helper.convertTensorType(TensorInfo.Type.QUANT8_ASYMM)).isEqualTo(2);
        assertThat(Helper.convertTensorType(TensorInfo.Type.BOOL)).isEqualTo(3);
        assertThat(Helper.convertTensorType(TensorInfo.Type.UINT8)).isEqualTo(4);
    }

    @Test
    public void test_convertOneRTTensorType_p() {
        assertThat(Helper.convertOneRTTensorType(0)).isEqualTo(TensorInfo.Type.FLOAT32);
        assertThat(Helper.convertOneRTTensorType(1)).isEqualTo(TensorInfo.Type.INT32);
        assertThat(Helper.convertOneRTTensorType(2)).isEqualTo(TensorInfo.Type.QUANT8_ASYMM);
        assertThat(Helper.convertOneRTTensorType(3)).isEqualTo(TensorInfo.Type.BOOL);
        assertThat(Helper.convertOneRTTensorType(4)).isEqualTo(TensorInfo.Type.UINT8);
    }
}