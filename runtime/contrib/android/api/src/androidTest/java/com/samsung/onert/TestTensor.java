package com.samsung.onert;

import androidx.test.filters.SmallTest;
import androidx.test.runner.AndroidJUnit4;
import org.junit.Before;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import static com.google.common.truth.Truth.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.stream.Collectors;

@RunWith(AndroidJUnit4.class)
@SmallTest
public class TestTensor {
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    @Test
    public void test_Tensor_p() {
        TensorInfo info = TestCommon.newTensorInfo();
        Tensor t = new Tensor(info);
        assertThat(t).isNotNull();
    }

    @Test
    public void test_Tensor_p2() {
        TensorInfo.Type type = TensorInfo.Type.FLOAT32;
        int[] shape = new int[]{2,2};
        Tensor t = new Tensor(shape, type);
        assertThat(t).isNotNull();
    }

    @Test
    public void test_shape_p() {
        TensorInfo info = TestCommon.newTensorInfo();
        Tensor t = new Tensor(info);
        assertThat(t.shape()).asList().
            containsExactlyElementsIn(Arrays.stream(info.shape).boxed().collect(Collectors.toList())).
            inOrder();
    }

    @Test
    public void test_type_p() {
        TensorInfo info = TestCommon.newTensorInfo();
        Tensor t = new Tensor(info);
        assertThat(t.type()).isEqualTo(info.type);
    }

    @Test
    public void test_buffer_p() {
        TensorInfo info = TestCommon.newTensorInfo();
        Tensor t = new Tensor(info);
        assertThat(t.buffer().capacity()).isEqualTo(TensorInfo.getByteSize(info));
    }

    @Test
    public void test_buffer_p2() {
        TensorInfo info = TestCommon.newTensorInfo();
        Tensor t = new Tensor(info);
        {
            ByteBuffer bb = ByteBuffer.allocateDirect(16).
                                order(ByteOrder.nativeOrder());
            bb.putInt(11);
            bb.putInt(22);
            bb.putInt(33);
            bb.putInt(44);
            t.buffer(bb);
        }
        ByteBuffer bb = t.buffer();
        bb.rewind();
        assertThat(bb.getInt()).isEqualTo(11);
        assertThat(bb.getInt()).isEqualTo(22);
        assertThat(bb.getInt()).isEqualTo(33);
        assertThat(bb.getInt()).isEqualTo(44);
    }

    @Test
    public void test_getByteSize_p() {
        TensorInfo info = TestCommon.newTensorInfo();
        Tensor t = new Tensor(info);
        assertThat(t.getByteSize()).isEqualTo(
            Arrays.stream(info.shape).reduce(TensorInfo.getTypeSize(info.type), (x, y) -> { return x * y; }));
    }

    @Test
    public void test_getByteSize_p2() {
        TensorInfo info = TestCommon.newTensorInfo2();
        Tensor t = new Tensor(info);
        assertThat(t.getByteSize()).isEqualTo(
            Arrays.stream(info.shape).reduce(TensorInfo.getTypeSize(info.type), (x, y) -> { return x * y; }));
    }

    @Test
    public void test_getSize_p() {
        TensorInfo info = TestCommon.newTensorInfo();
        Tensor t = new Tensor(info);
        assertThat(t.getSize()).isEqualTo(
            Arrays.stream(info.shape).reduce(1, (x, y) -> { return x * y; }));
    }

    @Test
    public void test_getSize_p2() {
        TensorInfo info = TestCommon.newTensorInfo2();
        Tensor t = new Tensor(info);
        assertThat(t.getSize()).isEqualTo(
            Arrays.stream(info.shape).reduce(1, (x, y) -> { return x * y; }));
    }

    @Test
    public void test_close_p() {
        Tensor t = new Tensor(TestCommon.newTensorInfo());
        t.close();
        assertThat(t.buffer()).isNull();
    }

    @Test
    public void test_validateBuffer_p() {
        Tensor t = new Tensor(TestCommon.newTensorInfo());
        assertThat(Tensor.validateBuffer(t.buffer())).isTrue();
    }

    @Test
    public void test_validateBuffer_n() {
        Tensor t = new Tensor(TestCommon.newTensorInfo());
        t.close();
        assertThat(Tensor.validateBuffer(t.buffer())).isFalse();
    }

    @Test
    public void test_validateBuffer_n2() {
        Tensor t = new Tensor(TestCommon.newTensorInfo());
        ByteBuffer bb = ByteBuffer.allocate(16);
        t.buffer(bb);
        assertThat(Tensor.validateBuffer(t.buffer())).isFalse();
    }
}