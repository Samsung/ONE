package com.samsung.onert;

import androidx.test.filters.SmallTest;
import androidx.test.runner.AndroidJUnit4;
import org.junit.Before;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import static com.google.common.truth.Truth.*;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@RunWith(AndroidJUnit4.class)
@SmallTest
public class TestTensorInfo {
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    @Test
    public void test_TensorInfo_p() {
        TensorInfo info = new TensorInfo();
        assertThat(info).isNotNull();
    }

    @Test
    public void test_TensorInfo_p2() {
        TensorInfo.Type type = TensorInfo.Type.FLOAT32;
        int[] shape = new int[]{2,2};
        int rank = shape.length;
        TensorInfo info = new TensorInfo(type, rank, shape);
        assertThat(info).isNotNull();
        assertThat(info.type).isEqualTo(type);
        assertThat(info.rank).isEqualTo(rank);
        assertThat(info.shape).asList().
            containsExactlyElementsIn(Arrays.stream(shape).boxed().collect(Collectors.toList())).
            inOrder();
    }

    @Test
    public void test_close_p() {
        TensorInfo info = new TensorInfo();
        info.close();
        assertThat(info.shape).isNull();
    }

    @Test
    public void test_getTypeSize_p() {
        assertThat(TensorInfo.getTypeSize(TensorInfo.Type.FLOAT32)).isEqualTo(4);
        assertThat(TensorInfo.getTypeSize(TensorInfo.Type.INT32)).isEqualTo(4);
        assertThat(TensorInfo.getTypeSize(TensorInfo.Type.QUANT8_ASYMM)).isEqualTo(1);
        assertThat(TensorInfo.getTypeSize(TensorInfo.Type.BOOL)).isEqualTo(1);
        assertThat(TensorInfo.getTypeSize(TensorInfo.Type.UINT8)).isEqualTo(1);
    }

    @Test
    public void test_getByteSize_p() {
        TensorInfo.Type type = TensorInfo.Type.FLOAT32;
        int[] shape = new int[]{2,2};
        int rank = shape.length;
        TensorInfo info = new TensorInfo(type, rank, shape);
        assertThat(TensorInfo.getByteSize(info)).isEqualTo(
            Arrays.stream(shape).reduce(TensorInfo.getTypeSize(type), (x, y) -> { return x * y; }));
    }

    @Test
    public void test_getByteSize_p2() {
        TensorInfo.Type type = TensorInfo.Type.UINT8;
        int[] shape = new int[]{1024, 128, 3};
        int rank = shape.length;
        TensorInfo info = new TensorInfo(type, rank, shape);
        assertThat(TensorInfo.getByteSize(info)).isEqualTo(
            Arrays.stream(shape).reduce(TensorInfo.getTypeSize(type), (x, y) -> { return x * y; }));
    }

    @Test
    public void test_getSize_p() {
        TensorInfo.Type type = TensorInfo.Type.FLOAT32;
        int[] shape = new int[]{2,2};
        int rank = shape.length;
        TensorInfo info = new TensorInfo(type, rank, shape);
        assertThat(TensorInfo.getSize(info)).isEqualTo(
            Arrays.stream(shape).reduce(1, (x, y) -> { return x * y; }));
    }

    @Test
    public void test_getSize_p2() {
        TensorInfo.Type type = TensorInfo.Type.UINT8;
        int[] shape = new int[]{1024, 128, 3};
        int rank = shape.length;
        TensorInfo info = new TensorInfo(type, rank, shape);
        assertThat(TensorInfo.getSize(info)).isEqualTo(
            Arrays.stream(shape).reduce(1, (x, y) -> { return x * y; }));
    }
}