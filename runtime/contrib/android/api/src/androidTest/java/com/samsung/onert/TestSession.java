package com.samsung.onert;

import androidx.test.filters.SmallTest;
import androidx.test.rule.GrantPermissionRule;
import androidx.test.runner.AndroidJUnit4;
import org.junit.Before;
import org.junit.After;
import org.junit.Test;
import org.junit.Rule;
import org.junit.runner.RunWith;
import static com.google.common.truth.Truth.*;

import java.util.Arrays;
import java.util.stream.Collectors;

import android.Manifest;
import android.os.Environment;

@RunWith(AndroidJUnit4.class)
@SmallTest
public class TestSession {
    @Rule
    public GrantPermissionRule permissionRule = grantPermissions();

    @Before
    public void setUp() {
        assertThat(TestCommon.exist(TestCommon.getNnpkgPath())).isTrue();
    }
    
    @After
    public void tearDown() {
    }

    @Test
    public void test_Session_p() {
        Session session = new Session(TestCommon.getNnpkgPath(), "cpu");
        assertThat(session).isNotNull();
    }

    @Test
    public void test_prepare_p() {
        Session session = newSession();
        assertThat(session.prepare()).isTrue();
    }

    @Test
    public void test_setInputs_p() {
        //Session session = newSession();
        //assertThat(session.prepare()).isTrue();
        //Tensor[] inputs = newTestInputTensor();
        //session.setInputs(inputs);
        //int input_size = session.getInputSize();
        //assertThat(input_size).isEqualTo(inputs.length);
        //for (int i = 0; i < input_size; ++i) {
        //    TensorInfo info = session.getInputTensorInfo(i);
        //    assertThat(inputs[i].shape()).asList().
        //        containsExactlyElementsIn(Arrays.stream(info.shape).boxed().collect(Collectors.toList())).
        //        inOrder();
        //}
    }

    @Test
    public void test_setOutputs_p() {
    }

    @Test
    public void test_run_p() {
    }

    @Test
    public void test_close_p() {
    }

    @Test
    public void test_getInputSize_p() {
    }

    @Test
    public void test_getOutputSize_p() {
    }

    @Test
    public void test_getInputTensorInfo_p() {
    }

    @Test
    public void test_getOutputTensorInfo_p() {
    }

    private Session newSession() {
        Session session = new Session(TestCommon.getNnpkgPath(), "cpu");
        return session;
    }

    private Tensor[] newTestInputTensor() {
        TensorInfo.Type type = TensorInfo.Type.FLOAT32;
        int[] shape = new int[]{1,299,299,3};
        int rank = shape.length;
        Tensor t = new Tensor(new TensorInfo(type, rank, shape));
        return new Tensor[]{t};
    }

    private Tensor[] newTestOutputTensor() {
        TensorInfo.Type type = TensorInfo.Type.FLOAT32;
        int[] shape = new int[]{1,149,149,3};
        int rank = shape.length;
        Tensor t = new Tensor(new TensorInfo(type, rank, shape));
        return new Tensor[]{t};
    }

    private GrantPermissionRule grantPermissions() {
        return GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE);
    }
}