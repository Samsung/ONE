/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "NeuralNetworksWrapper.h"
#if 0 // TODO-NNRT : Enable if we support OEM
#include "NeuralNetworksOEM.h"
#endif
#include <gtest/gtest.h>
#include <string>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
// Note: onert is allow to set activation operand constant only,
//       so we change test to set operand #2 to constant. (ANEURALNETWORKS_FUSED_NONE)
//       And model's input is changed: [0, 1, 2] -> [0, 1]
// This file tests all the validations done by the Neural Networks API.
namespace {

#ifndef PATH_MAX
#define PATH_MAX 256
#endif

static int shmem_num = 0;
static int shmem_create_region(size_t size)
{
    char temp[PATH_MAX];
#ifndef __ANDROID__
    snprintf(temp, sizeof(temp), "/tmp/nn-shmem-%d-%d-XXXXXXXXX", getpid(), shmem_num++);
#else
    snprintf(temp, sizeof(temp), "/data/local/tmp/nn-shmem-%d-%d-XXXXXXXXX", getpid(), shmem_num++);
#endif

    // Set umask and recover after generate temporary file to avoid security issue
    mode_t umaskPrev = umask(S_IRUSR|S_IWUSR);
    int fd = mkstemp(temp);
    umask(umaskPrev);

    if (fd == -1) return -1;

    unlink(temp);

    if (TEMP_FAILURE_RETRY(ftruncate(fd, size)) == -1) {
      close(fd);
      return -1;
    }

    return fd;
}

class ValidationTest : public ::testing::Test {
protected:
    virtual void SetUp() {}
};

class ValidationTestModel : public ValidationTest {
protected:
    virtual void SetUp() {
        ValidationTest::SetUp();
        ASSERT_EQ(ANeuralNetworksModel_create(&mModel), ANEURALNETWORKS_NO_ERROR);
    }
    virtual void TearDown() {
        ANeuralNetworksModel_free(mModel);
        ValidationTest::TearDown();
    }
    ANeuralNetworksModel* mModel = nullptr;
};

class ValidationTestIdentify : public ValidationTestModel {
    virtual void SetUp() {
        ValidationTestModel::SetUp();

        uint32_t dimensions[]{1};
        ANeuralNetworksOperandType tensorType{.type = ANEURALNETWORKS_TENSOR_FLOAT32,
                                              .dimensionCount = 1,
                                              .dimensions = dimensions};
        ANeuralNetworksOperandType scalarType{.type = ANEURALNETWORKS_INT32,
                                              .dimensionCount = 0,
                                              .dimensions = nullptr};
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &tensorType), ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &tensorType), ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &scalarType), ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &tensorType), ANEURALNETWORKS_NO_ERROR);
        // onert is allow to set activation operand constant only
        int32_t act = ANEURALNETWORKS_FUSED_NONE;
        ASSERT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 2, &act, sizeof(act)), ANEURALNETWORKS_NO_ERROR);
        uint32_t inList[3]{0, 1, 2};
        uint32_t outList[1]{3};
        ASSERT_EQ(ANeuralNetworksModel_addOperation(mModel, ANEURALNETWORKS_ADD, 3, inList, 1,
                                                    outList),
                  ANEURALNETWORKS_NO_ERROR);
    }
    virtual void TearDown() {
        ValidationTestModel::TearDown();
    }
};

class ValidationTestCompilation : public ValidationTestModel {
protected:
    virtual void SetUp() {
        ValidationTestModel::SetUp();

        uint32_t dimensions[]{1};
        ANeuralNetworksOperandType tensorType{.type = ANEURALNETWORKS_TENSOR_FLOAT32,
                                              .dimensionCount = 1,
                                              .dimensions = dimensions};
        ANeuralNetworksOperandType scalarType{.type = ANEURALNETWORKS_INT32,
                                              .dimensionCount = 0,
                                              .dimensions = nullptr};

        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &tensorType), ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &tensorType), ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &scalarType), ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_addOperand(mModel, &tensorType), ANEURALNETWORKS_NO_ERROR);
        // onert is allow to set activation operand constant only
        int32_t act = ANEURALNETWORKS_FUSED_NONE;
        ASSERT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 2, &act, sizeof(act)), ANEURALNETWORKS_NO_ERROR);
        uint32_t inList[3]{0, 1, 2};
        uint32_t outList[1]{3};
        ASSERT_EQ(ANeuralNetworksModel_addOperation(mModel, ANEURALNETWORKS_ADD, 3, inList, 1,
                                                    outList),
                  ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(mModel, 2, inList, 1, outList),
                  ANEURALNETWORKS_NO_ERROR);
        ASSERT_EQ(ANeuralNetworksModel_finish(mModel), ANEURALNETWORKS_NO_ERROR);

        ASSERT_EQ(ANeuralNetworksCompilation_create(mModel, &mCompilation),
                  ANEURALNETWORKS_NO_ERROR);
    }
    virtual void TearDown() {
        ANeuralNetworksCompilation_free(mCompilation);
        ValidationTestModel::TearDown();
    }
    ANeuralNetworksCompilation* mCompilation = nullptr;
};

class ValidationTestExecution : public ValidationTestCompilation {
protected:
    virtual void SetUp() {
        ValidationTestCompilation::SetUp();

        ASSERT_EQ(ANeuralNetworksCompilation_finish(mCompilation), ANEURALNETWORKS_NO_ERROR);

        ASSERT_EQ(ANeuralNetworksExecution_create(mCompilation, &mExecution),
                  ANEURALNETWORKS_NO_ERROR);
    }
    virtual void TearDown() {
        ANeuralNetworksExecution_free(mExecution);
        ValidationTestCompilation::TearDown();
    }
    ANeuralNetworksExecution* mExecution = nullptr;
};

TEST_F(ValidationTest, CreateModel) {
    EXPECT_EQ(ANeuralNetworksModel_create(nullptr), ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestModel, AddOperand) {
    ANeuralNetworksOperandType floatType{
                .type = ANEURALNETWORKS_FLOAT32, .dimensionCount = 0, .dimensions = nullptr};
    EXPECT_EQ(ANeuralNetworksModel_addOperand(nullptr, &floatType),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, nullptr), ANEURALNETWORKS_UNEXPECTED_NULL);

    ANeuralNetworksOperandType quant8TypeInvalidScale{
                .type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                .dimensionCount = 0,
                .dimensions = nullptr,
                // Scale has to be non-negative
                .scale = -1.0f,
                .zeroPoint = 0,
              };
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &quant8TypeInvalidScale),
              ANEURALNETWORKS_BAD_DATA);

    ANeuralNetworksOperandType quant8TypeInvalidZeroPoint{
                .type = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
                .dimensionCount = 0,
                .dimensions = nullptr,
                .scale = 1.0f,
                // zeroPoint has to be in [0, 255]
                .zeroPoint = -1,
              };
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &quant8TypeInvalidZeroPoint),
              ANEURALNETWORKS_BAD_DATA);

    uint32_t dim = 2;
    ANeuralNetworksOperandType invalidScalarType{
                .type = ANEURALNETWORKS_INT32,
                // scalar types can only 0 dimensions.
                .dimensionCount = 1,
                .dimensions = &dim,
              };
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &invalidScalarType),
              ANEURALNETWORKS_BAD_DATA);

    ANeuralNetworksModel_finish(mModel);
    // This should fail, as the model is already finished.
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &floatType),
              ANEURALNETWORKS_BAD_STATE);
}

TEST_F(ValidationTestModel, SetOptionalOperand) {
    ANeuralNetworksOperandType floatType{
                .type = ANEURALNETWORKS_FLOAT32, .dimensionCount = 0, .dimensions = nullptr};
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &floatType), ANEURALNETWORKS_NO_ERROR);

    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 0, nullptr, 0),
              ANEURALNETWORKS_NO_ERROR);
}

TEST_F(ValidationTestModel, SetOperandValue) {
    ANeuralNetworksOperandType floatType{
                .type = ANEURALNETWORKS_FLOAT32, .dimensionCount = 0, .dimensions = nullptr};
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &floatType), ANEURALNETWORKS_NO_ERROR);

    char buffer[20];
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(nullptr, 0, buffer, sizeof(buffer)),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 0, nullptr, sizeof(buffer)),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    // This should fail, since buffer is not the size of a float32.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 0, buffer, sizeof(buffer)),
              ANEURALNETWORKS_BAD_DATA);

    // This should succeed.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 0, buffer, sizeof(float)),
              ANEURALNETWORKS_NO_ERROR);

    // This should fail, as this operand does not exist.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 1, buffer, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    ANeuralNetworksModel_finish(mModel);
    // This should fail, as the model is already finished.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 0, buffer, sizeof(float)),
              ANEURALNETWORKS_BAD_STATE);
}

TEST_F(ValidationTestModel, SetOperandValueFromMemory) {
    uint32_t dimensions[]{1};
    ANeuralNetworksOperandType floatType{
                .type = ANEURALNETWORKS_TENSOR_FLOAT32,
                .dimensionCount = 1,
                .dimensions = dimensions};
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &floatType), ANEURALNETWORKS_NO_ERROR);

    const size_t memorySize = 20;
    int memoryFd = shmem_create_region(memorySize);
    ASSERT_GT(memoryFd, 0);

    ANeuralNetworksMemory* memory;
    EXPECT_EQ(ANeuralNetworksMemory_createFromFd(memorySize, PROT_READ | PROT_WRITE,
                                                 memoryFd, 0, &memory),
              ANEURALNETWORKS_NO_ERROR);

    EXPECT_EQ(ANeuralNetworksModel_setOperandValueFromMemory(nullptr, 0,
                                                             memory, 0, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_setOperandValueFromMemory(mModel, 0,
                                                             nullptr, 0, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    // This should fail, since the operand does not exist.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValueFromMemory(mModel, -1,
                                                             memory, 0, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since memory is not the size of a float32.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValueFromMemory(mModel, 0,
                                                             memory, 0, memorySize),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, as this operand does not exist.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValueFromMemory(mModel, 1,
                                                             memory, 0, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since offset is larger than memorySize.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValueFromMemory(mModel, 0,
                                                             memory, memorySize + 1,
                                                             sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since requested size is larger than the memory.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValueFromMemory(mModel, 0,
                                                             memory, memorySize - 3,
                                                             sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    ANeuralNetworksModel_finish(mModel);
    // This should fail, as the model is already finished.
    EXPECT_EQ(ANeuralNetworksModel_setOperandValueFromMemory(mModel, 0,
                                                             memory, 0,
                                                             sizeof(float)),
              ANEURALNETWORKS_BAD_STATE);
}


#if 0 // TODO-NNRT : Enable if we support OEM OP.
TEST_F(ValidationTestModel, AddOEMOperand) {
    ANeuralNetworksOperandType OEMScalarType{
                .type = ANEURALNETWORKS_OEM_SCALAR, .dimensionCount = 0, .dimensions = nullptr};
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &OEMScalarType), ANEURALNETWORKS_NO_ERROR);
    char buffer[20];
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 0, buffer, sizeof(buffer)),
              ANEURALNETWORKS_NO_ERROR);

    const size_t kByteSizeOfOEMTensor = 4;
    uint32_t dimensions[]{kByteSizeOfOEMTensor};
    ANeuralNetworksOperandType OEMTensorType{
                .type = ANEURALNETWORKS_TENSOR_OEM_BYTE,
                .dimensionCount = 1,
                .dimensions = dimensions};
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &OEMTensorType), ANEURALNETWORKS_NO_ERROR);
    EXPECT_EQ(ANeuralNetworksModel_setOperandValue(mModel, 1, buffer, kByteSizeOfOEMTensor),
              ANEURALNETWORKS_NO_ERROR);

    ANeuralNetworksModel_finish(mModel);
    // This should fail, as the model is already finished.
    EXPECT_EQ(ANeuralNetworksModel_addOperand(mModel, &OEMTensorType), ANEURALNETWORKS_BAD_STATE);
}
#endif // TODO-NNRT

TEST_F(ValidationTestModel, AddOperation) {
    uint32_t input = 0;
    uint32_t output = 0;
    EXPECT_EQ(ANeuralNetworksModel_addOperation(nullptr, ANEURALNETWORKS_AVERAGE_POOL_2D, 1, &input,
                                                1, &output),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_addOperation(mModel, ANEURALNETWORKS_AVERAGE_POOL_2D, 0, nullptr,
                                                1, &output),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_addOperation(mModel, ANEURALNETWORKS_AVERAGE_POOL_2D, 1, &input,
                                                0, nullptr),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    ANeuralNetworksOperationType invalidOp = -1;
    EXPECT_EQ(ANeuralNetworksModel_addOperation(mModel, invalidOp, 1, &input,
                                                1, &output),
              ANEURALNETWORKS_BAD_DATA);

    ANeuralNetworksModel_finish(mModel);
    // This should fail, as the model is already finished.
    EXPECT_EQ(ANeuralNetworksModel_addOperation(mModel, ANEURALNETWORKS_AVERAGE_POOL_2D, 1, &input,
                                                1, &output),
              ANEURALNETWORKS_BAD_STATE);
}

TEST_F(ValidationTestModel, IdentifyInputsAndOutputs) {
    uint32_t input = 0;
    uint32_t output = 0;
    EXPECT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(nullptr, 1, &input, 1, &output),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(mModel, 0, nullptr, 1, &output),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(mModel, 1, &input, 0, nullptr),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    ANeuralNetworksModel_finish(mModel);
    // This should fail, as the model is already finished.
    EXPECT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(mModel, 1, &input, 1, &output),
              ANEURALNETWORKS_BAD_STATE);
}

TEST_F(ValidationTestModel, RelaxComputationFloat32toFloat16) {
    EXPECT_EQ(ANeuralNetworksModel_relaxComputationFloat32toFloat16(nullptr, true),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    ANeuralNetworksModel_finish(mModel);
    // This should fail, as the model is already finished.
    EXPECT_EQ(ANeuralNetworksModel_relaxComputationFloat32toFloat16(mModel, true),
              ANEURALNETWORKS_BAD_STATE);
    EXPECT_EQ(ANeuralNetworksModel_relaxComputationFloat32toFloat16(mModel, false),
              ANEURALNETWORKS_BAD_STATE);
}

TEST_F(ValidationTestModel, Finish) {
    EXPECT_EQ(ANeuralNetworksModel_finish(nullptr), ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksModel_finish(mModel), ANEURALNETWORKS_NO_ERROR);
    EXPECT_EQ(ANeuralNetworksModel_finish(mModel), ANEURALNETWORKS_BAD_STATE);
}

TEST_F(ValidationTestModel, CreateCompilation) {
    ANeuralNetworksCompilation* compilation = nullptr;
    EXPECT_EQ(ANeuralNetworksCompilation_create(nullptr, &compilation),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksCompilation_create(mModel, nullptr), ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksCompilation_create(mModel, &compilation), ANEURALNETWORKS_BAD_STATE);
}

TEST_F(ValidationTestIdentify, Ok) {
    uint32_t inList[2]{0, 1};
    uint32_t outList[1]{3};

    ASSERT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(mModel, 2, inList, 1, outList),
              ANEURALNETWORKS_NO_ERROR);

    ASSERT_EQ(ANeuralNetworksModel_finish(mModel), ANEURALNETWORKS_NO_ERROR);
}

TEST_F(ValidationTestIdentify, InputIsOutput) {
    uint32_t inList[2]{0, 1};
    uint32_t outList[2]{3, 0};

    ASSERT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(mModel, 2, inList, 2, outList),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestIdentify, OutputIsInput) {
    uint32_t inList[3]{0, 1, 3};
    uint32_t outList[1]{3};

    ASSERT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(mModel, 3, inList, 1, outList),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestIdentify, DuplicateInputs) {
    uint32_t inList[3]{0, 1, 0};
    uint32_t outList[1]{3};

    ASSERT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(mModel, 3, inList, 1, outList),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestIdentify, DuplicateOutputs) {
    uint32_t inList[2]{0, 1};
    uint32_t outList[2]{3, 3};

    ASSERT_EQ(ANeuralNetworksModel_identifyInputsAndOutputs(mModel, 2, inList, 2, outList),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestCompilation, SetPreference) {
    EXPECT_EQ(ANeuralNetworksCompilation_setPreference(nullptr, ANEURALNETWORKS_PREFER_LOW_POWER),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    EXPECT_EQ(ANeuralNetworksCompilation_setPreference(mCompilation, 40), ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestCompilation, CreateExecution) {
    ANeuralNetworksExecution* execution = nullptr;
    EXPECT_EQ(ANeuralNetworksExecution_create(nullptr, &execution),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksExecution_create(mCompilation, nullptr),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksExecution_create(mCompilation, &execution),
              ANEURALNETWORKS_BAD_STATE);
}

TEST_F(ValidationTestCompilation, Finish) {
    EXPECT_EQ(ANeuralNetworksCompilation_finish(nullptr), ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksCompilation_finish(mCompilation), ANEURALNETWORKS_NO_ERROR);
    EXPECT_EQ(ANeuralNetworksCompilation_setPreference(mCompilation,
                                                       ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER),
              ANEURALNETWORKS_BAD_STATE);
    EXPECT_EQ(ANeuralNetworksCompilation_finish(mCompilation), ANEURALNETWORKS_BAD_STATE);
}

TEST_F(ValidationTestExecution, SetInput) {
    ANeuralNetworksExecution* execution;
    EXPECT_EQ(ANeuralNetworksExecution_create(mCompilation, &execution), ANEURALNETWORKS_NO_ERROR);

    char buffer[20];
    EXPECT_EQ(ANeuralNetworksExecution_setInput(nullptr, 0, nullptr, buffer, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksExecution_setInput(execution, 0, nullptr, nullptr, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    // This should fail, since memory is not the size of a float32.
    EXPECT_EQ(ANeuralNetworksExecution_setInput(execution, 0, nullptr, buffer, 20),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, as this operand does not exist.
    EXPECT_EQ(ANeuralNetworksExecution_setInput(execution, 999, nullptr, buffer, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, as this operand does not exist.
    EXPECT_EQ(ANeuralNetworksExecution_setInput(execution, -1, nullptr, buffer, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestExecution, SetOutput) {
    ANeuralNetworksExecution* execution;
    EXPECT_EQ(ANeuralNetworksExecution_create(mCompilation, &execution), ANEURALNETWORKS_NO_ERROR);

    char buffer[20];
    EXPECT_EQ(ANeuralNetworksExecution_setOutput(nullptr, 0, nullptr, buffer, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksExecution_setOutput(execution, 0, nullptr, nullptr, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    // This should fail, since memory is not the size of a float32.
    EXPECT_EQ(ANeuralNetworksExecution_setOutput(execution, 0, nullptr, buffer, 20),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, as this operand does not exist.
    EXPECT_EQ(ANeuralNetworksExecution_setOutput(execution, 999, nullptr, buffer, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, as this operand does not exist.
    EXPECT_EQ(ANeuralNetworksExecution_setOutput(execution, -1, nullptr, buffer, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestExecution, SetInputFromMemory) {
    ANeuralNetworksExecution* execution;
    EXPECT_EQ(ANeuralNetworksExecution_create(mCompilation, &execution), ANEURALNETWORKS_NO_ERROR);

    const size_t memorySize = 20;
    int memoryFd = shmem_create_region(memorySize);
    ASSERT_GT(memoryFd, 0);

    ANeuralNetworksMemory* memory;
    EXPECT_EQ(ANeuralNetworksMemory_createFromFd(memorySize, PROT_READ | PROT_WRITE,
                                                 memoryFd, 0, &memory),
              ANEURALNETWORKS_NO_ERROR);

    EXPECT_EQ(ANeuralNetworksExecution_setInputFromMemory(nullptr, 0, nullptr,
                                                          memory, 0, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksExecution_setInputFromMemory(execution, 0, nullptr,
                                                          nullptr, 0, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    // This should fail, since the operand does not exist.
    EXPECT_EQ(ANeuralNetworksExecution_setInputFromMemory(execution, 999, nullptr,
                                                          memory, 0, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since the operand does not exist.
    EXPECT_EQ(ANeuralNetworksExecution_setInputFromMemory(execution, -1, nullptr,
                                                          memory, 0, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since memory is not the size of a float32.
    EXPECT_EQ(ANeuralNetworksExecution_setInputFromMemory(execution, 0, nullptr,
                                                          memory, 0, memorySize),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since offset is larger than memorySize.
    EXPECT_EQ(ANeuralNetworksExecution_setInputFromMemory(execution, 0, nullptr,
                                                          memory, memorySize + 1, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since requested size is larger than the memory.
    EXPECT_EQ(ANeuralNetworksExecution_setInputFromMemory(execution, 0, nullptr,
                                                          memory, memorySize - 3, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestExecution, SetOutputFromMemory) {
    ANeuralNetworksExecution* execution;
    EXPECT_EQ(ANeuralNetworksExecution_create(mCompilation, &execution), ANEURALNETWORKS_NO_ERROR);

    const size_t memorySize = 20;
    int memoryFd = shmem_create_region(memorySize);
    ASSERT_GT(memoryFd, 0);

    ANeuralNetworksMemory* memory;
    EXPECT_EQ(ANeuralNetworksMemory_createFromFd(memorySize, PROT_READ | PROT_WRITE,
                                                 memoryFd, 0, &memory),
              ANEURALNETWORKS_NO_ERROR);

    EXPECT_EQ(ANeuralNetworksExecution_setOutputFromMemory(nullptr, 0, nullptr,
                                                           memory, 0, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksExecution_setOutputFromMemory(execution, 0, nullptr,
                                                           nullptr, 0, sizeof(float)),
              ANEURALNETWORKS_UNEXPECTED_NULL);

    // This should fail, since the operand does not exist.
    EXPECT_EQ(ANeuralNetworksExecution_setOutputFromMemory(execution, 999, nullptr,
                                                           memory, 0, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since the operand does not exist.
    EXPECT_EQ(ANeuralNetworksExecution_setOutputFromMemory(execution, -1, nullptr,
                                                           memory, 0, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since memory is not the size of a float32.
    EXPECT_EQ(ANeuralNetworksExecution_setOutputFromMemory(execution, 0, nullptr,
                                                           memory, 0, memorySize),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since offset is larger than memorySize.
    EXPECT_EQ(ANeuralNetworksExecution_setOutputFromMemory(execution, 0, nullptr,
                                                           memory, memorySize + 1, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);

    // This should fail, since requested size is larger than the memory.
    EXPECT_EQ(ANeuralNetworksExecution_setOutputFromMemory(execution, 0, nullptr,
                                                           memory, memorySize - 3, sizeof(float)),
              ANEURALNETWORKS_BAD_DATA);
}

TEST_F(ValidationTestExecution, StartCompute) {
    ANeuralNetworksExecution* execution;
    EXPECT_EQ(ANeuralNetworksExecution_create(mCompilation, &execution), ANEURALNETWORKS_NO_ERROR);

    ANeuralNetworksEvent* event;
    EXPECT_EQ(ANeuralNetworksExecution_startCompute(nullptr, &event),
              ANEURALNETWORKS_UNEXPECTED_NULL);
    EXPECT_EQ(ANeuralNetworksExecution_startCompute(execution, nullptr),
              ANEURALNETWORKS_UNEXPECTED_NULL);
}

TEST_F(ValidationTestExecution, EventWait) {
    EXPECT_EQ(ANeuralNetworksEvent_wait(nullptr), ANEURALNETWORKS_UNEXPECTED_NULL);
}
}  // namespace
