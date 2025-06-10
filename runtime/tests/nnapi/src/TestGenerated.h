/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef ANDROID_FRAMEWORK_ML_NN_RUNTIME_TEST_TESTGENERATED_H
#define ANDROID_FRAMEWORK_ML_NN_RUNTIME_TEST_TESTGENERATED_H

#include <gtest/gtest.h>

// Fix for NNFW: comment out include TestCompliance.h
//#include "TestCompliance.h"
#include "TestHarness.h"
#include "TestNeuralNetworksWrapper.h"

#ifdef NNTEST_CTS
#define NNTEST_COMPUTE_MODE
#endif

#ifdef NNTEST_COMPUTE_MODE
#define GENERATED_TESTS_BASE testing::TestWithParam<Execution::ComputeMode>
#undef TEST_F
#define TEST_F TEST_P
// Only generated tests include the TestGenerated.h header file, so only those
// tests will be affected by changing their TEST_F to TEST_P.  If we
// accidentally change TEST_F to TEST_P in some other context, we will get a
// compile-time failure, because TEST_F requires a non-value-parameterized
// fixture class whereas TEST_P requires a value-parameterized fixture class.
//
// Example failure:
//
// clang-format off
//     gtest-param-util.h:488:41: error: no type named 'ParamType' in '(anonymous namespace)::MemoryTest'
//       using ParamType = typename TestSuite::ParamType;
//                         ~~~~~~~~~~~~~~~~~~~~^~~~~~~~~
//     TestMemory.cpp:43:1: note: in instantiation of template class 'testing::internal::ParameterizedTestSuiteInfo<(anonymous namespace)::MemoryTest>' requested here
//     TEST_P(MemoryTest, TestFd) {
//     ^
//     gtest-param-test.h:428:11: note: expanded from macro 'TEST_P'
// clang-format on
#else
#define GENERATED_TESTS_BASE ::testing::Test
#endif

using namespace nnfw::rt::test_wrapper;
using namespace test_helper;

namespace generated_tests {

class GeneratedTests : public GENERATED_TESTS_BASE {
   protected:
    virtual void SetUp() override;
    virtual void TearDown() override;

    Compilation compileModel(const Model* model);
    void executeWithCompilation(const Model* model, Compilation* compilation,
                                std::function<bool(int)> isIgnored,
                                std::vector<MixedTypedExample>& examples, std::string dumpFile);
    void executeOnce(const Model* model, std::function<bool(int)> isIgnored,
                     std::vector<MixedTypedExample>& examples, std::string dumpFile);
    void executeMultithreadedOwnCompilation(const Model* model, std::function<bool(int)> isIgnored,
                                            std::vector<MixedTypedExample>& examples);
    void executeMultithreadedSharedCompilation(const Model* model,
                                               std::function<bool(int)> isIgnored,
                                               std::vector<MixedTypedExample>& examples);
    // Test driver for those generated from ml/nn/runtime/test/spec
    void execute(std::function<void(Model*)> createModel, std::function<bool(int)> isIgnored,
                 std::vector<MixedTypedExample>& examples, std::string dumpFile = "");

    std::string mCacheDir;
    std::vector<uint8_t> mToken;
    bool mTestCompilationCaching;
#ifdef NNTEST_COMPUTE_MODE
    // SetUp() uses Execution::setComputeMode() to establish a new ComputeMode,
    // and saves off the previous ComputeMode here; TearDown() restores that
    // previous ComputeMode, so that subsequent tests will not be affected by
    // the SetUp() ComputeMode change.
    Execution::ComputeMode mOldComputeMode;
#endif
};

// Tag for the dynamic output shape tests
class DynamicOutputShapeTest : public GeneratedTests {};

}  // namespace generated_tests

using namespace generated_tests;

#endif  // ANDROID_FRAMEWORK_ML_NN_RUNTIME_TEST_TESTGENERATED_H
