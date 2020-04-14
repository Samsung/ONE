/*
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

/* Header-only library for various helpers of test harness
 * See frameworks/ml/nn/runtime/test/TestGenerated.cpp for how this is used.
 */
#ifndef ANDROID_ML_NN_TOOLS_TEST_GENERATOR_TEST_HARNESS_H
#define ANDROID_ML_NN_TOOLS_TEST_GENERATOR_TEST_HARNESS_H

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <map>
#include <tuple>
#include <vector>

// Fix for onert: define _Float16 for gnu compiler
#if  __GNUC__ && !__clang__
#if __ARM_FP16_FORMAT_IEEE || __ARM_FP16_FORMAT_ALTERNATIVE
#define _Float16 __fp16
#else // __ARM_FP16_FORMAT_IEEE || __ARM_FP16_FORMAT_ALTERNATIVE
#define _Float16 float
#endif // __ARM_FP16_FORMAT_IEEE || __ARM_FP16_FORMAT_ALTERNATIVE
#endif // __GNUC__ && !__clang__

namespace test_helper {

constexpr const size_t gMaximumNumberOfErrorMessages = 10;

// TODO: Figure out the build dependency to make including "CpuOperationUtils.h" work.
inline void convertFloat16ToFloat32(const _Float16* input, std::vector<float>* output) {
    for (size_t i = 0; i < output->size(); ++i) {
        (*output)[i] = static_cast<float>(input[i]);
    }
}

// This class is a workaround for two issues our code relies on:
// 1. sizeof(bool) is implementation defined.
// 2. vector<bool> does not allow direct pointer access via the data() method.
class bool8 {
   public:
    bool8() : mValue() {}
    /* implicit */ bool8(bool value) : mValue(value) {}
    inline operator bool() const { return mValue != 0; }

   private:
    uint8_t mValue;
};

static_assert(sizeof(bool8) == 1, "size of bool8 must be 8 bits");

typedef std::map<int, std::vector<uint32_t>> OperandDimensions;
typedef std::map<int, std::vector<float>> Float32Operands;
typedef std::map<int, std::vector<int32_t>> Int32Operands;
typedef std::map<int, std::vector<uint8_t>> Quant8AsymmOperands;
typedef std::map<int, std::vector<int16_t>> Quant16SymmOperands;
typedef std::map<int, std::vector<_Float16>> Float16Operands;
typedef std::map<int, std::vector<bool8>> Bool8Operands;
typedef std::map<int, std::vector<int8_t>> Quant8ChannelOperands;
typedef std::map<int, std::vector<uint16_t>> Quant16AsymmOperands;
typedef std::map<int, std::vector<int8_t>> Quant8SymmOperands;
struct MixedTyped {
    static constexpr size_t kNumTypes = 9;
    OperandDimensions operandDimensions;
    Float32Operands float32Operands;
    Int32Operands int32Operands;
    Quant8AsymmOperands quant8AsymmOperands;
    Quant16SymmOperands quant16SymmOperands;
    Float16Operands float16Operands;
    Bool8Operands bool8Operands;
    Quant8ChannelOperands quant8ChannelOperands;
    Quant16AsymmOperands quant16AsymmOperands;
    Quant8SymmOperands quant8SymmOperands;
};
typedef std::pair<MixedTyped, MixedTyped> MixedTypedExampleType;

// Mixed-typed examples
typedef struct {
    MixedTypedExampleType operands;
    // Specifies the RANDOM_MULTINOMIAL distribution tolerance.
    // If set to greater than zero, the input is compared as log-probabilities
    // to the output and must be within this tolerance to pass.
    float expectedMultinomialDistributionTolerance = 0.0;
} MixedTypedExample;

// Go through all index-value pairs of a given input type
template <typename T>
inline void for_each(const std::map<int, std::vector<T>>& idx_and_data,
                     std::function<void(int, const std::vector<T>&)> execute) {
    for (auto& i : idx_and_data) {
        execute(i.first, i.second);
    }
}

// non-const variant of for_each
template <typename T>
inline void for_each(std::map<int, std::vector<T>>& idx_and_data,
                     std::function<void(int, std::vector<T>&)> execute) {
    for (auto& i : idx_and_data) {
        execute(i.first, i.second);
    }
}

// Go through all index-value pairs of a given input type
template <typename T>
inline void for_each(const std::map<int, std::vector<T>>& golden,
                     std::map<int, std::vector<T>>& test,
                     std::function<void(int, const std::vector<T>&, std::vector<T>&)> execute) {
    for_each<T>(golden, [&test, &execute](int index, const std::vector<T>& g) {
        auto& t = test[index];
        execute(index, g, t);
    });
}

// Go through all index-value pairs of a given input type
template <typename T>
inline void for_each(
        const std::map<int, std::vector<T>>& golden, const std::map<int, std::vector<T>>& test,
        std::function<void(int, const std::vector<T>&, const std::vector<T>&)> execute) {
    for_each<T>(golden, [&test, &execute](int index, const std::vector<T>& g) {
        auto t = test.find(index);
        ASSERT_NE(t, test.end());
        execute(index, g, t->second);
    });
}

// internal helper for for_all
template <typename T>
inline void for_all_internal(std::map<int, std::vector<T>>& idx_and_data,
                             std::function<void(int, void*, size_t)> execute_this) {
    for_each<T>(idx_and_data, [&execute_this](int idx, std::vector<T>& m) {
        execute_this(idx, static_cast<void*>(m.data()), m.size() * sizeof(T));
    });
}

// Go through all index-value pairs of all input types
// expects a functor that takes (int index, void *raw data, size_t sz)
inline void for_all(MixedTyped& idx_and_data,
                    std::function<void(int, void*, size_t)> execute_this) {
    for_all_internal(idx_and_data.float32Operands, execute_this);
    for_all_internal(idx_and_data.int32Operands, execute_this);
    for_all_internal(idx_and_data.quant8AsymmOperands, execute_this);
    for_all_internal(idx_and_data.quant16SymmOperands, execute_this);
    for_all_internal(idx_and_data.float16Operands, execute_this);
    for_all_internal(idx_and_data.bool8Operands, execute_this);
    for_all_internal(idx_and_data.quant8ChannelOperands, execute_this);
    for_all_internal(idx_and_data.quant16AsymmOperands, execute_this);
    for_all_internal(idx_and_data.quant8SymmOperands, execute_this);
    static_assert(9 == MixedTyped::kNumTypes,
                  "Number of types in MixedTyped changed, but for_all function wasn't updated");
}

// Const variant of internal helper for for_all
template <typename T>
inline void for_all_internal(const std::map<int, std::vector<T>>& idx_and_data,
                             std::function<void(int, const void*, size_t)> execute_this) {
    for_each<T>(idx_and_data, [&execute_this](int idx, const std::vector<T>& m) {
        execute_this(idx, static_cast<const void*>(m.data()), m.size() * sizeof(T));
    });
}

// Go through all index-value pairs (const variant)
// expects a functor that takes (int index, const void *raw data, size_t sz)
inline void for_all(const MixedTyped& idx_and_data,
                    std::function<void(int, const void*, size_t)> execute_this) {
    for_all_internal(idx_and_data.float32Operands, execute_this);
    for_all_internal(idx_and_data.int32Operands, execute_this);
    for_all_internal(idx_and_data.quant8AsymmOperands, execute_this);
    for_all_internal(idx_and_data.quant16SymmOperands, execute_this);
    for_all_internal(idx_and_data.float16Operands, execute_this);
    for_all_internal(idx_and_data.bool8Operands, execute_this);
    for_all_internal(idx_and_data.quant8ChannelOperands, execute_this);
    for_all_internal(idx_and_data.quant16AsymmOperands, execute_this);
    for_all_internal(idx_and_data.quant8SymmOperands, execute_this);
    static_assert(
            9 == MixedTyped::kNumTypes,
            "Number of types in MixedTyped changed, but const for_all function wasn't updated");
}

// Helper template - resize test output per golden
template <typename T>
inline void resize_accordingly_(const std::map<int, std::vector<T>>& golden,
                                std::map<int, std::vector<T>>& test) {
    for_each<T>(golden, test,
                [](int, const std::vector<T>& g, std::vector<T>& t) { t.resize(g.size()); });
}

template <>
inline void resize_accordingly_<uint32_t>(const OperandDimensions& golden,
                                          OperandDimensions& test) {
    for_each<uint32_t>(
            golden, test,
            [](int, const std::vector<uint32_t>& g, std::vector<uint32_t>& t) { t = g; });
}

inline void resize_accordingly(const MixedTyped& golden, MixedTyped& test) {
    resize_accordingly_(golden.operandDimensions, test.operandDimensions);
    resize_accordingly_(golden.float32Operands, test.float32Operands);
    resize_accordingly_(golden.int32Operands, test.int32Operands);
    resize_accordingly_(golden.quant8AsymmOperands, test.quant8AsymmOperands);
    resize_accordingly_(golden.quant16SymmOperands, test.quant16SymmOperands);
    resize_accordingly_(golden.float16Operands, test.float16Operands);
    resize_accordingly_(golden.bool8Operands, test.bool8Operands);
    resize_accordingly_(golden.quant8ChannelOperands, test.quant8ChannelOperands);
    resize_accordingly_(golden.quant16AsymmOperands, test.quant16AsymmOperands);
    resize_accordingly_(golden.quant8SymmOperands, test.quant8SymmOperands);
    static_assert(9 == MixedTyped::kNumTypes,
                  "Number of types in MixedTyped changed, but resize_accordingly function wasn't "
                  "updated");
}

template <typename T>
void filter_internal(const std::map<int, std::vector<T>>& golden,
                     std::map<int, std::vector<T>>* filtered, std::function<bool(int)> is_ignored) {
    for_each<T>(golden, [filtered, &is_ignored](int index, const std::vector<T>& m) {
        auto& g = *filtered;
        if (!is_ignored(index)) g[index] = m;
    });
}

inline MixedTyped filter(const MixedTyped& golden,
                         std::function<bool(int)> is_ignored) {
    MixedTyped filtered;
    filter_internal(golden.operandDimensions, &filtered.operandDimensions, is_ignored);
    filter_internal(golden.float32Operands, &filtered.float32Operands, is_ignored);
    filter_internal(golden.int32Operands, &filtered.int32Operands, is_ignored);
    filter_internal(golden.quant8AsymmOperands, &filtered.quant8AsymmOperands, is_ignored);
    filter_internal(golden.quant16SymmOperands, &filtered.quant16SymmOperands, is_ignored);
    filter_internal(golden.float16Operands, &filtered.float16Operands, is_ignored);
    filter_internal(golden.bool8Operands, &filtered.bool8Operands, is_ignored);
    filter_internal(golden.quant8ChannelOperands, &filtered.quant8ChannelOperands, is_ignored);
    filter_internal(golden.quant16AsymmOperands, &filtered.quant16AsymmOperands, is_ignored);
    filter_internal(golden.quant8SymmOperands, &filtered.quant8SymmOperands, is_ignored);
    static_assert(9 == MixedTyped::kNumTypes,
                  "Number of types in MixedTyped changed, but compare function wasn't updated");
    return filtered;
}

// Compare results
template <typename T>
void compare_(const std::map<int, std::vector<T>>& golden,
              const std::map<int, std::vector<T>>& test, std::function<void(T, T)> cmp) {
    for_each<T>(golden, test, [&cmp](int index, const std::vector<T>& g, const std::vector<T>& t) {
        for (unsigned int i = 0; i < g.size(); i++) {
            SCOPED_TRACE(testing::Message()
                         << "When comparing output " << index << " element " << i);
            cmp(g[i], t[i]);
        }
    });
}

// TODO: Allow passing accuracy criteria from spec.
// Currently we only need relaxed accuracy criteria on mobilenet tests, so we return the quant8
// tolerance simply based on the current test name.
inline int getQuant8AllowedError() {
    const ::testing::TestInfo* const testInfo =
            ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string testCaseName = testInfo->test_case_name();
    const std::string testName = testInfo->name();
    // We relax the quant8 precision for all tests with mobilenet:
    // - CTS/VTS GeneratedTest and DynamicOutputShapeTest with mobilenet
    // - VTS CompilationCachingTest and CompilationCachingSecurityTest except for TOCTOU tests
    if (testName.find("mobilenet") != std::string::npos ||
        (testCaseName.find("CompilationCaching") != std::string::npos &&
         testName.find("TOCTOU") == std::string::npos)) {
        return 2;
    } else {
        return 1;
    }
}

inline void compare(const MixedTyped& golden, const MixedTyped& test,
                    float fpAtol = 1e-5f, float fpRtol = 1e-5f) {
    int quant8AllowedError = getQuant8AllowedError();
    for_each<uint32_t>(
            golden.operandDimensions, test.operandDimensions,
            [](int index, const std::vector<uint32_t>& g, const std::vector<uint32_t>& t) {
                SCOPED_TRACE(testing::Message()
                             << "When comparing dimensions for output " << index);
                EXPECT_EQ(g, t);
            });
    size_t totalNumberOfErrors = 0;
    compare_<float>(golden.float32Operands, test.float32Operands,
                    [&totalNumberOfErrors, fpAtol, fpRtol](float expected, float actual) {
                        // Compute the range based on both absolute tolerance and relative tolerance
                        float fpRange = fpAtol + fpRtol * std::abs(expected);
                        if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                            EXPECT_NEAR(expected, actual, fpRange);
                        }
                        if (std::abs(expected - actual) > fpRange) {
                            totalNumberOfErrors++;
                        }
                    });
    compare_<int32_t>(golden.int32Operands, test.int32Operands,
                      [&totalNumberOfErrors](int32_t expected, int32_t actual) {
                          if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                              EXPECT_EQ(expected, actual);
                          }
                          if (expected != actual) {
                              totalNumberOfErrors++;
                          }
                      });
    compare_<uint8_t>(golden.quant8AsymmOperands, test.quant8AsymmOperands,
                      [&totalNumberOfErrors, quant8AllowedError](uint8_t expected, uint8_t actual) {
                          if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                              EXPECT_NEAR(expected, actual, quant8AllowedError);
                          }
                          if (std::abs(expected - actual) > quant8AllowedError) {
                              totalNumberOfErrors++;
                          }
                      });
    compare_<int16_t>(golden.quant16SymmOperands, test.quant16SymmOperands,
                      [&totalNumberOfErrors](int16_t expected, int16_t actual) {
                          if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                              EXPECT_NEAR(expected, actual, 1);
                          }
                          if (std::abs(expected - actual) > 1) {
                              totalNumberOfErrors++;
                          }
                      });
    compare_<_Float16>(golden.float16Operands, test.float16Operands,
                       [&totalNumberOfErrors, fpAtol, fpRtol](_Float16 expected, _Float16 actual) {
                           // Compute the range based on both absolute tolerance and relative
                           // tolerance
                           float fpRange = fpAtol + fpRtol * std::abs(static_cast<float>(expected));
                           if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                               EXPECT_NEAR(expected, actual, fpRange);
                           }
                           if (std::abs(static_cast<float>(expected - actual)) > fpRange) {
                               totalNumberOfErrors++;
                           }
                       });
    compare_<bool8>(golden.bool8Operands, test.bool8Operands,
                    [&totalNumberOfErrors](bool expected, bool actual) {
                        if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                            EXPECT_EQ(expected, actual);
                        }
                        if (expected != actual) {
                            totalNumberOfErrors++;
                        }
                    });
    compare_<int8_t>(golden.quant8ChannelOperands, test.quant8ChannelOperands,
                     [&totalNumberOfErrors, &quant8AllowedError](int8_t expected, int8_t actual) {
                         if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                             EXPECT_NEAR(expected, actual, quant8AllowedError);
                         }
                         if (std::abs(static_cast<int>(expected) - static_cast<int>(actual)) >
                             quant8AllowedError) {
                             totalNumberOfErrors++;
                         }
                     });
    compare_<uint16_t>(golden.quant16AsymmOperands, test.quant16AsymmOperands,
                       [&totalNumberOfErrors](int16_t expected, int16_t actual) {
                           if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                               EXPECT_NEAR(expected, actual, 1);
                           }
                           if (std::abs(expected - actual) > 1) {
                               totalNumberOfErrors++;
                           }
                       });
    compare_<int8_t>(golden.quant8SymmOperands, test.quant8SymmOperands,
                     [&totalNumberOfErrors, quant8AllowedError](int8_t expected, int8_t actual) {
                         if (totalNumberOfErrors < gMaximumNumberOfErrorMessages) {
                             EXPECT_NEAR(expected, actual, quant8AllowedError);
                         }
                         if (std::abs(static_cast<int>(expected) - static_cast<int>(actual)) >
                             quant8AllowedError) {
                             totalNumberOfErrors++;
                         }
                     });

    static_assert(9 == MixedTyped::kNumTypes,
                  "Number of types in MixedTyped changed, but compare function wasn't updated");
    EXPECT_EQ(size_t{0}, totalNumberOfErrors);
}

// Calculates the expected probability from the unnormalized log-probability of
// each class in the input and compares it to the actual ocurrence of that class
// in the output.
inline void expectMultinomialDistributionWithinTolerance(const MixedTyped& test,
                                                         const MixedTypedExample& example) {
    // TODO: These should be parameters but aren't currently preserved in the example.
    const int kBatchSize = 1;
    const int kNumClasses = 1024;
    const int kNumSamples = 128;

    std::vector<int32_t> output = test.int32Operands.at(0);
    std::vector<int> class_counts;
    class_counts.resize(kNumClasses);
    for (int index : output) {
        class_counts[index]++;
    }
    std::vector<float> input;
    Float32Operands float32Operands = example.operands.first.float32Operands;
    if (!float32Operands.empty()) {
        input = example.operands.first.float32Operands.at(0);
    } else {
        std::vector<_Float16> inputFloat16 = example.operands.first.float16Operands.at(0);
        input.resize(inputFloat16.size());
        convertFloat16ToFloat32(inputFloat16.data(), &input);
    }
    for (int b = 0; b < kBatchSize; ++b) {
        float probability_sum = 0;
        const int batch_index = kBatchSize * b;
        for (int i = 0; i < kNumClasses; ++i) {
            probability_sum += expf(input[batch_index + i]);
        }
        for (int i = 0; i < kNumClasses; ++i) {
            float probability =
                    static_cast<float>(class_counts[i]) / static_cast<float>(kNumSamples);
            float probability_expected = expf(input[batch_index + i]) / probability_sum;
            EXPECT_THAT(probability,
                        ::testing::FloatNear(probability_expected,
                                             example.expectedMultinomialDistributionTolerance));
        }
    }
}

};  // namespace test_helper

#endif  // ANDROID_ML_NN_TOOLS_TEST_GENERATOR_TEST_HARNESS_H
