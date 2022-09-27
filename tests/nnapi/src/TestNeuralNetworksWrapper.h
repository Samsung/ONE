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

// Provides C++ classes to more easily use the Neural Networks API.
// TODO(b/117845862): this should be auto generated from NeuralNetworksWrapper.h.

#ifndef ANDROID_ML_NN_RUNTIME_TEST_TEST_NEURAL_NETWORKS_WRAPPER_H
#define ANDROID_ML_NN_RUNTIME_TEST_TEST_NEURAL_NETWORKS_WRAPPER_H

#include "NeuralNetworks.h"
#include "NeuralNetworksWrapper.h"
// Fix for onert: comment out include NeuralNetworksWrapperExtensions.h
//#include "NeuralNetworksWrapperExtensions.h"

#include <math.h>
#include <optional>
#include <string>
#include <vector>

namespace nnfw {
namespace rt {
namespace test_wrapper {

using wrapper::Event;
using wrapper::ExecutePreference;
// Fix for onert: comment out ExtensionModel, ExtensionOperandParams and ExtensionOperandType
//using wrapper::ExtensionModel;
//using wrapper::ExtensionOperandParams;
//using wrapper::ExtensionOperandType;
using wrapper::Memory;
using wrapper::Model;
using wrapper::OperandType;
using wrapper::Result;
using wrapper::SymmPerChannelQuantParams;
using wrapper::Type;

class Compilation {
   public:
    Compilation(const Model* model) {
        int result = ANeuralNetworksCompilation_create(model->getHandle(), &mCompilation);
        if (result != 0) {
            // TODO Handle the error
        }
    }

    Compilation() {}

    ~Compilation() { ANeuralNetworksCompilation_free(mCompilation); }

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Compilation(const Compilation&) = delete;
    Compilation& operator=(const Compilation&) = delete;

    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Compilation(Compilation&& other) { *this = std::move(other); }
    Compilation& operator=(Compilation&& other) {
        if (this != &other) {
            ANeuralNetworksCompilation_free(mCompilation);
            mCompilation = other.mCompilation;
            other.mCompilation = nullptr;
        }
        return *this;
    }

    Result setPreference(ExecutePreference preference) {
        return static_cast<Result>(ANeuralNetworksCompilation_setPreference(
                mCompilation, static_cast<int32_t>(preference)));
    }

    Result setCaching(const std::string& cacheDir, const std::vector<uint8_t>& token) {
        if (token.size() != ANEURALNETWORKS_BYTE_SIZE_OF_CACHE_TOKEN) {
            return Result::BAD_DATA;
        }
        return static_cast<Result>(ANeuralNetworksCompilation_setCaching(
                mCompilation, cacheDir.c_str(), token.data()));
    }

    Result finish() { return static_cast<Result>(ANeuralNetworksCompilation_finish(mCompilation)); }

    ANeuralNetworksCompilation* getHandle() const { return mCompilation; }

   protected:
    ANeuralNetworksCompilation* mCompilation = nullptr;
};

class Execution {
   public:
    Execution(const Compilation* compilation) : mCompilation(compilation->getHandle()) {
        int result = ANeuralNetworksExecution_create(compilation->getHandle(), &mExecution);
        if (result != 0) {
            // TODO Handle the error
        }
    }

    ~Execution() { ANeuralNetworksExecution_free(mExecution); }

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Execution(const Execution&) = delete;
    Execution& operator=(const Execution&) = delete;

    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Execution(Execution&& other) { *this = std::move(other); }
    Execution& operator=(Execution&& other) {
        if (this != &other) {
            ANeuralNetworksExecution_free(mExecution);
            mCompilation = other.mCompilation;
            other.mCompilation = nullptr;
            mExecution = other.mExecution;
            other.mExecution = nullptr;
        }
        return *this;
    }

    Result setInput(uint32_t index, const void* buffer, size_t length,
                    const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(
                ANeuralNetworksExecution_setInput(mExecution, index, type, buffer, length));
    }

    Result setInputFromMemory(uint32_t index, const Memory* memory, uint32_t offset,
                              uint32_t length, const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(ANeuralNetworksExecution_setInputFromMemory(
                mExecution, index, type, memory->get(), offset, length));
    }

    Result setOutput(uint32_t index, void* buffer, size_t length,
                     const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(
                ANeuralNetworksExecution_setOutput(mExecution, index, type, buffer, length));
    }

    Result setOutputFromMemory(uint32_t index, const Memory* memory, uint32_t offset,
                               uint32_t length, const ANeuralNetworksOperandType* type = nullptr) {
        return static_cast<Result>(ANeuralNetworksExecution_setOutputFromMemory(
                mExecution, index, type, memory->get(), offset, length));
    }

    Result startCompute(Event* event) {
        ANeuralNetworksEvent* ev = nullptr;
        Result result = static_cast<Result>(ANeuralNetworksExecution_startCompute(mExecution, &ev));
        event->set(ev);
        return result;
    }

    Result compute() {
        switch (mComputeMode) {
            case ComputeMode::SYNC: {
                return static_cast<Result>(ANeuralNetworksExecution_compute(mExecution));
            }
            case ComputeMode::ASYNC: {
                ANeuralNetworksEvent* event = nullptr;
                Result result = static_cast<Result>(
                        ANeuralNetworksExecution_startCompute(mExecution, &event));
                if (result != Result::NO_ERROR) {
                    return result;
                }
                // TODO how to manage the lifetime of events when multiple waiters is not
                // clear.
                result = static_cast<Result>(ANeuralNetworksEvent_wait(event));
                ANeuralNetworksEvent_free(event);
                return result;
            }
            case ComputeMode::BURST: {
                ANeuralNetworksBurst* burst = nullptr;
                Result result =
                        static_cast<Result>(ANeuralNetworksBurst_create(mCompilation, &burst));
                if (result != Result::NO_ERROR) {
                    return result;
                }
                result = static_cast<Result>(
                        ANeuralNetworksExecution_burstCompute(mExecution, burst));
                ANeuralNetworksBurst_free(burst);
                return result;
            }
        }
        return Result::BAD_DATA;
    }

    // By default, compute() uses the synchronous API. setComputeMode() can be
    // used to change the behavior of compute() to either:
    // - use the asynchronous API and then wait for computation to complete
    // or
    // - use the burst API
    // Returns the previous ComputeMode.
    enum class ComputeMode { SYNC, ASYNC, BURST };
    static ComputeMode setComputeMode(ComputeMode mode) {
        ComputeMode oldComputeMode = mComputeMode;
        mComputeMode = mode;
        return oldComputeMode;
    }

    Result getOutputOperandDimensions(uint32_t index, std::vector<uint32_t>* dimensions) {
        uint32_t rank = 0;
        Result result = static_cast<Result>(
                ANeuralNetworksExecution_getOutputOperandRank(mExecution, index, &rank));
        dimensions->resize(rank);
        if ((result != Result::NO_ERROR && result != Result::OUTPUT_INSUFFICIENT_SIZE) ||
            rank == 0) {
            return result;
        }
        result = static_cast<Result>(ANeuralNetworksExecution_getOutputOperandDimensions(
                mExecution, index, dimensions->data()));
        return result;
    }

   private:
    ANeuralNetworksCompilation* mCompilation = nullptr;
    ANeuralNetworksExecution* mExecution = nullptr;

    // Initialized to ComputeMode::SYNC in TestNeuralNetworksWrapper.cpp.
    static ComputeMode mComputeMode;
};

}  // namespace test_wrapper
}  // namespace nn
}  // namespace android

#endif  // ANDROID_ML_NN_RUNTIME_TEST_TEST_NEURAL_NETWORKS_WRAPPER_H
