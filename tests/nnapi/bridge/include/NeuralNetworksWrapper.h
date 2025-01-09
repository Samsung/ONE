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

// Provides C++ classes to more easily use the Neural Networks API.

#ifndef __NNFW_RT_NEURAL_NETWORKS_WRAPPER_H__
#define __NNFW_RT_NEURAL_NETWORKS_WRAPPER_H__

// Fix for onert:
//  Additional include NeuralNetworksEx.h
#include "NeuralNetworks.h"
#include "NeuralNetworksEx.h"

#include <math.h>
#include <optional>
#include <string>
#include <vector>

namespace nnfw {
namespace rt {
namespace wrapper {

enum class Type {
    FLOAT32 = ANEURALNETWORKS_FLOAT32,
    INT32 = ANEURALNETWORKS_INT32,
    UINT32 = ANEURALNETWORKS_UINT32,
    TENSOR_FLOAT32 = ANEURALNETWORKS_TENSOR_FLOAT32,
    TENSOR_INT32 = ANEURALNETWORKS_TENSOR_INT32,
    TENSOR_QUANT8_ASYMM = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM,
    BOOL = ANEURALNETWORKS_BOOL,
    TENSOR_QUANT16_SYMM = ANEURALNETWORKS_TENSOR_QUANT16_SYMM,
    TENSOR_FLOAT16 = ANEURALNETWORKS_TENSOR_FLOAT16,
    TENSOR_BOOL8 = ANEURALNETWORKS_TENSOR_BOOL8,
    FLOAT16 = ANEURALNETWORKS_FLOAT16,
    TENSOR_QUANT8_SYMM_PER_CHANNEL = ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL,
    TENSOR_QUANT16_ASYMM = ANEURALNETWORKS_TENSOR_QUANT16_ASYMM,
    TENSOR_QUANT8_SYMM = ANEURALNETWORKS_TENSOR_QUANT8_SYMM,
};

enum class ExecutePreference {
    PREFER_LOW_POWER = ANEURALNETWORKS_PREFER_LOW_POWER,
    PREFER_FAST_SINGLE_ANSWER = ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER,
    PREFER_SUSTAINED_SPEED = ANEURALNETWORKS_PREFER_SUSTAINED_SPEED
};

enum class Result {
    NO_ERROR = ANEURALNETWORKS_NO_ERROR,
    OUT_OF_MEMORY = ANEURALNETWORKS_OUT_OF_MEMORY,
    INCOMPLETE = ANEURALNETWORKS_INCOMPLETE,
    UNEXPECTED_NULL = ANEURALNETWORKS_UNEXPECTED_NULL,
    BAD_DATA = ANEURALNETWORKS_BAD_DATA,
    OP_FAILED = ANEURALNETWORKS_OP_FAILED,
    UNMAPPABLE = ANEURALNETWORKS_UNMAPPABLE,
    BAD_STATE = ANEURALNETWORKS_BAD_STATE,
    OUTPUT_INSUFFICIENT_SIZE = ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE,
    UNAVAILABLE_DEVICE = ANEURALNETWORKS_UNAVAILABLE_DEVICE,
};

struct SymmPerChannelQuantParams {
    ANeuralNetworksSymmPerChannelQuantParams params;
    std::vector<float> scales;

    SymmPerChannelQuantParams(std::vector<float> scalesVec, uint32_t channelDim)
        : scales(std::move(scalesVec)) {
        params = {
                .channelDim = channelDim,
                .scaleCount = static_cast<uint32_t>(scales.size()),
                .scales = scales.size() > 0 ? scales.data() : nullptr,
        };
    }

    SymmPerChannelQuantParams(const SymmPerChannelQuantParams& other)
        : params(other.params), scales(other.scales) {
        params.scales = scales.size() > 0 ? scales.data() : nullptr;
    }

    SymmPerChannelQuantParams& operator=(const SymmPerChannelQuantParams& other) {
        if (this != &other) {
            params = other.params;
            scales = other.scales;
            params.scales = scales.size() > 0 ? scales.data() : nullptr;
        }
        return *this;
    }
};

struct OperandType {
    ANeuralNetworksOperandType operandType;
    std::vector<uint32_t> dimensions;
    std::optional<SymmPerChannelQuantParams> channelQuant;

    OperandType(const OperandType& other)
        : operandType(other.operandType),
          dimensions(other.dimensions),
          channelQuant(other.channelQuant) {
        operandType.dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr;
    }

    OperandType& operator=(const OperandType& other) {
        if (this != &other) {
            operandType = other.operandType;
            dimensions = other.dimensions;
            channelQuant = other.channelQuant;
            operandType.dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr;
        }
        return *this;
    }

    OperandType(Type type, std::vector<uint32_t> d, float scale = 0.0f, int32_t zeroPoint = 0)
        : dimensions(std::move(d)), channelQuant(std::nullopt) {
        operandType = {
                .type = static_cast<int32_t>(type),
                .dimensionCount = static_cast<uint32_t>(dimensions.size()),
                .dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr,
                .scale = scale,
                .zeroPoint = zeroPoint,
        };
    }

    OperandType(Type type, std::vector<uint32_t> data, float scale, int32_t zeroPoint,
                SymmPerChannelQuantParams&& channelQuant)
        : dimensions(std::move(data)), channelQuant(std::move(channelQuant)) {
        operandType = {
                .type = static_cast<int32_t>(type),
                .dimensionCount = static_cast<uint32_t>(dimensions.size()),
                .dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr,
                .scale = scale,
                .zeroPoint = zeroPoint,
        };
    }
};

class Memory {
   public:
    Memory(size_t size, int protect, int fd, size_t offset) {
        mValid = ANeuralNetworksMemory_createFromFd(size, protect, fd, offset, &mMemory) ==
                 ANEURALNETWORKS_NO_ERROR;
    }

    Memory(AHardwareBuffer* buffer) {
        mValid = ANeuralNetworksMemory_createFromAHardwareBuffer(buffer, &mMemory) ==
                 ANEURALNETWORKS_NO_ERROR;
    }

    ~Memory() { ANeuralNetworksMemory_free(mMemory); }

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Memory(Memory&& other) { *this = std::move(other); }
    Memory& operator=(Memory&& other) {
        if (this != &other) {
            ANeuralNetworksMemory_free(mMemory);
            mMemory = other.mMemory;
            mValid = other.mValid;
            other.mMemory = nullptr;
            other.mValid = false;
        }
        return *this;
    }

    ANeuralNetworksMemory* get() const { return mMemory; }
    bool isValid() const { return mValid; }

   private:
    ANeuralNetworksMemory* mMemory = nullptr;
    bool mValid = true;
};

class Model {
   public:
    Model() {
        // TODO handle the value returned by this call
        ANeuralNetworksModel_create(&mModel);
    }
    ~Model() { ANeuralNetworksModel_free(mModel); }

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Model(Model&& other) { *this = std::move(other); }
    Model& operator=(Model&& other) {
        if (this != &other) {
            ANeuralNetworksModel_free(mModel);
            mModel = other.mModel;
            mNextOperandId = other.mNextOperandId;
            mValid = other.mValid;
            other.mModel = nullptr;
            other.mNextOperandId = 0;
            other.mValid = false;
        }
        return *this;
    }

    Result finish() {
        if (mValid) {
            auto result = static_cast<Result>(ANeuralNetworksModel_finish(mModel));
            if (result != Result::NO_ERROR) {
                mValid = false;
            }
            return result;
        } else {
            return Result::BAD_STATE;
        }
    }

    uint32_t addOperand(const OperandType* type) {
        if (ANeuralNetworksModel_addOperand(mModel, &(type->operandType)) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
        if (type->channelQuant) {
            if (ANeuralNetworksModel_setOperandSymmPerChannelQuantParams(
                        mModel, mNextOperandId, &type->channelQuant.value().params) !=
                ANEURALNETWORKS_NO_ERROR) {
                mValid = false;
            }
        }
        return mNextOperandId++;
    }

    void setOperandValue(uint32_t index, const void* buffer, size_t length) {
        if (ANeuralNetworksModel_setOperandValue(mModel, index, buffer, length) !=
            ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }

    void setOperandValueFromMemory(uint32_t index, const Memory* memory, uint32_t offset,
                                   size_t length) {
        if (ANeuralNetworksModel_setOperandValueFromMemory(mModel, index, memory->get(), offset,
                                                           length) != ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }

    void addOperation(ANeuralNetworksOperationType type, const std::vector<uint32_t>& inputs,
                      const std::vector<uint32_t>& outputs) {
        if (ANeuralNetworksModel_addOperation(mModel, type, static_cast<uint32_t>(inputs.size()),
                                              inputs.data(), static_cast<uint32_t>(outputs.size()),
                                              outputs.data()) != ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }

    // Fix for onert: addOperationEx for operation support extension (NeuralNetworksEx.h)
    void addOperationEx(ANeuralNetworksOperationTypeEx type, const std::vector<uint32_t>& inputs,
                      const std::vector<uint32_t>& outputs) {
        if (ANeuralNetworksModel_addOperationEx(mModel, type, static_cast<uint32_t>(inputs.size()),
                                              inputs.data(), static_cast<uint32_t>(outputs.size()),
                                              outputs.data()) != ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }

    void identifyInputsAndOutputs(const std::vector<uint32_t>& inputs,
                                  const std::vector<uint32_t>& outputs) {
        if (ANeuralNetworksModel_identifyInputsAndOutputs(
                    mModel, static_cast<uint32_t>(inputs.size()), inputs.data(),
                    static_cast<uint32_t>(outputs.size()),
                    outputs.data()) != ANEURALNETWORKS_NO_ERROR) {
            mValid = false;
        }
    }

    void relaxComputationFloat32toFloat16(bool isRelax) {
        if (ANeuralNetworksModel_relaxComputationFloat32toFloat16(mModel, isRelax) ==
            ANEURALNETWORKS_NO_ERROR) {
            mRelaxed = isRelax;
        }
    }

    ANeuralNetworksModel* getHandle() const { return mModel; }
    bool isValid() const { return mValid; }
    bool isRelaxed() const { return mRelaxed; }

   protected:
    ANeuralNetworksModel* mModel = nullptr;
    // We keep track of the operand ID as a convenience to the caller.
    uint32_t mNextOperandId = 0;
    bool mValid = true;
    bool mRelaxed = false;
};

class Event {
   public:
    Event() {}
    ~Event() { ANeuralNetworksEvent_free(mEvent); }

    // Disallow copy semantics to ensure the runtime object can only be freed
    // once. Copy semantics could be enabled if some sort of reference counting
    // or deep-copy system for runtime objects is added later.
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    // Move semantics to remove access to the runtime object from the wrapper
    // object that is being moved. This ensures the runtime object will be
    // freed only once.
    Event(Event&& other) { *this = std::move(other); }
    Event& operator=(Event&& other) {
        if (this != &other) {
            ANeuralNetworksEvent_free(mEvent);
            mEvent = other.mEvent;
            other.mEvent = nullptr;
        }
        return *this;
    }

    Result wait() { return static_cast<Result>(ANeuralNetworksEvent_wait(mEvent)); }

    // Only for use by Execution
    void set(ANeuralNetworksEvent* newEvent) {
        ANeuralNetworksEvent_free(mEvent);
        mEvent = newEvent;
    }

   private:
    ANeuralNetworksEvent* mEvent = nullptr;
};

class Compilation {
   public:
    Compilation(const Model* model) {
        int result = ANeuralNetworksCompilation_create(model->getHandle(), &mCompilation);
        if (result != 0) {
            // TODO Handle the error
        }
    }

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

   private:
    ANeuralNetworksCompilation* mCompilation = nullptr;
};

class Execution {
   public:
    Execution(const Compilation* compilation) {
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

    Result compute() { return static_cast<Result>(ANeuralNetworksExecution_compute(mExecution)); }

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
    ANeuralNetworksExecution* mExecution = nullptr;
};

}  // namespace wrapper
}  // namespace rt
}  // namespace nnfw

#endif  // __NNFW_RT_NEURAL_NETWORKS_WRAPPER_H__
