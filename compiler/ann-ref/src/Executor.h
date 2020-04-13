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

#ifndef __EXECUTOR_H__
#define __EXECUTOR_H__

#include "Model.h"

#include "Shape.h"
#include "Request.h"

#include <vector>

// Information we maintain about each operand during execution that
// may change during execution.
struct RunTimeOperandInfo
{
  // TODO Storing the type here is redundant, as it won't change during execution.
  OperandType type;

  // The type and dimensions of the operand.  The dimensions can
  // change at runtime.  We include the type because it's useful
  // to pass together with the dimension to the functions implementing
  // the operators.
  //
  // Q: Is it possible??
  std::vector<uint32_t> dimensions;
  float scale;
  int32_t zeroPoint;

  // Where the operand's data is stored.  Check the corresponding
  // location information in the model to figure out if this points
  // to memory we have allocated for an temporary operand.
  uint8_t *buffer;
  // The length of the buffer.
  uint32_t length;

  // Whether this is a temporary variable, a model input, a constant, etc.
  OperandLifeTime lifetime;

  // Keeps track of how many operations have yet to make use
  // of this temporary variable.  When the count is decremented to 0,
  // we free the buffer.  For non-temporary variables, this count is
  // always 0.
  uint32_t numberOfUsesLeft;

  Shape shape() const
  {
    return Shape{.type = type, .dimensions = dimensions, .scale = scale, .offset = zeroPoint};
  }
};

// Used to keep a pointer to each of the memory pools
struct RunTimePoolInfo
{
  uint8_t *buffer;

  bool set(uint8_t *m)
  {
    buffer = m;
    return true;
  }
};

// This class is used to execute a model on the CPU.
class Executor
{
public:
  // Executes the model. The results will be stored at the locations
  // specified in the constructor.
  // The model must outlive the executor.  We prevent it from being modified
  // while this is executing.
  int run(const Model &model, const Request &request,
          const std::vector<RunTimePoolInfo> &modelPoolInfos,
          const std::vector<RunTimePoolInfo> &requestPoolInfos);

private:
  bool initializeRunTimeInfo(const std::vector<RunTimePoolInfo> &modelPoolInfos,
                             const std::vector<RunTimePoolInfo> &requestPoolInfos);
  // Runs one operation of the graph.
  int executeOperation(const Operation &entry);
  // Decrement the usage count for the operands listed.  Frees the memory
  // allocated for any temporary variable with a count of zero.
  void freeNoLongerUsedOperands(const std::vector<uint32_t> &inputs);

  // The model and the request that we'll execute. Only valid while run()
  // is being executed.
  const Model *mModel = nullptr;
  const Request *mRequest = nullptr;

  // We're copying the list of all the dimensions from the model, as
  // these may be modified when we run the operatins.  Since we're
  // making a full copy, the indexes used in the operand description
  // stay valid.
  //    std::vector<uint32_t> mDimensions;
  // Runtime information about all the operands.
  std::vector<RunTimeOperandInfo> mOperands;
};

#endif // __CPU_EXECUTOR_H__
