/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MemoryManagement.h"

#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "memory_management/NaiveAssignment.h"
#include "memory_management/Types.h"
#include "Shape.h"
#include "Types.h"
#include "Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

OffsetsAssignment ObjectsToOffsets(const ObjectsAssignment<size_t> &obj_assignment)
{
  size_t num_tensors = obj_assignment.object_ids.size();
  size_t num_objects = obj_assignment.object_sizes.size();
  OffsetsAssignment result = {/*offsets=*/std::vector<size_t>(num_tensors),
                              /*total_size=*/0};
  std::vector<size_t> ids_to_offset(num_objects);
  for (size_t i = 0; i < num_objects; ++i)
  {
    ids_to_offset[i] = result.total_size;
    result.total_size += obj_assignment.object_sizes[i];
  }
  for (size_t i = 0; i < num_tensors; ++i)
  {
    result.offsets[i] = ids_to_offset[obj_assignment.object_ids[i]];
  }
  return result;
}

template <>
absl::Status AssignObjectsToTensors(const std::vector<TensorUsageRecord<size_t>> &usage_records,
                                    MemoryStrategy strategy, ObjectsAssignment<size_t> *assignment,
                                    const UsageGraph *)
{
  switch (strategy)
  {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    default:
      return absl::InternalError("MemoryStrategy is not supported with current tensor size type.");
  }
}

template <>
absl::Status AssignObjectsToTensors(const std::vector<TensorUsageRecord<BHWC>> &usage_records,
                                    MemoryStrategy strategy, ObjectsAssignment<BHWC> *assignment,
                                    const UsageGraph *)
{
  switch (strategy)
  {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    default:
      return absl::InternalError("MemoryStrategy is not supported with current tensor size type.");
  }
}

template <>
absl::Status AssignObjectsToTensors(const std::vector<TensorUsageRecord<uint2>> &usage_records,
                                    MemoryStrategy strategy, ObjectsAssignment<uint2> *assignment,
                                    const UsageGraph *)
{
  switch (strategy)
  {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    default:
      return absl::InternalError("MemoryStrategy is not supported with current tensor size type.");
  }
}

template <>
absl::Status AssignObjectsToTensors(const std::vector<TensorUsageRecord<uint3>> &usage_records,
                                    MemoryStrategy strategy, ObjectsAssignment<uint3> *assignment,
                                    const UsageGraph *)
{
  switch (strategy)
  {
    case MemoryStrategy::NAIVE:
      return NaiveAssignment(usage_records, assignment);
    default:
      return absl::InternalError("MemoryStrategy is not supported with current tensor size type.");
  }
}

absl::Status AssignOffsetsToTensors(const std::vector<TensorUsageRecord<size_t>> &usage_records,
                                    const MemoryStrategy &strategy, OffsetsAssignment *assignment,
                                    const UsageGraph *reallocation_graph)
{

  ObjectsAssignment<size_t> objects_assignment;
  RETURN_IF_ERROR(
    AssignObjectsToTensors(usage_records, strategy, &objects_assignment, reallocation_graph));
  *assignment = ObjectsToOffsets(objects_assignment);
  return absl::OkStatus();
}

} // namespace gpu_cl
} // namespace backend
} // namespace onert
