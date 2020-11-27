/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef NNCC_PASSDATA_H
#define NNCC_PASSDATA_H

#include "mir/Graph.h"
#include "mir/TensorVariant.h"

namespace nnc
{

/**
 * @brief class that encapsulate value returned and taken by pass
 */
class PassData
{
public:
  /* implicit */ PassData(std::nullptr_t data)
      : // NOLINT(google-explicit-constructor, hicpp-explicit-conversions)
        _dataContainer{.unknown = data}, _dataType(PDT::UNKNOWN)
  {
  }

  /**
   * @brief Implicit conversion from Graph* to PassData
   */
  /* implicit */ PassData(mir::Graph *graph)
      : // NOLINT(google-explicit-constructor, hicpp-explicit-conversions)
        _dataContainer{.graph = graph}, _dataType(PDT::GRAPH)
  {
  }

  /**
   * @brief Implicit conversion from PassData to Graph*
   */
  /* implicit */ operator mir::Graph *() const
  { // NOLINT(google-explicit-constructor, hicpp-explicit-conversions)
    if (_dataType != PDT::GRAPH)
      return nullptr;
    return _dataContainer.graph;
  }

  /**
   * @brief Implicit conversion from Graph* to PassData
   */
  /* implicit */ PassData(mir::TensorVariant *tv)
      : // NOLINT(google-explicit-constructor, hicpp-explicit-conversions)
        _dataContainer{.tensorVariant = tv}, _dataType(PDT::TENSOR_VARIANT)
  {
  }

  /**
   * @brief Implicit conversion from PassData to Graph*
   */
  /* implicit */ operator mir::TensorVariant *() const
  { // NOLINT(google-explicit-constructor, hicpp-explicit-conversions)
    if (_dataType != PDT::TENSOR_VARIANT)
      return nullptr;
    return _dataContainer.tensorVariant;
  }

private:
  // types that PassData can contain
  enum class PDT : char
  {
    GRAPH,
    TENSOR_VARIANT,
    UNKNOWN

  } _dataType;

  // union contains all pointers to objects that can be returned from passes
  union {
    mir::Graph *graph;
    mir::TensorVariant *tensorVariant;
    void *unknown;

  } _dataContainer;
};

} // namespace nnc

#endif // NNCC_PASSDATA_H
