/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __CIRCLE_MODEL_H__
#define __CIRCLE_MODEL_H__

#include <mio/tflite/schema_generated.h>
#include <mio/circle/schema_generated.h>

#include <iostream>
#include <string>
#include <vector>

#include "TFLModel.h"

namespace tflite2circle
{

using FlatBufBuilder = std::unique_ptr<flatbuffers::FlatBufferBuilder>;

struct OperatorCodeLink
{
  using TFL = flatbuffers::Offset<::tflite::OperatorCode>;
  using CIR = flatbuffers::Offset<::circle::OperatorCode>;
};

struct SubGraphLink
{
  using TFL = flatbuffers::Offset<::tflite::SubGraph>;
  using CIR = flatbuffers::Offset<::circle::SubGraph>;
};

struct BufferLink
{
  using TFL = flatbuffers::Offset<::tflite::Buffer>;
  using CIR = flatbuffers::Offset<::circle::Buffer>;
};

struct MetaDataBufferLink
{
  using TFL = int32_t;
  using CIR = int32_t;
};

template <typename T> class Offset
{
private:
  using TFLFlatBufVec = flatbuffers::Vector<typename T::TFL>;
  using CIRFlatBufVecOffset = flatbuffers::Offset<flatbuffers::Vector<typename T::CIR>>;

public:
  Offset(void) = delete;
  Offset(FlatBufBuilder &fb, const TFLFlatBufVec *tflite_flatbuffer_vec);

public:
  CIRFlatBufVecOffset offset(void) const { return _circle_flatbuffer_vec_offset; }

private:
  CIRFlatBufVecOffset _circle_flatbuffer_vec_offset;
};

class CircleModel
{
private:
  using Description = flatbuffers::Offset<flatbuffers::String>;

public:
  CircleModel(void) = delete;
  CircleModel(FlatBufBuilder &fb, TFLModel &tfl_model);

public:
  void model_build(void) const;
  const char *base(void) const;
  size_t size(void) const;

private:
  uint32_t _version;
  Description _description;
  FlatBufBuilder &_fb;
  std::unique_ptr<Offset<OperatorCodeLink>> _operator_codes_offset;
  std::unique_ptr<Offset<SubGraphLink>> _subGraphs_offset;
  std::unique_ptr<Offset<BufferLink>> _buffers_offset;
  std::unique_ptr<Offset<MetaDataBufferLink>> _metadata_buffer_offset;
};

} // namespace tflite2circle

#endif // __CIRCLE_MODEL_H__
