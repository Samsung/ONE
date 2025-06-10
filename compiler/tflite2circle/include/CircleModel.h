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

using BufferData = std::vector<uint8_t>;
using MapBufferData = std::map<int32_t, BufferData>;

template <typename T> class Offset
{
private:
  using TFLFlatBufVec = flatbuffers::Vector<typename T::TFL>;
  using CIRFlatBufVecOffset = flatbuffers::Offset<flatbuffers::Vector<typename T::CIR>>;
  using SignatureDefs = flatbuffers::Vector<flatbuffers::Offset<::tflite::SignatureDef>>;

public:
  Offset(void) = delete;
  Offset(FlatBufBuilder &fb) : _fb{fb} {};

public:
  void set_signature_defs(const SignatureDefs *offset) { _tfl_signature_def_offsets = offset; }
  void set_buffer_data_map(MapBufferData *map) { _buffer_data_map = map; }
  void set_file_raw(const std::vector<char> *raw) { _file_raw = raw; }

public:
  void build(const TFLFlatBufVec *tflite_flatbuffer_vec);

public:
  CIRFlatBufVecOffset offset(void) const { return _circle_flatbuffer_vec_offset; }

private:
  FlatBufBuilder &_fb;
  CIRFlatBufVecOffset _circle_flatbuffer_vec_offset;
  // TODO revise this when Circle supports SignatureDef
  const SignatureDefs *_tfl_signature_def_offsets = nullptr;
  // for extended buffer for size > 2G
  const std::vector<char> *_file_raw = nullptr;
  MapBufferData *_buffer_data_map = nullptr;
};

class CircleModel
{
private:
  using Description = flatbuffers::Offset<flatbuffers::String>;

public:
  CircleModel(void) = delete;
  CircleModel(FlatBufBuilder &fb, const std::vector<char> &fr);

public:
  void load_offsets(const tflite::Model *tfl_model);
  void model_build(void) const;
  void finalize(void);
  const char *base(void) const;
  size_t size(void) const;

private:
  uint32_t _version;
  Description _description;
  FlatBufBuilder &_fb;
  const std::vector<char> &_file_raw;
  std::unique_ptr<Offset<OperatorCodeLink>> _operator_codes_offset;
  std::unique_ptr<Offset<SubGraphLink>> _subGraphs_offset;
  std::unique_ptr<Offset<BufferLink>> _buffers_offset;
  std::unique_ptr<Offset<MetaDataBufferLink>> _metadata_buffer_offset;

  MapBufferData _buffer_data_map;
  std::string _fb_data_with_ext;
};

} // namespace tflite2circle

#endif // __CIRCLE_MODEL_H__
