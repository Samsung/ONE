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

#ifndef __OP_CHEF_H__
#define __OP_CHEF_H__

#include <circlechef.pb.h>
#include <mio/circle/schema_generated.h>

#include <memory>

struct OpChef
{
  virtual ~OpChef() = default;

  virtual circle::BuiltinOperator code(void) const = 0;
  virtual circle::BuiltinOptions type(void) const = 0;
  virtual flatbuffers::Offset<void> value(flatbuffers::FlatBufferBuilder &fbb) const = 0;

  // TODO Find a way to place this method in a better place
  virtual flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
  custom_value(flatbuffers::FlatBufferBuilder &fbb) const
  {
    return flatbuffers::Offset<flatbuffers::Vector<uint8_t>>();
  }
};

struct OpChefFactory
{
  virtual ~OpChefFactory() = default;

  virtual std::unique_ptr<OpChef> create(const circlechef::Operation *operation) const = 0;
};

#endif // __OP_CHEF_H__
