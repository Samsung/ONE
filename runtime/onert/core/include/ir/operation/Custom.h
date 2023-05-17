/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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
#ifndef __ONERT_IR_OPERATION_CUSTOM_H__
#define __ONERT_IR_OPERATION_CUSTOM_H__

#include "ir/Operation.h"

#include <cstring>

namespace onert
{
namespace ir
{
namespace operation
{

class Custom : public Operation
{
public:
  struct Userdata
  {
    char *data;
    size_t size;

    Userdata() : data{nullptr}, size{0} {}
    Userdata(const Userdata &o)
    {
      size = o.size;
      data = new char[size];
      std::memcpy(data, o.data, size);
    }
    ~Userdata() { delete[] data; }
  };

  Custom(OperandConstraint input_constr, const OperandIndexSequence &inputs,
         const OperandIndexSequence &outputs, std::string id, const Userdata &userdata);

  void accept(OperationVisitor &v) const override;
  void accept(MutableOperationVisitor &v) override;

public:
  /**
   * @return unique operation identifier
   */
  const std::string &id() const;

  std::string name() const override;
  OpCode opcode() const final { return OpCode::Custom; }

  /**
   * @return user-provided data
   */
  const Userdata &userdata() const;

private:
  std::string _id;
  Userdata _userdata;
};

} // namespace operation
} // namespace ir
} // namespace onert
#endif // __ONERT_IR_OPERATION_CUSTOM_H__
