/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __SUPPORT_H__
#define __SUPPORT_H__

#include <tensorflow/core/framework/graph.pb.h>

#include <iostream>
#include <string>

namespace tfkit
{
namespace tf
{

bool HasAttr(const tensorflow::NodeDef &, const std::string &);
tensorflow::DataType GetDataTypeAttr(const tensorflow::NodeDef &, const std::string &);
tensorflow::TensorProto *GetTensorAttr(tensorflow::NodeDef &, const std::string &);
/// GetElementCount returns -1 for rank-0 tensor shape
int GetElementCount(const tensorflow::TensorShapeProto &);

} // namespace tf

class CmdArguments
{
public:
  CmdArguments() = delete;
  CmdArguments(int argc, const char *const *argv)
    : _argc(static_cast<unsigned int>(argc)), _argv{argv}
  {
  }

  std::string get(unsigned int index) const;
  std::string get_or(unsigned int index, const std::string &) const;

private:
  unsigned int _argc;
  const char *const *_argv;
};

class IOConfiguration
{
public:
  std::istream *in() const { return _in ? _in.get() : &std::cin; }
  std::ostream *out() const { return _out ? _out.get() : &std::cout; }

public:
  void in(std::unique_ptr<std::istream> &&in) { _in = std::move(in); }
  void out(std::unique_ptr<std::ostream> &&out) { _out = std::move(out); }

private:
  std::unique_ptr<std::istream> _in;
  std::unique_ptr<std::ostream> _out;
};

std::unique_ptr<IOConfiguration> make_ioconfig(const CmdArguments &cmdargs);

} // namespace tfkit

#endif // __SUPPORT_H__
