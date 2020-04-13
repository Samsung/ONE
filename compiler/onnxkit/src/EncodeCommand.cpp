/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "EncodeCommand.hpp"
#include "Support.hpp"

#include <onnx/onnx.pb.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

int EncodeCommand::run(int argc, const char *const *argv) const
{
  onnx::ModelProto model;

  Cmdline cmdline(argc, argv);

  auto ui = make_ui(cmdline);

  google::protobuf::io::IstreamInputStream is{ui->in()};

  if (!google::protobuf::TextFormat::Parse(&is, &model))
  {
    std::cerr << "ERROR: Failed to parse text" << std::endl;
    return 255;
  }

  google::protobuf::io::OstreamOutputStream os{ui->out()};
  google::protobuf::io::CodedOutputStream coded_os{&os};

  if (!model.SerializeToCodedStream(&coded_os))
  {
    std::cerr << "ERROR: Failed to serialize a ONNX model" << std::endl;
    return 255;
  }

  return 0;
}
