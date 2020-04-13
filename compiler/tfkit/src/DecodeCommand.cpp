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

#include "DecodeCommand.hpp"
#include "Support.hpp"

#include <tensorflow/core/framework/graph.pb.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace tfkit
{

int DecodeCommand::run(int argc, const char *const *argv) const
{
  tensorflow::GraphDef graph_def;

  CmdArguments cmdargs(argc, argv);

  auto ioconfig = make_ioconfig(cmdargs);

  google::protobuf::io::IstreamInputStream is{ioconfig->in()};
  google::protobuf::io::CodedInputStream coded_is{&is};

  if (!graph_def.ParseFromCodedStream(&coded_is))
  {
    std::cerr << "ERROR: Failed to parse tensorflow model" << std::endl;
    return 255;
  }

  google::protobuf::io::OstreamOutputStream os{ioconfig->out()};
  google::protobuf::TextFormat::Print(graph_def, &os);

  return 0;
}

} // namespace tfkit
