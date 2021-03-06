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

#include "DecodeCommand.h"

#include <caffe/proto/caffe.pb.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <iostream>

int DecodeCommand::run(int, const char *const *) const
{
  caffe::NetParameter param;

  // Load binary from standard input
  google::protobuf::io::IstreamInputStream is{&std::cin};
  google::protobuf::io::CodedInputStream coded_is{&is};

  if (!param.ParseFromCodedStream(&coded_is))
  {
    std::cerr << "ERROR: Failed to parse caffemodel" << std::endl;
    return 255;
  }

  // Write text into standard output
  google::protobuf::io::OstreamOutputStream os{&std::cout};
  google::protobuf::TextFormat::Print(param, &os);

  return 0;
}
