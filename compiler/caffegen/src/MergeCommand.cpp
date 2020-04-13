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

#include "MergeCommand.h"

#include <caffe/proto/caffe.pb.h>
#include <caffe/caffe.hpp>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <iostream>
#include <string>

int MergeCommand::run(int argc, const char *const *argv) const
{
  if (argc != 2)
  {
    std::cerr << "ERROR: this command requires exactly 2 arguments" << std::endl;
    return 254;
  }

  std::string model_file = argv[0];
  std::string trained_file = argv[1];

  // Load the network
  caffe::Net<float> caffe_test_net(model_file, caffe::TEST);
  // Load the weights
  caffe_test_net.CopyTrainedLayersFrom(trained_file);

  caffe::NetParameter net_param;
  caffe_test_net.ToProto(&net_param);

  // Write binary with initialized params into standard output
  google::protobuf::io::OstreamOutputStream os(&std::cout);
  google::protobuf::io::CodedOutputStream coded_os{&os};

  if (!net_param.SerializeToCodedStream(&coded_os))
  {
    std::cerr << "ERROR: Failed to serialize" << std::endl;
    return 255;
  }
  return 0;
}
