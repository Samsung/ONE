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

#include "Importer.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

bool from_txt(std::istream &is, ::caffe::NetParameter &param)
{
  google::protobuf::io::IstreamInputStream iis{&is};

  if (!google::protobuf::TextFormat::Parse(&iis, &param))
  {
    return false;
  }

  return true;
}

bool from_bin(std::istream &is, ::caffe::NetParameter &param)
{
  google::protobuf::io::IstreamInputStream iis{&is};
  google::protobuf::io::CodedInputStream cis{&iis};

  if (!param.ParseFromCodedStream(&cis))
  {
    return false;
  }

  return true;
}

bool from_txt(std::istream &is, ::caffe::PoolingParameter &param)
{
  ::google::protobuf::io::IstreamInputStream iis{&is};
  return google::protobuf::TextFormat::Parse(&iis, &param);
}
