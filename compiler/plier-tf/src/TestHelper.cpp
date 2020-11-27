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

// This is copied from moco

#include <plier/tf/TestHelper.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <istream>

namespace
{

struct membuf : std::streambuf
{
  membuf(char const *base, size_t size)
  {
    char *p(const_cast<char *>(base));
    this->setg(p, p, p + size);
  }
};

struct imemstream : virtual membuf, std::istream
{
  imemstream(char const *base, size_t size)
    : membuf(base, size), std::istream(static_cast<std::streambuf *>(this))
  {
  }
};

} // namespace

namespace plier
{
namespace tf
{

bool parse_graphdef(char const *pbtxt, tensorflow::GraphDef &graphdef)
{
  imemstream mempb(pbtxt, std::strlen(pbtxt));
  google::protobuf::io::IstreamInputStream iis(&mempb);
  return google::protobuf::TextFormat::Parse(&iis, &graphdef);
}

bool parse_nodedef(char const *pbtxt, tensorflow::NodeDef &nodedef)
{
  imemstream mempb(pbtxt, std::strlen(pbtxt));
  google::protobuf::io::IstreamInputStream iis(&mempb);
  return google::protobuf::TextFormat::Parse(&iis, &nodedef);
}

} // namespace tf
} // namespace plier
