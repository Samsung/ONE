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

#ifndef MIR_CAFFE2_PROTO_HELPER_H
#define MIR_CAFFE2_PROTO_HELPER_H

#include "caffe2/proto/caffe2.pb.h"

namespace mir_caffe2
{

using RepArgument = const ::google::protobuf::RepeatedPtrField<::caffe2::Argument> &;

const ::caffe2::Argument &findArgumentByName(RepArgument args, const std::string &name);

const bool hasArgument(RepArgument args, const std::string &name);

int getSingleArgument(const ::caffe2::OperatorDef &op, const std::string &argument_name,
                      int default_value);
float getSingleArgument(const ::caffe2::OperatorDef &op, const std::string &argument_name,
                        float default_value);
std::string getSingleArgument(const ::caffe2::OperatorDef &op, const std::string &argument_name,
                              const std::string &default_value);

} // namespace mir_caffe2

#endif // MIR_CAFFE2_PROTO_HELPER_H
