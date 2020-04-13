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

/**
 * @file TestHelper.h
 */

#ifndef __PLIER_TF_TEST_HELPER_H__
#define __PLIER_TF_TEST_HELPER_H__

#include <tensorflow/core/framework/graph.pb.h>

namespace plier
{
namespace tf
{

bool parse_graphdef(char const *pbtxt, tensorflow::GraphDef &graphdef);

bool parse_nodedef(char const *pbtxt, tensorflow::NodeDef &nodedef);

} // namespace tf
} // namespace plier

#endif // __PLIER_TF_TEST_HELPER_H__
