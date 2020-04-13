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

#ifndef _MIR_ONNX_IMPORTER_H
#define _MIR_ONNX_IMPORTER_H

#include "mir/Graph.h"

#include <memory>
#include <string>

namespace mir_onnx
{

std::unique_ptr<mir::Graph> importModelFromBinaryFile(const std::string &filename);
std::unique_ptr<mir::Graph> importModelFromTextFile(const std::string &filename);
// TODO Remove after changing all uses.
std::unique_ptr<mir::Graph> loadModel(const std::string &filename);

} // namespace mir_onnx

#endif // _MIR_ONNX_IMPORTER_H
