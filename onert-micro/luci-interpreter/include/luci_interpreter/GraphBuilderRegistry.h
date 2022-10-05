/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_INTERPRETER_GRAPH_BUILDER_REGISTRY__
#define __LUCI_INTERPRETER_GRAPH_BUILDER_REGISTRY__

#include <luci/Import/GraphBuilderRegistry.h>

namespace luci_interpreter
{

/**
 * @brief Creates and returns GraphBuilderSource, which allows to not copy constant buffers from
 * model's file.
 *
 * @warning Use this source only in case when model's buffer alive longer than Interpreter.
 */
std::unique_ptr<luci::GraphBuilderSource> source_without_constant_copying();

} // namespace luci_interpreter

#endif // __LUCI_INTERPRETER_GRAPH_BUILDER_REGISTRY__
