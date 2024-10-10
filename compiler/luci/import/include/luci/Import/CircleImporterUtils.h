/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __CIRCLE_IMPORTER_UTILS_H__
#define __CIRCLE_IMPORTER_UTILS_H__

#include <luci/IR/CircleNodes.h>

#include <loco.h>

#include <mio/circle/schema_generated.h>

namespace luci
{

luci::CompressionType from_circle_compressiontype(circle::CompressionType type);

} // namespace luci

#endif // __CIRCLE_IMPORTER_UTILS_H__
