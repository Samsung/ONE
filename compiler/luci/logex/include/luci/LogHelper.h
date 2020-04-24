/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_LOG_HELPER_H__
#define __LUCI_LOG_HELPER_H__

#include "RawGraphDumper.h"

#include <locop/FormattedGraph.h>
#include <loco.h>

#include <memory>

namespace luci
{

using FormattedGraph = locop::FormattedGraphImpl<locop::Formatter::LinearV1>;

FormattedGraph fmt(loco::Graph *g);

static inline FormattedGraph fmt(const std::unique_ptr<loco::Graph> &g) { return fmt(g.get()); }

RawDumpGraph raw(loco::Graph *g);

} // namespace luci

#endif // __LUCI_LOG_HELPER_H__
