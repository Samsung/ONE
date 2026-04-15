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

#ifndef __LOG_HELPER_H__
#define __LOG_HELPER_H__

#include <locop/FormattedGraph.h>

#include <loco/IR/FeatureShape.h>
#include <loco/IR/FilterShape.h>
#include <loco/IR/TensorShape.h>

#include <sstream>
#include <vector>

namespace loco
{

/**
 * @brief dump FeatureShape values to stream
 */
std::ostream &operator<<(std::ostream &os, const loco::FeatureShape &feature_shape);

/**
 * @brief dump FilterShape values to stream
 */
std::ostream &operator<<(std::ostream &os, const loco::FilterShape &filter_shape);

/**
 * @brief dump TensorShape values to stream
 */
std::ostream &operator<<(std::ostream &os, const loco::TensorShape &tensor_shape);

/**
 * @brief dump Padding2D values to stream
 */
std::ostream &operator<<(std::ostream &os, const loco::Padding2D &pad);

} // namespace loco

/**
 * @brief dump std::vector<int64_t> values to stream
 */
std::ostream &operator<<(std::ostream &os, const std::vector<int64_t> &vi64);

namespace moco
{
namespace tf
{

using FormattedGraph = locop::FormattedGraphImpl<locop::Formatter::LinearV1>;

FormattedGraph fmt(loco::Graph *g);

static inline FormattedGraph fmt(const std::unique_ptr<loco::Graph> &g) { return fmt(g.get()); }

} // namespace tf
} // namespace moco

#endif // __LOG_HELPER_H__
