/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_ACL_COMMON_LAYOUT_RESOLVER_H__
#define __ONERT_BACKEND_ACL_COMMON_LAYOUT_RESOLVER_H__

#include <backend/BackendContext.h>
#include <ir/Layout.h>

namespace onert
{
namespace backend
{
namespace acl_common
{

class LayoutResolver
{
public:
  void operator()(backend::ContextData &data);

private:
  void insertTransposeOps(backend::ContextData &data);
  void removeTwofoldTransposeOps(backend::ContextData &data);
  ir::Layout backendLayout() const;
  bool checkAllOfLegalLayout(const ir::Graph &graph, ir::Layout backend_layout) const;
};

} // namespace acl_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ACL_COMMON_LAYOUT_RESOLVER_H__
