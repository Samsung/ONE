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

#ifndef __LUCI_SPARSIFY_TENSOR_PASS_H__
#define __LUCI_SPARSIFY_TENSOR_PASS_H__

#include <logo/Pass.h>

#include <luci/IR/SparsityParam.h>

namespace luci
{

class CircleConst;

/**
 * @brief  Pass to sparsify tensor
 */
struct SparsifyTensorPass final : public logo::Pass
{
public:
  SparsifyTensorPass(std::string &tensor_name, std::vector<int32_t> &traversal_order,
                     std::vector<DimensionType> &format, std::vector<int32_t> &block_size,
                     std::vector<int32_t> &block_map)
      : _tensor_name{tensor_name}, _traversal_order{traversal_order}, _format{format},
        _block_size{block_size}, _block_map{block_map}
  {
    // DO NOTHING
  }

public:
  const char *name(void) const final { return "luci::SparsifyTensorPass"; }

  bool run(loco::Graph *g) final;

  bool sparsify_tensor(luci::CircleConst *cop);

private:
  std::string _tensor_name;
  std::vector<int32_t> _traversal_order;
  std::vector<DimensionType> _format;
  std::vector<int32_t> _block_size;
  std::vector<int32_t> _block_map;
};

} // namespace luci

#endif // __LUCI_SPARSIFY_TENSOR_PASS_H__
