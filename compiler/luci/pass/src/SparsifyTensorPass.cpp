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

#include "luci/Pass/SparsifyTensorPass.h"

#include "Sparsifier.h"

#include <luci/IR/CircleNodes.h>

namespace luci
{

template <loco::DataType DT> void SparsifyTensorPass::sparsify_tensor(luci::CircleConst *cop)
{
  using PRIMITIVE_DTYPE = typename loco::DataTypeImpl<DT>::Type;

  std::vector<int32_t> dense_tensor_shape(cop->rank());
  for (uint32_t d = 0; d < cop->rank(); d++)
  {
    dense_tensor_shape.at(d) = cop->dim(d).value();
  }

  Sparsifier<PRIMITIVE_DTYPE> sparsifier(dense_tensor_shape, _traversal_order, _format, _block_size,
                                         _block_map);
  // get dense tensor data
  uint32_t dense_tensor_data_size = cop->size<DT>();
  std::vector<PRIMITIVE_DTYPE> dense_tensor_data(dense_tensor_data_size);
  for (uint32_t i = 0; i < dense_tensor_data_size; i++)
  {
    dense_tensor_data.at(i) = cop->at<DT>(i);
  }
  // sparsify
  sparsifier.DenseToSparse(dense_tensor_data.data());
  // get sparse tensor data
  std::vector<PRIMITIVE_DTYPE> sparse_tensor_data = sparsifier.GetData();
  uint32_t sparse_tensor_data_size = sparse_tensor_data.size();
  cop->size<DT>(sparse_tensor_data_size);
  for (uint32_t i = 0; i < sparse_tensor_data_size; i++)
  {
    cop->at<DT>(i) = sparse_tensor_data.at(i);
  }
  // make sparsity parameter
  auto sparsityparam = std::make_unique<SparsityParam>();
  sparsityparam->traversal_order = _traversal_order;
  sparsityparam->block_map = _block_map;
  // get dimension meta data
  const auto dim_metadata = sparsifier.GetDimMetadata();
  for (uint32_t idx = 0; idx < _format.size(); idx++)
  {
    if (_format.at(idx) == DimensionType::DENSE)
    {
      sparsityparam->dim_metadata.emplace_back(DimensionType::DENSE,
                                               dim_metadata.at(idx * 2).at(0));
    }
    // TODO Set SparseIndexVectorType according to its data range
    else if (_format.at(idx) == DimensionType::SPARSE_CSR)
    {
      sparsityparam->dim_metadata.emplace_back(
          DimensionType::SPARSE_CSR, /* dense size */ 0,
          /* array_segments */ SparseIndexVector{SparseIndexVectorType::U16,
                                                 dim_metadata.at(idx * 2)},
          /* array_indices */ SparseIndexVector{SparseIndexVectorType::U16,
                                                dim_metadata.at(idx * 2 + 1)});
    }
  }
  for (uint32_t i = 0; i < _block_size.size(); i++)
  {
    assert(_block_size.at(i) == dim_metadata.at((_format.size() + i) * 2).at(0));
    sparsityparam->dim_metadata.emplace_back(DimensionType::DENSE, _block_size.at(i));
  }
  cop->sparsityparam(std::move(sparsityparam));
}

bool SparsifyTensorPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cop = dynamic_cast<luci::CircleConst *>(node);
    if (not cop)
      continue;

    if (cop->name() != _tensor_name)
      continue;

    switch (cop->dtype())
    {
      case loco::DataType::S32:
        sparsify_tensor<loco::DataType::S32>(cop);
        break;
      case loco::DataType::S8:
        sparsify_tensor<loco::DataType::S8>(cop);
        break;
      case loco::DataType::FLOAT32:
        sparsify_tensor<loco::DataType::FLOAT32>(cop);
        break;
      default:
        throw std::runtime_error("SparsifyTensorPass: Unsupported dtype.");
    }
    changed = true;
  }

  return changed;
}

template void SparsifyTensorPass::sparsify_tensor<loco::DataType::S32>(luci::CircleConst *cop);
template void SparsifyTensorPass::sparsify_tensor<loco::DataType::S8>(luci::CircleConst *cop);
template void SparsifyTensorPass::sparsify_tensor<loco::DataType::FLOAT32>(luci::CircleConst *cop);

} // namespace luci
