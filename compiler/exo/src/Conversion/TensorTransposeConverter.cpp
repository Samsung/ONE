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

#include "TensorTransposeConverter.h"

#include "Dialect/IR/TFLNodes.h"

#include <loco.h>
#include <loco/Service/ShapeInference.h>

#include <oops/InternalExn.h>

#include <algorithm>
#include <cassert>
#include <vector>

namespace
{

void validate_perm(loco::TensorTranspose *origin)
{
  // check perm values are correct
  std::vector<uint32_t> base_perms; // such as {0, 1, 2, 3, ... }
  std::vector<uint32_t> perms;      // perm values in TensorTranspose

  base_perms.resize(origin->perm()->size());
  perms.resize(origin->perm()->size());
  for (loco::TensorAxis x = 0; x < origin->perm()->size(); x++)
  {
    base_perms[x] = x;
    perms[x] = origin->perm()->axis(x);
  }

  if (!std::is_permutation(base_perms.begin(), base_perms.end(), perms.begin()))
    INTERNAL_EXN("wrong perm value");
}

} // namespace

namespace exo
{
/**
 * @brief Converts loco::TensorTranspose to locoex::TFLTranspose
 */
bool TensorTransposeConverter::convert(loco::TensorTranspose *origin)
{
  auto *graph = origin->graph();

  auto tfl_transpose = graph->nodes()->create<locoex::TFLTranspose>();
  {
    // validation
    {
      assert(origin->input() != nullptr);

      auto input_rank = loco::shape_get(origin->input()).as<loco::TensorShape>().rank();
      if (input_rank != origin->perm()->size())
        INTERNAL_EXN_V("perm size should be same with input rank",
                       oops::to_uint32(origin->perm()->size()));

      validate_perm(origin);
    }

    tfl_transpose->a(origin->input());

    // perm : set TFLConst
    auto perm_const = graph->nodes()->create<locoex::TFLConst>();
    {
      perm_const->dtype(loco::DataType::S32);
      perm_const->rank(1);
      perm_const->dim(0) = origin->perm()->size();
      perm_const->size<loco::DataType::S32>(origin->perm()->size());

      // add perm values into perm TFLConst
      for (loco::TensorAxis x = 0; x < origin->perm()->size(); x++)
      {
        perm_const->at<loco::DataType::S32>(x) = origin->perm()->axis(x);
      }
    }
    tfl_transpose->perm(perm_const);
  }

  // replace canonical node
  loco::replace(origin).with(tfl_transpose);
  origin->input(nullptr);

  return true;
}

} // namespace exo
