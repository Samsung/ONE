/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_NNPKG_H__
#define __ONERT_IR_NNPKG_H__

#include <memory>

#include "ir/Index.h"
#include "ir/Model.h"

namespace onert
{
namespace ir
{

class NNPkg
{
public:
  NNPkg() = default;
  NNPkg(const NNPkg &obj) = default;
  NNPkg(NNPkg &&) = default;
  NNPkg &operator=(const NNPkg &) = default;
  NNPkg &operator=(NNPkg &&) = default;
  ~NNPkg() = default;

  NNPkg(std::shared_ptr<Model> model) { _models[ModelIndex{0}] = model; }
  std::shared_ptr<Model> primary_model() { return _models.at(onert::ir::ModelIndex{0}); }

  /**
   * @brief Put model at index
   *
   * @param[in] model Model to be pushed
   * @param[in] index Index where Model is to be pushed
   */
  void push(ModelIndex index, const std::shared_ptr<Model> &model) { _models[index] = model; }

  /**
   * @brief Get the count of model
   *
   * @return the count of models
   */
  size_t model_count() const { return _models.size(); }

  /**
   * @brief Get model at index
   *
   * @param[in] index Index of the model to be returned
   * @return Model at index
   */
  const std::shared_ptr<Model> &model(const ModelIndex &index) const { return _models.at(index); }
  /**
   * @brief Get model at index
   *
   * @param[in] index Index of the model to be returned
   * @return Model at index
   */
  std::shared_ptr<Model> &model(const ModelIndex &index) { return _models.at(index); }

private:
  std::unordered_map<ModelIndex, std::shared_ptr<Model>> _models;
  // TODO: Add connection between models
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_NNPKG_H__
