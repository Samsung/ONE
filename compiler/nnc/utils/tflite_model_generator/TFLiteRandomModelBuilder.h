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

#ifndef TFLITEBUILDER_H
#define TFLITEBUILDER_H

#include <vector>
#include <memory>

#include "Tree.h"
#include "RandomModelBuilder.h"
#include "schema_generated.h"

namespace modelgen
{
using namespace tflite;

/**
 * @brief TFLiteModelSaver contains the unique_ptr to model and does convertation of model
 * to FlatBuffer format and it's saving.
 */
class TFLiteModelSaver : public ModelSaver
{
public:
  TFLiteModelSaver() = default;
  ~TFLiteModelSaver() override = default;

  TFLiteModelSaver(std::unique_ptr<ModelT> &&m) : ModelSaver(), _model(std::move(m)) {}

  void saveModel() override;

private:
  flatbuffers::FlatBufferBuilder _flatBufferBuilder;
  std::unique_ptr<ModelT> _model;
};

/**
 * @brief TFLiteRandomModelBuilder does build random TFLiteModel
 * and gives unique_ptr to ModelSaver for export it to file.
 */
class TFLiteRandomModelBuilder : public RandomModelBuilder
{
public:
  TFLiteRandomModelBuilder();
  ~TFLiteRandomModelBuilder() override = default;

  void convertTreeToModel(treebuilder::Tree *t) override;

  std::unique_ptr<ModelSaver> createModelSaver() override;

protected:
  void createInput(treebuilder::Tree *t) override;
  void addOperator(treebuilder::Tree *t, treebuilder::Operation *op) override;
  /**
   *  Operations:
   */
  void createLayerCONV_2D(treebuilder::Tree *t, treebuilder::Operation *op) override;
  void createLayerCONCATENATION(treebuilder::Tree *t, treebuilder::Operation *op) override;
  void createLayerDEPTHWISE_CONV_2D(treebuilder::Tree *t, treebuilder::Operation *op) override;
  void createLayerX_POOL_2D(treebuilder::Tree *t, treebuilder::Operation *op,
                            OpCodes opcode) override;
  void createLayerSOFTMAX(treebuilder::Tree *t, treebuilder::Operation *op) override;
  void createLayerFULLY_CONNECTED(treebuilder::Tree *t, treebuilder::Operation *op) override;

private:
  /**
   * @brief createEmptyTensor does create tensor without buffer
   * and add it to tensors array of SubGraphT.
   * @param shape is a shape of tensor.
   * @param name is a name of tensor.
   * @return unique_ptr to created tensor.
   */
  std::unique_ptr<TensorT> createEmptyTensor(const std::vector<int32_t> &shape, const char *name);
  /**
   * @brief createTensorWthBuffer does create tensor with buffer
   * and add it to tensors array of SubGraphT.
   * @param shape is a shape of tensor.
   * @param name is a name of tensor.
   * @return unique_ptr to created tensor.
   */
  std::unique_ptr<TensorT> createTensorWthBuffer(const std::vector<int32_t> &shape,
                                                 const char *name);
  /**
   * @brief createEmptyOperator does create operator without tensors and
   * add operator to operators array of SubGraphT
   * @param opcode is operator's code.
   * @return unique_ptr to created operator.
   */
  std::unique_ptr<OperatorT> createEmptyOperator(treebuilder::Operation *op);

  std::unique_ptr<ModelT> _model;
  /**
   * @details This vector contains a index of tensor (in subgraph tflite vector)
   *          for output operand of tree's node `i`.
   */
  std::vector<int32_t> _operandTree2tensor;

  /**
   * @brief mapOperatorCode contains indexes to operator_codes array in ModelT.
   */
  int32_t _mapOperatorCode[static_cast<int32_t>(OpCodes::opCount)];
};
} // namespace modelgen

#endif // TFLITEBUILDER_H
