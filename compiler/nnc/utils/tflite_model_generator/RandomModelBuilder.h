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

#ifndef RANDOM_MODELBUILDER_H
#define RANDOM_MODELBUILDER_H

#include <iostream>
#include <memory>
#include <functional>
#include <random>

#include "Tree.h"

namespace modelgen
{
/**
 *  @brief ModelSaver this class describes a common interface for saving model.
 */
class ModelSaver
{
public:
  ModelSaver() = default;
  virtual ~ModelSaver() = default;

  /**
   * @brief saveModel does export a owned model.
   */
  virtual void saveModel() = 0;
};

/**
 * @brief RandomModelBuilder this class describe a common interface for building
 * random neural network models.
 */
class RandomModelBuilder
{
public:
  /**
   * @brief constructor for RandomModelBuilder.
   * @details Be careful, opCreators's initializer list should save the order
   * from the OpCode enum.
   */
  RandomModelBuilder()
      : _operatorCounts{0}, _gen(_rd()),
        _floatRand(std::numeric_limits<float>::min(), std::numeric_limits<float>::max()),
        _intRand(static_cast<int32_t>(OpCodes::opFirst), static_cast<int32_t>(OpCodes::opLast))
  {
    _opCreators[static_cast<int>(OpCodes::opConv2d)] =
        [this](treebuilder::Tree *t, treebuilder::Operation *op) { createLayerCONV_2D(t, op); };
    _opCreators[static_cast<int>(OpCodes::opConcatenation)] = [this](
        treebuilder::Tree *t, treebuilder::Operation *op) { createLayerCONCATENATION(t, op); };
    _opCreators[static_cast<int>(OpCodes::opDepthwiseConv2d)] = [this](
        treebuilder::Tree *t, treebuilder::Operation *op) { createLayerDEPTHWISE_CONV_2D(t, op); };
    _opCreators[static_cast<int>(OpCodes::opOpMaxPool2d)] = [this](treebuilder::Tree *t,
                                                                   treebuilder::Operation *op) {
      createLayerX_POOL_2D(t, op, OpCodes::opOpMaxPool2d);
    };
    _opCreators[static_cast<int>(OpCodes::opAveragePool2d)] = [this](treebuilder::Tree *t,
                                                                     treebuilder::Operation *op) {
      createLayerX_POOL_2D(t, op, OpCodes::opAveragePool2d);
    };
    _opCreators[static_cast<int>(OpCodes::opSoftmax)] =
        [this](treebuilder::Tree *t, treebuilder::Operation *op) { createLayerSOFTMAX(t, op); };
    _opCreators[static_cast<int>(OpCodes::opFullyConnected)] = [this](
        treebuilder::Tree *t, treebuilder::Operation *op) { createLayerFULLY_CONNECTED(t, op); };
  };

  virtual ~RandomModelBuilder() = default;

  virtual void convertTreeToModel(treebuilder::Tree *t) = 0;

  /**
   * @brief getModelSaver does create unique_ptr to ModelSaver.
   * @return unique_ptr to ModelSaver.
   */
  virtual std::unique_ptr<ModelSaver> createModelSaver() = 0;

protected:
  /**
   * @Brief createInput does add input tensor to model.
   */
  virtual void createInput(treebuilder::Tree *t) = 0;
  /**
   * @brief addOperator does add a new layer to model.
   */
  virtual void addOperator(treebuilder::Tree *t, treebuilder::Operation *op) = 0;

  /**
   * @brief createLayerXXX are creator for operators.
   * @param input_tensor_id is id of input tensor.
   */
  virtual void createLayerCONV_2D(treebuilder::Tree *, treebuilder::Operation *) = 0;
  virtual void createLayerCONCATENATION(treebuilder::Tree *t, treebuilder::Operation *op) = 0;
  virtual void createLayerDEPTHWISE_CONV_2D(treebuilder::Tree *t, treebuilder::Operation *op) = 0;
  virtual void createLayerX_POOL_2D(treebuilder::Tree *t, treebuilder::Operation *op,
                                    OpCodes opcode) = 0;
  virtual void createLayerSOFTMAX(treebuilder::Tree *t, treebuilder::Operation *op) = 0;
  virtual void createLayerFULLY_CONNECTED(treebuilder::Tree *t, treebuilder::Operation *op) = 0;

  /**
   * @brief opCreators this array contains a lambda with call of method
   * for building specified operator.
   * @details This array is used for convenient creation random operators,
   * like follow: opCreators[OpCodes::opCount]
   * For example: opCreators[OpCodes::opConv2d](0) -- will lead to call createLayerCONV_2D method.
   */
  std::function<void(treebuilder::Tree *, treebuilder::Operation *)>
      _opCreators[static_cast<int32_t>(OpCodes::opCount)];
  /**
   * @brief operatorCounts this array contains amount of used operators in generated model.
   * @details For example: operatorCounts[Op_CONV_2D] -- amount of used 2D convolution operators.
   */
  int _operatorCounts[static_cast<int32_t>(OpCodes::opCount)];

  std::random_device _rd;
  std::mt19937 _gen;
  std::uniform_real_distribution<float> _floatRand;
  std::uniform_int_distribution<int> _intRand;
};
} // namespace modelgen

#endif // RANDOM_MODELBUILDER_H
