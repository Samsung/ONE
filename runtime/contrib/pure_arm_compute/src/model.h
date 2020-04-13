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

/**
 * @file model.h
 * @brief This file contains ANeuralNetworksModel classe for handling Model NNAPI such as
 * ANeuralNetworksModel_create, ANeuralNetworksModel_addOperand
 * @ingroup COM_AI_RUNTIME
 */

#ifndef __MODEL_H__
#define __MODEL_H__

#include "internal/Model.h"

/**
 * @brief struct to express Model of NNAPI
 */
struct ANeuralNetworksModel
{
public:
  /**
   * @brief Construct without params
   */
  ANeuralNetworksModel();

public:
  /**
   * @brief Get reference of internal::tflite::Model
   * @return Reference of internal::tflite::Model
   */
  internal::tflite::Model &deref(void) { return *_model; }

public:
  /**
   * @brief Release internal::tflite::Model pointer to param
   * @param [in] model To get released internal::tflite::Model pointer
   * @return N/A
   */
  void release(std::shared_ptr<const internal::tflite::Model> &model) { model = _model; }
  /**
   * @brief Get @c true if ANeuralNetworksModel_finish has been called, otherwise @c false
   * @return @c true if ANeuralNetworksModel_finish has been called, otherwise @c false
   */
  bool isFinished() { return _isFinished == true; }
  /**
   * @brief Mark model process finished
   * @return N/A
   */
  void markAsFinished() { _isFinished = true; }

private:
  std::shared_ptr<internal::tflite::Model> _model;
  bool _isFinished{false};
};

#endif // __MODEL_H__
