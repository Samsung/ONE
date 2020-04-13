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
 * @file     InferenceInterface.h
 * @brief    This file contains class for running the actual inference model
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __TFLITE_CLASSIFY_INFERENCE_INTERFACE_H__
#define __TFLITE_CLASSIFY_INFERENCE_INTERFACE_H__

#include "tflite/ext/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "tflite/InterpreterSession.h"
#include "tflite/NNAPISession.h"

#include <iostream>
#include <string>

/**
 * @brief Class to define a inference interface for recognizing data
 */
class InferenceInterface
{
public:
  /**
   * @brief Construct a new InferenceInterface object with parameters
   * @param[in] model_file  The filepath of the model FlatBuffer protocol buffer
   * @param[in] use_nnapi   The flag to distinguish between TfLite interpreter and NNFW runtime
   */
  InferenceInterface(const std::string &model_file, const bool use_nnapi);

  /**
   * @brief Destructor an InferenceInterface object
   */
  ~InferenceInterface();

  /**
   * @brief Copy the input data into model
   * @param[in] input_name  The label of the image input node
   * @param[in] data        The actual data to be copied into input tensor
   * @param[in] batch       The number of batch size
   * @param[in] height      The number of height size
   * @param[in] width       The number of width size
   * @param[in] channel     The number of channel size
   * @return N/A
   */
  void feed(const std::string &input_name, const std::vector<float> &data, const int batch,
            const int height, const int width, const int channel);
  /**
   * @brief Run the inference call
   * @param[in] output_name The label of the output node
   * @return N/A
   */
  void run(const std::string &output_name);

  /**
   * @brief Copy the output tensor back into the output array
   * @param[in] output_node The label of the output node
   * @param[in] outputs     The output data array
   * @return N/A
   */
  void fetch(const std::string &output_name, std::vector<float> &outputs);

  /**
   * @brief Get tensor size
   * @param[in] name  The label of the node
   * @result The size of tensor
   */
  int getTensorSize(const std::string &name);

private:
  std::unique_ptr<tflite::Interpreter> _interpreter;
  std::unique_ptr<tflite::FlatBufferModel> _model;
  std::shared_ptr<nnfw::tflite::Session> _sess;
};

#endif // __TFLITE_CLASSIFY_INFERENCE_INTERFACE_H__
