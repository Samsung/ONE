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
 * @file     ImageClassifier.h
 * @brief    This file contains ImageClassifier class and Recognition structure
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __TFLITE_CLASSIFY_IMAGE_CLASSIFIER_H__
#define __TFLITE_CLASSIFY_IMAGE_CLASSIFIER_H__

#include "InferenceInterface.h"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

/**
 * @brief struct to define an immutable result returned by a Classifier
 */
struct Recognition
{
public:
  /**
   * @brief Construct a new Recognition object with confidence and title
   * @param[in] _confidence A sortable score for how good the recognition is relative to others.
   * Higher should be better.
   * @param[in] _title      Display name for the recognition
   */
  Recognition(float _confidence, std::string _title) : confidence(_confidence), title(_title) {}

  float confidence;  /** A sortable score for how good the recognition is relative to others. Higher
                        should be better. */
  std::string title; /** Display name for the recognition */
};

/**
 * @brief Class to define a classifier specialized to label images
 */
class ImageClassifier
{
public:
  /**
   * @brief Construct a new ImageClassifier object with parameters
   * @param[in] model_file  The filepath of the model FlatBuffer protocol buffer
   * @param[in] label_file  The filepath of label file for classes
   * @param[in] input_size  The input size. A square image of input_size x input_size is assumed
   * @param[in] image_mean  The assumed mean of the image values
   * @param[in] image_std   The assumed std of the image values
   * @param[in] input_name  The label of the image input node
   * @param[in] output_name The label of the output node
   * @param[in] use_nnapi   The flag to distinguish between TfLite interpreter and NNFW runtime
   */
  ImageClassifier(const std::string &model_file, const std::string &label_file,
                  const int input_size, const int image_mean, const int image_std,
                  const std::string &input_name, const std::string &output_name,
                  const bool use_nnapi);

  /**
   * @brief Recognize the given image data
   * @param[in] image   The image data to recognize
   * @return An immutable result vector array
   */
  std::vector<Recognition> recognizeImage(const cv::Mat &image);

private:
  const float _threshold = 0.1f;
  const unsigned int _max_results = 3;

  std::unique_ptr<InferenceInterface> _inference;
  int _input_size;
  int _image_mean;
  int _image_std;
  std::string _input_name;
  std::string _output_name;

  std::vector<std::string> _labels;
  std::vector<float> _fdata;
  std::vector<float> _outputs;
  int _num_classes;
};

#endif // __TFLITE_CLASSIFY_IMAGE_CLASSIFIER_H__
