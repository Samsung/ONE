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

#include "ImageClassifier.h"

#include <fstream>
#include <queue>
#include <algorithm>

ImageClassifier::ImageClassifier(const std::string &model_file, const std::string &label_file,
                                 const int input_size, const int image_mean, const int image_std,
                                 const std::string &input_name, const std::string &output_name,
                                 const bool use_nnapi)
  : _inference(new InferenceInterface(model_file, use_nnapi)), _input_size(input_size),
    _image_mean(image_mean), _image_std(image_std), _input_name(input_name),
    _output_name(output_name)
{
  // Load label
  std::ifstream label_stream(label_file.c_str());
  assert(label_stream);

  std::string line;
  while (std::getline(label_stream, line))
  {
    _labels.push_back(line);
  }
  _num_classes = _inference->getTensorSize(_output_name);
  std::cout << "Output tensor size is " << _num_classes << ", label size is " << _labels.size()
            << std::endl;

  // Pre-allocate buffers
  _fdata.reserve(_input_size * _input_size * 3);
  _outputs.reserve(_num_classes);
}

std::vector<Recognition> ImageClassifier::recognizeImage(const cv::Mat &image)
{
  // Resize image
  cv::Mat cropped;
  cv::resize(image, cropped, cv::Size(_input_size, _input_size), 0, 0, cv::INTER_AREA);

  // Preprocess the image data from 0~255 int to normalized float based
  // on the provided parameters
  _fdata.clear();
  for (int y = 0; y < cropped.rows; ++y)
  {
    for (int x = 0; x < cropped.cols; ++x)
    {
      cv::Vec3b color = cropped.at<cv::Vec3b>(y, x);
      color[0] = color[0] - (float)_image_mean / _image_std;
      color[1] = color[1] - (float)_image_mean / _image_std;
      color[2] = color[2] - (float)_image_mean / _image_std;

      _fdata.push_back(color[0]);
      _fdata.push_back(color[1]);
      _fdata.push_back(color[2]);

      cropped.at<cv::Vec3b>(y, x) = color;
    }
  }

  // Copy the input data into model
  _inference->feed(_input_name, _fdata, 1, _input_size, _input_size, 3);

  // Run the inference call
  _inference->run(_output_name);

  // Copy the output tensor back into the output array
  _inference->fetch(_output_name, _outputs);

  // Find the best classifications
  auto compare = [](const Recognition &lhs, const Recognition &rhs) {
    return lhs.confidence < rhs.confidence;
  };

  std::priority_queue<Recognition, std::vector<Recognition>, decltype(compare)> pq(compare);
  for (int i = 0; i < _num_classes; ++i)
  {
    if (_outputs[i] > _threshold)
    {
      pq.push(Recognition(_outputs[i], _labels[i]));
    }
  }

  std::vector<Recognition> results;
  int min = std::min(pq.size(), _max_results);
  for (int i = 0; i < min; ++i)
  {
    results.push_back(pq.top());
    pq.pop();
  }

  return results;
}
