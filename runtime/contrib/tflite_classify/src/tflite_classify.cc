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

#include <iostream>
#include <filesystem>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int main(const int argc, char **argv)
{
  const std::string MODEL_FILE = "tensorflow_inception_graph.tflite";
  const std::string LABEL_FILE = "imagenet_comp_graph_label_strings.txt";

  const std::string INPUT_NAME = "input";
  const std::string OUTPUT_NAME = "output";
  const int INPUT_SIZE = 224;
  const int IMAGE_MEAN = 117;
  const int IMAGE_STD = 1;
  const int OUTPUT_SIZE = 1008;

  const int FRAME_WIDTH = 640;
  const int FRAME_HEIGHT = 480;

  const bool use_nnapi = nnfw::misc::EnvVar("USE_NNAPI").asBool(false);
  const bool debug_mode = nnfw::misc::EnvVar("DEBUG_MODE").asBool(false);

  std::cout << "USE_NNAPI : " << use_nnapi << std::endl;
  std::cout << "DEBUG_MODE : " << debug_mode << std::endl;

  std::cout << "Model : " << MODEL_FILE << std::endl;
  std::cout << "Label : " << LABEL_FILE << std::endl;

  if (!fs::exists(MODEL_FILE))
  {
    std::cerr << "model file not found: " << MODEL_FILE << std::endl;
    exit(1);
  }

  if (!fs::exists(LABEL_FILE))
  {
    std::cerr << "label file not found: " << LABEL_FILE << std::endl;
    exit(1);
  }

  // Create ImageClassifier
  std::unique_ptr<ImageClassifier> classifier(new ImageClassifier(
    MODEL_FILE, LABEL_FILE, INPUT_SIZE, IMAGE_MEAN, IMAGE_STD, INPUT_NAME, OUTPUT_NAME, use_nnapi));

  // Cam setting
  cv::VideoCapture cap(0);
  cv::Mat frame;

  // Initialize camera
  cap.set(CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);
  cap.set(CV_CAP_PROP_FPS, 5);

  std::vector<Recognition> results;
  clock_t begin, end;
  while (cap.isOpened())
  {
    // Get image data
    if (!cap.read(frame))
    {
      std::cout << "Frame is null..." << std::endl;
      break;
    }

    if (debug_mode)
    {
      begin = clock();
    }
    // Recognize image
    results = classifier->recognizeImage(frame);
    if (debug_mode)
    {
      end = clock();
    }

    // Show result data
    std::cout << std::endl;
    if (results.size() > 0)
    {
      for (int i = 0; i < results.size(); ++i)
      {
        std::cout << results[i].title << ": " << results[i].confidence << std::endl;
      }
    }
    else
    {
      std::cout << "." << std::endl;
    }
    if (debug_mode)
    {
      std::cout << "Frame: " << FRAME_WIDTH << "x" << FRAME_HEIGHT << std::endl;
      std::cout << "Crop: " << INPUT_SIZE << "x" << INPUT_SIZE << std::endl;
      std::cout << "Inference time(ms): " << ((end - begin) / (CLOCKS_PER_SEC / 1000)) << std::endl;
    }
  }

  cap.release();

  return 0;
}
