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

#include "Options.h"
#include "Definitions.h"

#include <string>

namespace nnc
{
namespace cli
{

/**
 * Options for *compiler driver*
 */
Option<bool> Help(optname("--help, -h"), overview("print usage and exit"), false, optional(true));
Option<bool> caffeFrontend(optname("--caffe"), overview("treat input file as Caffe model"), false,
                           optional(true), optvalues(""), nullptr, separators(""),
#ifdef NNC_FRONTEND_CAFFE_ENABLED
                           showopt(true)
#else
                           showopt(false)
#endif // NNC_FRONTEND_CAFFE_ENABLED
);
Option<bool> onnxFrontend(optname("--onnx"), overview("treat input file as ONNX model"), false,
                          optional(true), optvalues(""), nullptr, separators(""),
#ifdef NNC_FRONTEND_ONNX_ENABLED
                          showopt(true)
#else
                          showopt(false)
#endif // NNC_FRONTEND_ONNX_ENABLED
);

Option<bool> caffe2Frontend(optname("--caffe2"),
                            overview("treat input file as Caffe2 model (predict_net.pb)"), false,
                            optional(false), optvalues(""), nullptr, separators(""),
#ifdef NNC_FRONTEND_CAFFE2_ENABLED
                            showopt(true),
#else
                            showopt(false),
#endif // NNC_FRONTEND_CAFFE2_ENABLED
                            IOption::Group::caffe2);

Option<std::vector<int>> inputShapes(optname("--input-shape"), overview("Shape of caffe2 input"),
                                     std::vector<int>{}, optional(false), optvalues(""), nullptr,
                                     separators(""),
#ifdef NNC_FRONTEND_CAFFE2_ENABLED
                                     showopt(true),
#else
                                     showopt(false),
#endif // NNC_FRONTEND_CAFFE2_ENABLED
                                     IOption::Group::caffe2);

Option<std::string> initNet(optname("--init-net"),
                            overview("path to Caffe2 model weights (init_net.pb)"), std::string(),
                            optional(false), optvalues(""), checkInFile, separators(""),
#ifdef NNC_FRONTEND_CAFFE2_ENABLED
                            showopt(true),
#else
                            showopt(false),
#endif // NNC_FRONTEND_CAFFE2_ENABLED
                            IOption::Group::caffe2);

Option<bool> tflFrontend(optname("--tflite"),
                         overview("treat input file as Tensor Flow Lite model"), false,
                         optional(true), optvalues(""), nullptr, separators(""),
#ifdef NNC_FRONTEND_TFLITE_ENABLED
                         showopt(true)
#else
                         showopt(false)
#endif // NNC_FRONTEND_TFLITE_ENABLED
);
Option<std::string>
    target(optname("--target"),
           overview("select target language to emit for given architecture."
                    "Valid values are '" NNC_TARGET_ARM_CPP "', '" NNC_TARGET_X86_CPP
                    "', '" NNC_TARGET_ARM_GPU_CPP "', '" NNC_TARGET_INTERPRETER "'"),
           std::string(), optional(false),
           optvalues(NNC_TARGET_ARM_CPP "," NNC_TARGET_X86_CPP "," NNC_TARGET_ARM_GPU_CPP
                                        "," NNC_TARGET_INTERPRETER),
           nullptr, separators("="));

/**
 * Options for *frontend*
 */
Option<std::string> inputFile(optname("--nnmodel, -m"),
                              overview("specify input file with serialized NN models"),
                              std::string(), optional(false), optvalues(""), checkInFile);

/**
 * Options for *optimizer*
 */
Option<bool> doOptimizationPass(optname("-O"), overview("whether to optimize model or not"), false,
                                optional(true), optvalues(""), nullptr, separators(""),
                                showopt(true));

Option<bool> dumpGraph(optname("--dump, -D"),
                       overview("dump graph to dot files after optimization passes"), false,
                       optional(true), optvalues(""), nullptr, separators(""), showopt(true));

/**
 * Options for *backend*
 */
// options for soft backend
Option<std::string> artifactName(optname("--output, -o"), overview("specify name for output files"),
                                 "nnmodel", optional(true), optvalues(""), checkOutFile);
Option<std::string> artifactDir(optname("--output-dir, -d"),
                                overview("specify directory for output files"),
                                ".", // default is current directory
                                optional(true), optvalues(""), checkOutDir, separators("="));

/**
 * Options for *interpreter*
 */
Option<std::string> interInputDataDir(optname("--input-data-dir"),
                                      overview("specify directory with binary files "
                                               "containing the input data for the model "
                                               "(one file for each input with the same name)"),
                                      ".", // default is current directory
                                      optional(true), optvalues(""), checkInDir);

} // namespace cli
} // namespace nnc
