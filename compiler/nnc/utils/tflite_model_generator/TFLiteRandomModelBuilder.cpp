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

#include <memory>
#include <random>

#include "TFLiteRandomModelBuilder.h"

namespace modelgen
{
static const char *modelFileOut = "model.tflite";
static const int notInitialized = -1;

static const char *const opNames[] = {
#define DEF_OPERATOR(Name, OpCode, TFLiteOpCode, CreatorFunc) Name,
#include "Operator.def"
#undef DEF_OPERATOR
};

static const tflite::BuiltinOperator internalOpCode2TFLiteOpCode[]{
#define DEF_OPERATOR(Name, OpCode, TFLiteOpCode, CreatorFunc) TFLiteOpCode,
#include "Operator.def"
#undef DEF_OPERATOR
};

void TFLiteModelSaver::saveModel()
{
  auto m = tflite::Model::Pack(_flatBufferBuilder, _model.get());
  tflite::FinishModelBuffer(_flatBufferBuilder, m);

  auto model_data = _flatBufferBuilder.GetBufferPointer();
  auto size = _flatBufferBuilder.GetSize();

  std::string fileName(modelFileOut);
  auto f = fopen(fileName.c_str(), "wb");
  assert(f != nullptr);
  auto rlen = fwrite(model_data, size, 1, f);
  assert(rlen == 1);
  fclose(f);
}

TFLiteRandomModelBuilder::TFLiteRandomModelBuilder() : RandomModelBuilder(), _model(new ModelT())
{
  std::fill_n(_mapOperatorCode, static_cast<long>(OpCodes::opCount), notInitialized);
}

void TFLiteRandomModelBuilder::convertTreeToModel(treebuilder::Tree *t)
{
  createInput(t);

  for (auto &op : t->opList)
  {
    addOperator(t, op.get());
  }
}

std::unique_ptr<ModelSaver> TFLiteRandomModelBuilder::createModelSaver()
{
  return std::unique_ptr<ModelSaver>(new TFLiteModelSaver(std::move(_model)));
}

/**
 * @todo Add support several inputs.
 */
void TFLiteRandomModelBuilder::createInput(treebuilder::Tree *t)
{
  assert(_model->subgraphs.empty() && "Subgraph is already created");

  std::unique_ptr<SubGraphT> subgraph(new SubGraphT);
  subgraph->inputs.push_back(0);
  subgraph->outputs.push_back(0);

  subgraph->tensors.push_back(createEmptyTensor(t->inputShapeTree, "input"));
  _operandTree2tensor.push_back(0); // it is same as: push_pack(subgraph->tensors.size() - 1);

  _model->subgraphs.push_back(std::move(subgraph));
  _model->description = "Random tflite model";
  _model->version = 3;
}

void TFLiteRandomModelBuilder::addOperator(treebuilder::Tree *t, treebuilder::Operation *op)
{
  assert(!_model->subgraphs.empty() && "Subgraph is not created");

  std::cout << "Add operator [" << opNames[static_cast<int32_t>(op->opcode)] << "] on the level [ "
            << op->levelOwner << " ]" << std::endl;
  _opCreators[static_cast<int32_t>(op->opcode)](t, op);
  _model->subgraphs[0]->outputs[0] = (*_model->subgraphs[0]->operators.rbegin())->outputs[0];
}

void TFLiteRandomModelBuilder::createLayerCONV_2D(treebuilder::Tree *t, treebuilder::Operation *op)
{
  std::string output_name(opNames[static_cast<int32_t>(OpCodes::opConv2d)]);
  output_name += "_" + std::to_string(_operatorCounts[static_cast<int32_t>(OpCodes::opConv2d)]);
  auto operator_ptr = createEmptyOperator(op);

  auto out_tensor_ptr = createEmptyTensor(op->outputShape, output_name.c_str());
  auto kernel_ptr = createTensorWthBuffer(op->kernelShape, "Kernel");
  auto bias_ptr = createTensorWthBuffer({op->outputShape[3]}, "bias");

  auto input_tensor_id = op->levelOwner == 0 ? _operandTree2tensor[0]
                                             : _operandTree2tensor[t->inputCnt + op->inputs[0]];

  operator_ptr->inputs.push_back(input_tensor_id);
  operator_ptr->inputs.push_back(static_cast<int32_t>(_model->subgraphs[0]->tensors.size()));
  _model->subgraphs[0]->tensors.push_back(std::move(kernel_ptr));
  operator_ptr->inputs.push_back(static_cast<int32_t>(_model->subgraphs[0]->tensors.size()));
  _model->subgraphs[0]->tensors.push_back(std::move(bias_ptr));

  auto output_tensor_id = static_cast<int32_t>(_model->subgraphs[0]->tensors.size());
  _operandTree2tensor.push_back(output_tensor_id);
  operator_ptr->outputs.push_back(output_tensor_id);
  _model->subgraphs[0]->tensors.push_back(std::move(out_tensor_ptr));

  operator_ptr->builtin_options.Set(tflite::Conv2DOptionsT());
  auto conv2D_opt = operator_ptr->builtin_options.AsConv2DOptions();
  conv2D_opt->stride_w = conv2D_opt->stride_h = 1;
  conv2D_opt->fused_activation_function =
      tflite::ActivationFunctionType::ActivationFunctionType_RELU6;

  _model->subgraphs[0]->operators.push_back(std::move(operator_ptr));
}

void TFLiteRandomModelBuilder::createLayerCONCATENATION(treebuilder::Tree *t,
                                                        treebuilder::Operation *op)
{
  std::string output_name(opNames[static_cast<int32_t>(OpCodes::opConcatenation)]);
  output_name +=
      "_" + std::to_string(_operatorCounts[static_cast<int32_t>(OpCodes::opConcatenation)]);
  auto operator_ptr = createEmptyOperator(op);

  auto out_tensor_ptr = createEmptyTensor(op->outputShape, output_name.c_str());

  std::cout << "Concatination inputs [ ";
  for (auto it : op->inputs)
  {
    std::cout << it << "/";
    auto input_tensor_id =
        op->levelOwner == 0 ? _operandTree2tensor[0] : _operandTree2tensor[t->inputCnt + it];
    std::cout << input_tensor_id << " ";
    operator_ptr->inputs.push_back(input_tensor_id);
  }
  std::cout << "]" << std::endl;

  auto output_tensor_id = static_cast<int32_t>(_model->subgraphs[0]->tensors.size());
  _operandTree2tensor.push_back(output_tensor_id);
  operator_ptr->outputs.push_back(output_tensor_id);
  _model->subgraphs[0]->tensors.push_back(std::move(out_tensor_ptr));

  operator_ptr->builtin_options.Set(tflite::ConcatenationOptionsT());
  auto concat_opt = operator_ptr->builtin_options.AsConcatenationOptions();
  concat_opt->fused_activation_function =
      tflite::ActivationFunctionType::ActivationFunctionType_RELU6;
  concat_opt->axis = 0;

  for (auto it : op->inputShape)
  {
    if (it == -1)
      break;
    concat_opt->axis++;
  }

  _model->subgraphs[0]->operators.push_back(std::move(operator_ptr));
}

void TFLiteRandomModelBuilder::createLayerDEPTHWISE_CONV_2D(treebuilder::Tree *t,
                                                            treebuilder::Operation *op)
{
  std::string output_name(opNames[static_cast<int32_t>(OpCodes::opDepthwiseConv2d)]);
  output_name +=
      "_" + std::to_string(_operatorCounts[static_cast<int32_t>(OpCodes::opDepthwiseConv2d)]);
  auto operator_ptr = createEmptyOperator(op);

  auto out_tensor_ptr = createEmptyTensor(op->outputShape, output_name.c_str());
  auto kernel_ptr = createTensorWthBuffer(op->kernelShape, "Kernel");
  auto bias_ptr = createTensorWthBuffer({op->outputShape[3]}, "bias");

  auto input_tensor_id = op->levelOwner == 0 ? _operandTree2tensor[0]
                                             : _operandTree2tensor[t->inputCnt + op->inputs[0]];

  operator_ptr->inputs.push_back(input_tensor_id);
  operator_ptr->inputs.push_back(static_cast<int32_t>(_model->subgraphs[0]->tensors.size()));
  _model->subgraphs[0]->tensors.push_back(std::move(kernel_ptr));
  operator_ptr->inputs.push_back(static_cast<int32_t>(_model->subgraphs[0]->tensors.size()));
  _model->subgraphs[0]->tensors.push_back(std::move(bias_ptr));

  auto output_tensor_id = static_cast<int32_t>(_model->subgraphs[0]->tensors.size());
  _operandTree2tensor.push_back(output_tensor_id);
  operator_ptr->outputs.push_back(output_tensor_id);
  _model->subgraphs[0]->tensors.push_back(std::move(out_tensor_ptr));

  operator_ptr->builtin_options.Set(tflite::DepthwiseConv2DOptionsT());
  auto depthwise_conv2d_opt = operator_ptr->builtin_options.AsDepthwiseConv2DOptions();
  depthwise_conv2d_opt->stride_w = depthwise_conv2d_opt->stride_h = 1;
  depthwise_conv2d_opt->depth_multiplier = 1;
  depthwise_conv2d_opt->fused_activation_function =
      tflite::ActivationFunctionType::ActivationFunctionType_RELU6;

  _model->subgraphs[0]->operators.push_back(std::move(operator_ptr));
}

void TFLiteRandomModelBuilder::createLayerX_POOL_2D(treebuilder::Tree *t,
                                                    treebuilder::Operation *op, OpCodes opcode)
{
  std::string output_name(opNames[static_cast<int32_t>(opcode)]);
  output_name += "_" + std::to_string(_operatorCounts[static_cast<int32_t>(opcode)]);
  auto operator_ptr = createEmptyOperator(op);

  auto out_tensor_ptr = createEmptyTensor(op->outputShape, output_name.c_str());

  auto input_tensor_id = op->levelOwner == 0 ? _operandTree2tensor[0]
                                             : _operandTree2tensor[t->inputCnt + op->inputs[0]];
  operator_ptr->inputs.push_back(input_tensor_id);

  auto output_tensor_id = static_cast<int32_t>(_model->subgraphs[0]->tensors.size());
  _operandTree2tensor.push_back(output_tensor_id);
  operator_ptr->outputs.push_back(output_tensor_id);
  _model->subgraphs[0]->tensors.push_back(std::move(out_tensor_ptr));

  /**
   * @todo generate random filter width/height.
   */
  operator_ptr->builtin_options.Set(tflite::Pool2DOptionsT());
  auto pool2d_opt = operator_ptr->builtin_options.AsPool2DOptions();
  pool2d_opt->stride_w = pool2d_opt->stride_h = 1;
  pool2d_opt->filter_width = pool2d_opt->filter_height = 3;

  _model->subgraphs[0]->operators.push_back(std::move(operator_ptr));
}

void TFLiteRandomModelBuilder::createLayerSOFTMAX(treebuilder::Tree *t, treebuilder::Operation *op)
{
  std::string output_name(opNames[static_cast<int32_t>(OpCodes::opSoftmax)]);
  output_name += "_" + std::to_string(_operatorCounts[static_cast<int32_t>(OpCodes::opSoftmax)]);
  auto operator_ptr = createEmptyOperator(op);

  auto out_tensor_ptr = createEmptyTensor(op->outputShape, output_name.c_str());

  auto input_tensor_id = op->levelOwner == 0 ? _operandTree2tensor[0]
                                             : _operandTree2tensor[t->inputCnt + op->inputs[0]];
  operator_ptr->inputs.push_back(input_tensor_id);

  auto output_tensor_id = static_cast<int32_t>(_model->subgraphs[0]->tensors.size());
  _operandTree2tensor.push_back(output_tensor_id);
  operator_ptr->outputs.push_back(output_tensor_id);
  _model->subgraphs[0]->tensors.push_back(std::move(out_tensor_ptr));

  operator_ptr->builtin_options.Set(tflite::SoftmaxOptionsT());
  auto softmax_opt = operator_ptr->builtin_options.AsSoftmaxOptions();
  softmax_opt->beta = _floatRand(_gen);

  _model->subgraphs[0]->operators.push_back(std::move(operator_ptr));
}

void TFLiteRandomModelBuilder::createLayerFULLY_CONNECTED(treebuilder::Tree *t,
                                                          treebuilder::Operation *op)
{
  std::string output_name(opNames[static_cast<int32_t>(OpCodes::opFullyConnected)]);
  output_name +=
      "_" + std::to_string(_operatorCounts[static_cast<int32_t>(OpCodes::opFullyConnected)]);
  auto operator_ptr = createEmptyOperator(op);

  auto out_tensor_ptr = createEmptyTensor(op->outputShape, output_name.c_str());
  auto kernel_ptr = createTensorWthBuffer(op->kernelShape, "Kernel");
  auto bias_ptr = createTensorWthBuffer({op->outputShape[3]}, "bias");

  auto input_tensor_id = op->levelOwner == 0 ? _operandTree2tensor[0]
                                             : _operandTree2tensor[t->inputCnt + op->inputs[0]];

  operator_ptr->inputs.push_back(input_tensor_id);
  operator_ptr->inputs.push_back(static_cast<int32_t>(_model->subgraphs[0]->tensors.size()));
  _model->subgraphs[0]->tensors.push_back(std::move(kernel_ptr));
  operator_ptr->inputs.push_back(static_cast<int32_t>(_model->subgraphs[0]->tensors.size()));
  _model->subgraphs[0]->tensors.push_back(std::move(bias_ptr));

  auto output_tensor_id = static_cast<int32_t>(_model->subgraphs[0]->tensors.size());
  _operandTree2tensor.push_back(output_tensor_id);
  operator_ptr->outputs.push_back(output_tensor_id);
  _model->subgraphs[0]->tensors.push_back(std::move(out_tensor_ptr));

  operator_ptr->builtin_options.Set(tflite::FullyConnectedOptionsT());
  auto fullcon_opt = operator_ptr->builtin_options.AsFullyConnectedOptions();
  fullcon_opt->fused_activation_function =
      tflite::ActivationFunctionType::ActivationFunctionType_RELU6;

  _model->subgraphs[0]->operators.push_back(std::move(operator_ptr));
}

std::unique_ptr<TensorT>
TFLiteRandomModelBuilder::createEmptyTensor(const std::vector<int32_t> &shape, const char *name)
{
  auto tensor_ptr = std::unique_ptr<TensorT>(new TensorT);

  tensor_ptr->type = tflite::TensorType_FLOAT32;
  tensor_ptr->name = name;
  tensor_ptr->shape = shape;
  tensor_ptr->buffer = static_cast<uint32_t>(_model->buffers.size());
  _model->buffers.push_back(std::unique_ptr<BufferT>(new BufferT));

  return tensor_ptr;
}

std::unique_ptr<TensorT>
TFLiteRandomModelBuilder::createTensorWthBuffer(const std::vector<int32_t> &shape, const char *name)
{
  auto tensor_ptr = createEmptyTensor(shape, name);

  size_t buffer_size = 1;
  for (auto s : shape)
  {
    buffer_size *= s;
  }
  buffer_size *= sizeof(float);

  _model->buffers[tensor_ptr->buffer]->data.resize(buffer_size);

  for (size_t i = 0; i < buffer_size; i += sizeof(float))
  {
    float val = _floatRand(_gen);
    memcpy(_model->buffers[tensor_ptr->buffer]->data.data() + i, &val, sizeof(float));
  }
  return tensor_ptr;
}

std::unique_ptr<OperatorT> TFLiteRandomModelBuilder::createEmptyOperator(treebuilder::Operation *op)
{
  auto operator_ptr = std::unique_ptr<OperatorT>(new OperatorT);
  auto opcode_id = _mapOperatorCode[static_cast<int32_t>(op->opcode)];
  auto tflite_opcode = internalOpCode2TFLiteOpCode[static_cast<int32_t>(op->opcode)];

  if (opcode_id == notInitialized)
  {
    auto opcode_ptr = std::unique_ptr<OperatorCodeT>(new OperatorCodeT);
    opcode_ptr->builtin_code = tflite_opcode;
    opcode_ptr->custom_code = tflite::EnumNamesBuiltinOperator()[tflite_opcode];
    opcode_id = static_cast<int32_t>(_model->operator_codes.size());
    _model->operator_codes.push_back(std::move(opcode_ptr));
    _mapOperatorCode[static_cast<int32_t>(op->opcode)] = opcode_id;
  }
  operator_ptr->opcode_index = static_cast<uint32_t>(opcode_id);
  _operatorCounts[static_cast<int32_t>(op->opcode)]++;

  return operator_ptr;
}

} // namespace modelgen
