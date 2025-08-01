/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_EXPERIMENTAL_H__
#define __NNFW_EXPERIMENTAL_H__

#include "nnfw.h"

#ifdef __cplusplus
extern "C" {
#endif

// Used for custom kernel development

/*
 * operand type, used only for custom operations
 */
typedef struct
{
  nnfw_tensorinfo type;
  void *allocation;
} nnfw_operand;

/*
 * Used as input to custom operation eval function
 */
typedef struct
{
  size_t ninputs;
  nnfw_operand *inputs;

  size_t noutputs;
  nnfw_operand *outputs;
} nnfw_custom_kernel_params;

/*
 * Custom kernel evaluation function
 *
 * param[in] params custom operation parameters
 * param[in] userdata pointer to user-specified buffer( kernel instance specific )
 */
typedef void (*nnfw_custom_eval)(nnfw_custom_kernel_params *params, char *userdata,
                                 size_t userdata_size);

/*
 * custom operation registration info
 */
typedef struct
{
  nnfw_custom_eval eval_function;
} custom_kernel_registration_info;

NNFW_STATUS nnfw_register_custom_op_info(nnfw_session *session, const char *id,
                                         custom_kernel_registration_info *info);

/**
 * @brief Get the input tensor index by name
 *
 * This function finds an input tensor of the given name.
 * If found, the index value is set to the address that @c index points to, and returns
 * @c NNFW_STATUS_NO_ERROR. Otherwise, @c index is unchanged and returns @c NNFW_STATUS_ERROR .
 *
 * @note If two or more input tensors are of the same name, the one with the lowest index is always
 *       returned.
 *
 * @param[in]  session    the session object
 * @param[in]  tensorname the name of the tensor to find, a null terminated char pointer string
 * @param[out] index      the index to be ret
 * @return     @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_input_tensorindex(nnfw_session *session, const char *tensorname, uint32_t *index);

/**
 * @brief Get the input tensor index by name
 *
 * This function finds an input tensor of the given name.
 * If found, the index value is set to the address that @c index points to, and returns
 * @c NNFW_STATUS_NO_ERROR. Otherwise, @c index is unchanged and returns @c NNFW_STATUS_ERROR .
 *
 * @note If two or more input tensors are of the same name, the one with the lowest index is always
 *       returned.
 *
 * @param[in]  session    the session object
 * @param[in]  tensorname the name of the tensor to find, a null terminated char pointer string
 * @param[out] index      the index to be ret
 * @return     @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_output_tensorindex(nnfw_session *session, const char *tensorname, uint32_t *index);

/**
 * @brief Set the backend for each operation in the session
 *
 * This function assigns backends (acl_cl, acl_neon, cpu) to each operation in the session.
 * If successful,the function returns @c NNFW_STATUS_NO_ERROR. Otherwise, the function returns
 * @c NNFW_STATUS_ERROR.
 *
 * @note The argument specifying backends must be in the format
 *       "OP_BACKEND_MAP=\"0=acl_cl;1=cpu;2=acl_cl\"".
 *
 * @param[in]  session          the session object
 * @param[in]  backend_settings String containing backend assignments indexed by operation sequence
 * @return     @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_backends_per_operation(nnfw_session *session, const char *backend_settings);

/**
 * @brief Prepare session to be ready for inference
 *
 * This phase may finalize model compilation, scheduling, and additional settings.
 *
 * @deprecated Deprecated since 1.22.1
 *
 * @param session the session to be prepared
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_prepare_pipeline(nnfw_session *session, const char *map_file_path = nullptr);

/**
 * @brief     Set input buffer
 *
 * This function must be called after {@link nnfw_prepare_pipeline}, \p inputs given to this
 * function can be reused for many inferences. \p lengths must be greater or equal than the operand
 * requires. if you give empty \p inputs to this function, then this function will join all threads.
 *
 * @deprecated Deprecated since 1.22.1
 *
 * @param[in] session Session to the input is to be set
 * @param[in] inputs  Raw buffers for input, it must be \p std::vector<void *> type pointer for
 * multiple input model
 * @param[in] lengths Size of bytes of input buffers, it must be \p std::vector<uint32_t> type
 * pointer for multiple input model
 *
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_push_pipeline_input(nnfw_session *session, void *inputs, void *lengths);

/**
 * @brief       Get last outputs of partitioned model in session
 *
 * This function must be called after {@link nnfw_prepare_pipeline}, \p outputs given to this
 * function must be cleared for memory management.
 *
 * @deprecated Deprecated since 1.22.1
 *
 * @param[in]   session Session from last outputs is to be extracted
 * @param[out]  outputs Raw buffer for outputs, it must be \p std::vector<void *> type pointer for
 * multiple output model
 *
 * @return      @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_pop_pipeline_output(nnfw_session *session, void *outputs);

/**
 * @brief Set the type of an input
 *
 * User can call this function to set the type of an input.
 * Then user can pass input data of that type.
 * If it is not called, runtime will infer the type from the model.
 * This function should be called after {@link nnfw_load_model_from_file} and
 * before {@link nnfw_prepare}.
 * Now only NNFW_TYPE_TENSOR_FLOAT32 is supported. If other types are passed,
 * runtime will return error.
 *
 * @param[in] session session from input is to be extracted
 * @param[in] index   index of input to be set (0-indexed)
 * @param[in] type    type to set to target input. This can be NNFW_TYPE_TENSOR_FLOAT32 only.
 *
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_input_type(nnfw_session *session, uint32_t index, NNFW_TYPE type);

/**
 * @brief Set the type of an output
 *
 * User can call this function to set the type of an output.
 * Then user can pass output data of that type.
 * If it is not called, runtime will output the type from the model.
 * This function should be called after {@link nnfw_load_model_from_file} and
 * before {@link nnfw_prepare}.
 * Now only NNFW_TYPE_TENSOR_FLOAT32 is supported. If other types are passed,
 * runtime will return error.
 *
 * @param[in] session session from output is to be extracted
 * @param[in] index   index of output to be set (0-indexed)
 * @param[in] type    type to set to target output. This can be NNFW_TYPE_TENSOR_FLOAT32 only.
 *
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_output_type(nnfw_session *session, uint32_t index, NNFW_TYPE type);

/**
 *  Training C APIs
 *
 * Training APIs are designed to be used in the following order for training
 * 1. nnfw_train_prepare
 * 2. nnfw_train_set_input, nnfw_train_set_expected for inputs & expected outputs
 * 3. nnfw_train
 * 4. nnfw_train_get_loss
 *
 * If you want to inference after training with the same session, you can use the following order
 * 1. nnfw_set_input
 * 2. nnfw_set_output
 * 3. nnfw_run
 */

//////////////////////////////////////////////
// Essential APIs for training
//////////////////////////////////////////////
typedef enum
{
  NNFW_TRAIN_LOSS_UNDEFINED = 0,
  NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR = 1,
  NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY = 2,
} NNFW_TRAIN_LOSS;

typedef enum
{
  /** Undefined */
  NNFW_TRAIN_LOSS_REDUCTION_UNDEFINED = 0,
  /** Scalar sum divided by number of elements in losses */
  NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE = 1,
  /** Scalar sum of weighted losses */
  NNFW_TRAIN_LOSS_REDUCTION_SUM = 2,
} NNFW_TRAIN_LOSS_REDUCTION;

typedef enum
{
  NNFW_TRAIN_OPTIMIZER_UNDEFINED = 0,
  NNFW_TRAIN_OPTIMIZER_SGD = 1,
  NNFW_TRAIN_OPTIMIZER_ADAM = 2,
} NNFW_TRAIN_OPTIMIZER;

typedef struct nnfw_loss_info
{
  NNFW_TRAIN_LOSS loss;
  NNFW_TRAIN_LOSS_REDUCTION reduction_type;
} nnfw_loss_info;

/**
 * @brief Special values of num_of_trainable_ops.
 *        Positive values are used to indicate layers to be trained from the back of the graph.
 */
typedef enum
{
  /** Error value of number of trainable ops */
  NNFW_TRAIN_TRAINABLE_INCORRECT_STATE = -2,
  /** All layers will be trained */
  NNFW_TRAIN_TRAINABLE_ALL = -1,
  /** No layer will be trained */
  NNFW_TRAIN_TRAINABLE_NONE = 0,

} NNFW_TRAIN_NUM_OF_TRAINABLE_OPS_SPECIAL_VALUES;

/**
 * @brief Training information to prepare training
 * @todo  Add more training information
 *        (e.g. optimizer, loss function, ...)
 */
typedef struct nnfw_train_info
{
  /** Learning rate */
  float learning_rate = 0.001f;
  /** Batch size */
  uint32_t batch_size = 1;
  /** loss info
   * Note that you don't need to worry about whether the model you use does not include softmax
   * when you try to use NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY. Using
   * NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY will ensure that the predicted input of loss is
   * the result of performing softmax once regardless of whether the output of the model is
   * the result of softmax or not.
   */
  nnfw_loss_info loss_info{.loss = NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR,
                           .reduction_type = NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE};
  /** optimizer type */
  NNFW_TRAIN_OPTIMIZER opt = NNFW_TRAIN_OPTIMIZER_SGD;

  /** Number of layers to be trained from the back of the graph.
   *  Note that some values have special meaning. "-1" means that all layers will be trained.
   * "0" means that no layer will be trained. Negative value less than -1 means error.
   *  The special values are collected in NNFW_TRAIN_NUM_OF_TRAINABLE_OPS_SPECIAL_VALUES enum.
   */
  int32_t num_of_trainable_ops = NNFW_TRAIN_TRAINABLE_NONE;
} nnfw_train_info;

/**
 * @brief Get training information
 * @note  This function should be called after calling {@link nnfw_load_model_from_file}
 *
 *        For the field which is not set in training information, it returns training information
 *        filled with default value. The default value of each field is as follows :
 *        learning_rate = 0.0f, batch_size = 0, *_UNDEF for other enums
 *
 * @param[in]   session   The session to get training information
 * @param[out]  info      Training information
 *
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_get_traininfo(nnfw_session *session, nnfw_train_info *info);

/**
 * @brief Set training information
 * @note  This function should be called after calling {@link nnfw_load_model_from_file}
 *        and before calling {@link nnfw_train_prepare}
 *
 * @param[in] session The session to be set training information
 * @param[in] info    The training information
 *
 *  @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_set_traininfo(nnfw_session *session, const nnfw_train_info *info);

/**
 * @brief Prepare session to be ready for training
 * @note  The session will be entered into training mode
 *
 *        If training info is NOT set in session, this function returns @c NNFW_STATUS_ERROR .
 *        You should set training info using {@link nnfw_train_set_traininfo}.
 *
 * @param[in] session The session to be prepared for training
 *
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_prepare(nnfw_session *session);

/**
 * @brief Set training input
 * @note  This function should be called after {@link nnfw_train_prepare}
 *
 * @param[in] session     The session to be set training inputs and expected model outputs
 * @param[in] index       The index of training input
 * @param[in] input       The input buffers for training
 * @param[in] input_info  The shape and type of input buffer
 *                        If it is nullptr, it will not change shape and batch size
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_set_input(nnfw_session *session, uint32_t index, const void *input,
                                 const nnfw_tensorinfo *input_info);

/**
 * @brief Set training expected output
 * @note  This function should be called after {@link nnfw_train_prepare}
 *
 * @param session       The session to be set training inputs and expected model outputs
 * @param index         The index of training expected output
 * @param expected      The expected buffers for training
 * @param expected_info The shape and type of expected buffer
 *                      If it is nullptr, it will not change shape and batch size
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_set_expected(nnfw_session *session, uint32_t index, const void *expected,
                                    const nnfw_tensorinfo *expected_info);

/**
 * @brief Set training output buffer
 *
 * This function must be called after {@link nnfw_train_prepare}, \p buffer given to this function
 * can be reused for training. \p length must be greater or equal than the operand requires.
 * An output operand can have unspecified shape and deduced dynamically during the execution. You
 * must provide \p buffer large enough.
 *
 * @param[in]   session Session from inference output is to be extracted
 * @param[in]   index   Index of output to be set (0-indexed)
 * @param[in]   type    Type of the output
 * @param[out]  buffer  Raw buffer for output
 * @param[in]   length  Size of bytes of output buffer
 *
 * @return      @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_set_output(nnfw_session *session, uint32_t index, NNFW_TYPE type,
                                  void *buffer, size_t length);

/**
 * @brief Train the model
 * @note  This function should be called after {@link nnfw_train_set_input} and
 *        {@link nnfw_train_set_expected} for each input and expected output
 *
 *        In order to use \p update_weights as false, it should be called after
 *        {@link nnfw_train_set_output}.
 *
 * @param[in] session The session to be trained
 * @param[in] update_weights If true, update weights of the model
 *                           If false, do not update weights of the model (for validation)
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train(nnfw_session *session, bool update_weights);

/**
 * @brief Get loss value for expected output
 * @note  This function should be called after {@link nnfw_train}
 *
 * @param[in]   session The session to get loss value
 * @param[in]   index   The index of loss value [0, number of expected outputs)
 * @param[out]  loss    The loss value
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_get_loss(nnfw_session *session, uint32_t index, float *loss);

/**
 * @brief Export circle model
 * @note  This function should be called on training mode
 *        This function should be called after {@link nnfw_train}
 *
 * @param[in] session The session to export inference model
 * @param[in] path    The path to export inference model
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_export_circle(nnfw_session *session, const char *path);

/**
 * @brief Import circle checkpoint
 * @note  This function should be called on training mode
 *        This function should be called before {@link nnfw_train}
 *
 * @param[in] session The session to export a checkpoint
 * @param[in] path    The path to export a checkpoint
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_import_checkpoint(nnfw_session *session, const char *path);

/**
 * @brief Export circle checkpoint
 * @note  This function should be called on training mode
 *        This function should be called after {@link nnfw_train}
 *
 * @param[in] session The session to export a checkpoint
 * @param[in] path    The path to export a checkpoint
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_export_checkpoint(nnfw_session *session, const char *path);

//////////////////////////////////////////////
// Optional APIs for training
//////////////////////////////////////////////

/**
 * @brief Get the training model input information
 * @note  This function should be called after {@link nnfw_train_prepare}
 *
 * @param[in]   session The session to get the training model input information
 * @param[in]   index   The index of training model input
 * @param[out]  info    The shape and type of training model input
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_input_tensorinfo(nnfw_session *session, uint32_t index,
                                        nnfw_tensorinfo *info);

/**
 * @brief Get the training model expected output information
 * @note  This function should be called after {@link nnfw_train_prepare}
 *
 * @param[in]   session The session to get the training model expected output information
 * @param[in]   index   The index of training model expected output
 * @param[out]  info    The shape and type of training model expected output
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_train_expected_tensorinfo(nnfw_session *session, uint32_t index,
                                           nnfw_tensorinfo *info);

//////////////////////////////////////////////
// Not planned to be implemented
//////////////////////////////////////////////

/**
 * @brief Convert between training mode and inference mode
 * @note  This function should be called after {@link nnfw_train} or {@link nnfw_prepare}
 *
 * @param[in] session The session to convert training mode to inference mode
 * @param[in] train   If false, convert training model to inference model
 *                    If true, convert inference model to training model
 * @return  @c NNFW_STATUS_NO_ERROR if successful
 */
// NNFW_STATUS nnfw_set_training_mode(nnfw_session *session, bool train);

/**
 *  On-Device Quantization APIs
 *
 * On-Device Quantization APIs are designed to be used in the following order
 * 1. nnfw_set_quantization_type
 * 2. nnfw_set_quantized_model_path
 * 3. nnfw_quantize
 *
 * You should use Quantization APIs after {@link nnfw_load_model_from_file},
 * before {@link nnfw_prepare} and {@link nnfw_set_input_tensorinfo}.
 */

/**
 * @brief quantization type
 */
typedef enum
{
  /** default value: type not set */
  NNFW_QUANTIZE_TYPE_NOT_SET,
  /** asymmetric quantization with a scale and zero point */
  NNFW_QUANTIZE_TYPE_U8_ASYM,
  /** symmetric quantization with a scale only */
  NNFW_QUANTIZE_TYPE_I16_SYM,
  /** weight-only int8 symmetric quantization */
  NNFW_QUANTIZE_TYPE_WO_I8_SYM,
  /** weight-only int16 symmetric quantization */
  NNFW_QUANTIZE_TYPE_WO_I16_SYM,

} NNFW_QUANTIZE_TYPE;

/**
 * @brief Set quantization type
 *
 * This function should be called before {@link nnfw_quantize} is invoked.
 *
 * @param[in] session nnfw_session to set quantization type
 * @param[in] pref @c NNFW_QUANTIZE_TYPE
 * @return    @c NNFW_STATUS_NO_ERROR if successful,
 *            @c NNFW_STATUS_UNEXPECTED_NULL if session is null,
 *            otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_set_quantization_type(nnfw_session *session, NNFW_QUANTIZE_TYPE qtype);

/**
 * @brief Set exported quantized model path
 *
 * This function should be called before {@link nnfw_quantize} is invoked.
 *
 * TODO: If this function is not called, quantized model will not be exported
 *
 * @param[in] session nnfw_session to set quantized model path
 * @param[in] path    Quantized model path
 * @return    @c NNFW_STATUS_NO_ERROR if successful, otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_set_quantized_model_path(nnfw_session *session, const char *path);

/**
 * @brief Quantize circle model
 *
 * @param[in] session nnfw_session to quantize
 * @return    @c ODC_STATUS_NO_ERROR if successful, otherwise return @c ODC_STATUS_ERROR
 */
NNFW_STATUS nnfw_quantize(nnfw_session *session);

/**
 * @brief Preference for target-dependent code generation
 */
typedef enum
{
  /** Use the default configuration */
  NNFW_CODEGEN_PREF_DEFAULT,
  // TODO Support Traffic and Cycle code generation preference
  /** Do best efforts to generate target-dependent code for performance */
  NNFW_CODEGEN_PREF_PERFORMANCE_FIRST,
  /** Do best efforts to generate target-dependent code for reducing host memory usage */
  NNFW_CODEGEN_PREF_MEMORY_FIRST,
  /** Do best efforts to generate target-dependent code for reducing compilation time */
  NNFW_CODEGEN_PREF_COMPILE_TIME_FIRST,
} NNFW_CODEGEN_PREF;

/**
 * @brief Set exported codegen model path
 *
 * This function should be called before {@link nnfw_codegen} is invoked.
 *
 * @param[in] session nnfw_session to set codegen model path
 * @param[in] path    Target-dependent model path
 * @return    @c NNFW_STATUS_NO_ERROR if successful, otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_set_codegen_model_path(nnfw_session *session, const char *path);

/**
 * @brief Generate target-dependent code
 *
 * This function opens a dynamic shared object. It searches for the object as flollows
 * ld.so(8) search rules. If the {@link nnfw_set_codegen_model_path} is not called before
 * this function, the codegen model path is automatically defined and used using the same
 * directory of the original model/package with the target backend extension.
 *
 * @param[in] session nnfw_session the session which contains information about compilation
 * @param[in] target  Target backend to generate code
 *                    This target string will be used to find a backend library.
 *                    The name of target backend library should follow the following rules:
 *                      'lib' +  {backend extension} + '-gen' + {lib extension}
 *                    And the target string should be a name except 'lib' and {lib extension}.
 *                    For example, if the backend extension is 'aaa', the backend library should
 *                    be 'libaaa-gen.so', and the target string should be 'aaa-gen'.
 * @param[in] pref @c NNFW_CODEGEN_PREF
 * @return    @c NNFW_STATUS_NO_ERROR if successful, otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_codegen(nnfw_session *session, const char *target, NNFW_CODEGEN_PREF pref);

/**
 * @brief  Set MinMax records count in auto compilation mode with on-device compiler
 *
 * This function set MinMax records count for quantization in auto compilation mode.
 * To enable automatic compilation mode, use  {@link nnfw_run_with_auto_compilation}
 *
 * @param[in] session nnfw_session
 * @param[in] minmax_records_count    minmax records count
 * @return    @c NNFW_STATUS_NO_ERROR if successful, otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_set_odc_param_minmax_records_count(nnfw_session *session,
                                                    int minmax_records_count);

/**
 * @brief  Delete MinMax file for on-device compiler
 *
 * @param[in] session nnfw_session
 * @return    @c NNFW_STATUS_NO_ERROR if successful, otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_odc_delete_minmax_file(nnfw_session *session);

/**
 * @brief  Run inference with auto compilation
 *
 * This function runs inference float model with automatic compilation and
 * replaces the original model with a quantized or compiled model inside.
 *
 * During the inference, the minmax statistics is collected and after that
 * quantization is performed.
 * If quantization was successful, try to code generating for target backend,
 * otherwise run original float model.
 *
 * If compilation was successful, run compiled model, otherwise run quantized model.
 *
 * Auto compilation uses on-device compiler (ODC), and ODC provides
 * quantization and compilation functionality.
 *
 * Additionally the following parameters should be set up :
 * 1. Quantization type {@link nnfw_set_quantization_type }
 * 2. Quantized model path {@link  nnfw_set_quantized_model_path }
 * 3. Minmax records threshold for quantization {@link nnfw_set_odc_param_minmax_records_count }
 * 4. File with minMax statistics can be removed by {@link nnfw_odc_delete_minmax_file}
 * 5. Compiled model path {@link  nnfw_set_codegen_model_path}
 *
 * Model is loaded by {@link nnfw_load_model_from_file},
 * session is prepared for inference by {@link nnfw_prepare},
 * set input and output float buffers by {@link nnfw_set_input} and {@link nnfw_set_output}.
 * This function must be called after model loading, preparation, and buffer setup are complete.
 *
 * After auto compilation, quantized model still uses float input/output buffer
 * and cast them to quantized type in runtime automatically.
 *
 * @param[in] session nnfw_session
 * @param[in] target  Target backend to generate code as in {@link nnfw_codegen}
 * @param[in] pref @c NNFW_CODEGEN_PREF

 * @return    @c NNFW_STATUS_NO_ERROR if successful, otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_run_with_auto_compilation(nnfw_session *session, const char *target,
                                           NNFW_CODEGEN_PREF pref);

//////////////////////////////////////////////
// APIs for configuration
//////////////////////////////////////////////

/**
 * @brief Configuration key for prepare (compile and schedule)
 */
typedef enum
{
  /**
   * Prepare to dump execution time profile file (not require value setting)
   * TODO: Use workspace
   */
  NNFW_PREPARE_CONFIG_PROFILE,
  /**
   * Enable internal allocation for model outputs instead of using external buffer
   */
  NNFW_ENABLE_INTERNAL_OUTPUT_ALLOC,
} NNFW_PREPARE_CONFIG;

/**
 * @brief      Set prepare configuration
 *
 * This function set prepare configuration to decide additional compiling and scheduing feature.
 * If you enable configuration to prepare dumping execution data into workspace,
 * refer {@link nnfw_set_workspace} to use workspace directory.
 *
 * @param[in] session nnfw_session to set prepare configuration
 * @param[in] key     prepare configuration key
 * @param[in] value   prepare configuration value
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_prepare_config(nnfw_session *session, NNFW_PREPARE_CONFIG key,
                                    const char *value);

/**
 * @brief     Reset prepare configurations
 *
 * This function reset all prepare configuration.
 *
 * @param[in] session nnfw_session to reset all prepare configurations
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_reset_prepare_config(nnfw_session *session);

/**
 * @brief Configuration key for execution
 */
typedef enum
{
  /** Dump minmax data for each layers to workspace (not require value setting) */
  NNFW_RUN_CONFIG_DUMP_MINMAX,
  /** Dump execution event file to workspace (not require value setting) */
  NNFW_RUN_CONFIG_TRACE,
  /**
   * Dump execution time profile file (not require value setting)
   *
   * You should set prepare configuration {@link NNFW_PREPARE_CONFIG_PROFILE} before prepare.
   * Otherwise, this configuration will be ignored.
   *
   * TODO: Use workspace
   */
  NNFW_RUN_CONFIG_PROFILE,
} NNFW_RUN_CONFIG;

/**
 * @brief     Set execution (run or train) configuration
 *
 * This function set execution configuration to dump execution data to workspace.
 * If you enable configuration to dump execution data into workspace and want to change workspace,
 * refer {@link nnfw_set_workspace} to use workspace directory.
 *
 * @param[in] session nnfw_session to set execution configuration
 * @param[in] key     execution configuration key
 * @param[in] value   execution configuration value if needed, otherwise set NULL
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_execute_config(nnfw_session *session, const NNFW_RUN_CONFIG key,
                                    const char *value);

/**
 * @brief     Reset execution (run or train) configurations
 *
 * This function reset all execution configuration.
 *
 * @param[in] session nnfw_session to reset all execution configurations
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_reset_execute_config(nnfw_session *session);

#ifdef __cplusplus
}
#endif

#endif // __NNFW_EXPERIMENTAL_H__
