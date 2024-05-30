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

/**
 * @file  nnfw.h
 * @brief This file describes runtime API
 */
#ifndef __NNFW_H__
#define __NNFW_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Session to query with runtime
 *
 * <p>nnfw_session is started and passed by calling {@link nnfw_create_session}.
 * Each session has its own inference environment, such as model to inference, backend usage, etc.
 *
 * <p>Load model by calling {@link nnfw_load_model_from_file}
 *
 * <p>After loading, prepare inference by calling {@link nnfw_prepare}.
 * Application can set runtime environment before prepare by calling
 * {@link nnfw_set_available_backends} and {@link nnfw_set_op_backend}, and it is optional.
 *
 * <p>Application can inference by calling {@link nnfw_run}.
 * Before inference, application has responsibility to set input tensor to set input data by calling
 * {@link nnfw_set_output}, and output tensor to get output by calling {@link nnfw_set_input}
 *
 * <p>To support input and output setting, application can get
 * input and output tensor information by calling<ul>
 * <li>{@link nnfw_input_size}</li>
 * <li>{@link nnfw_output_size}</li>
 * <li>{@link nnfw_input_tensorinfo}</li>
 * <li>{@link nnfw_output_tensorinfo}</li>
 * </ul>
 *
 * <p>Application can inference many times using one session,
 * but next inference can do after prior inference end
 *
 * <p>Application cannot use muitiple model using one session
 */
typedef struct nnfw_session nnfw_session;

/**
 * @brief Tensor types
 *
 * The type of tensor represented in {@link nnfw_tensorinfo}
 */
typedef enum
{
  /** A tensor of 32 bit floating point */
  NNFW_TYPE_TENSOR_FLOAT32 = 0,
  /** A tensor of 32 bit signed integer */
  NNFW_TYPE_TENSOR_INT32 = 1,
  /**
   * A tensor of 8 bit unsigned integers that represent real numbers.
   *
   * real_value = (integer_value - zeroPoint) * scale.
   */
  NNFW_TYPE_TENSOR_QUANT8_ASYMM = 2,
  /** A tensor of boolean */
  NNFW_TYPE_TENSOR_BOOL = 3,

  /** A tensor of 8 bit unsigned integer */
  NNFW_TYPE_TENSOR_UINT8 = 4,

  /** A tensor of 64 bit signed integer */
  NNFW_TYPE_TENSOR_INT64 = 5,

  /**
   * A tensor of 8 bit signed integers that represent real numbers.
   *
   * real_value = (integer_value - zeroPoint) * scale.
   */
  NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED = 6,

  /**
   * A tensor of 16 bit signed integers that represent real numbers.
   *
   * real_value = (integer_value - zeroPoint) * scale.
   *
   * Forced to have zeroPoint equal to 0.
   */
  NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED = 7,

} NNFW_TYPE;

/**
 * @brief Result values returned from a call to an API function
 */
typedef enum
{
  /** Successful */
  NNFW_STATUS_NO_ERROR = 0,
  /**
   * An error code for general use.
   * Mostly used when there is no specific value for that certain situation.
   */
  NNFW_STATUS_ERROR = 1,
  /** Unexpected null argument is given. */
  NNFW_STATUS_UNEXPECTED_NULL = 2,
  /** When a function was called but it is not valid for the current session state. */
  NNFW_STATUS_INVALID_STATE = 3,
  /** When it is out of memory */
  NNFW_STATUS_OUT_OF_MEMORY = 4,
  /** When it was given an insufficient output buffer */
  NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE = 5,
  /** When API is deprecated */
  NNFW_STATUS_DEPRECATED_API = 6,
} NNFW_STATUS;

/**
 * @brief Data format of a tensor
 */
typedef enum
{
  /** Don't care layout */
  NNFW_LAYOUT_NONE = 0,
  /**
   * Channel last layout
   * If rank is 4, layout is NHWC
   */
  NNFW_LAYOUT_CHANNELS_LAST = 1,
  /**
   * Channel first layout
   * If rank is 4, layout is NCHW
   */
  NNFW_LAYOUT_CHANNELS_FIRST = 2,
} NNFW_LAYOUT;

/**
 * @brief Information ID for retrieving information on nnfw (e.g. version)
 */
typedef enum
{
  /** nnfw runtime version
   * Its value is uint32 in 0xMMmmmmPP, where MM = major, mmmm = minor, PP = patch.
   */
  NNFW_INFO_ID_VERSION = 0,
} NNFW_INFO_ID;

/**
 * @brief Maximum rank expressible with nnfw
 */
#define NNFW_MAX_RANK (6)

/**
 * @brief tensor info describes the type and shape of tensors
 *
 * <p>This structure is used to describe input and output tensors.
 * Application can get input and output tensor type and shape described in model by using
 * {@link nnfw_input_tensorinfo} and {@link nnfw_output_tensorinfo}
 *
 * <p>Maximum rank is 6 (NNFW_MAX_RANK). And tensor's dimension value is filled in 'dims' field from
 * index 0.
 * For example, if tensor's rank is 4,
 * application can get dimension value from dims[0], dims[1], dims[2], and dims[3]
 */
typedef struct nnfw_tensorinfo
{
  /** The data type */
  NNFW_TYPE dtype;
  /** The number of dimensions (rank) */
  int32_t rank;
  /**
   * The dimension of tensor.
   * Maximum rank is 6 (NNFW_MAX_RANK).
   */
  int32_t dims[NNFW_MAX_RANK];
} nnfw_tensorinfo;

/**
 * @brief Create a new session instance.
 *
 * <p>This only creates a session.
 * Model is loaded after {@link nnfw_load_model_from_file} is invoked.
 * And inference is performed after {@link nnfw_run} is invoked.
 *
 * <p>{@link nnfw_close_session} should be called once
 * if session is no longer needed
 *
 * @param[out]  session The session to be created
 * @return      NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_create_session(nnfw_session **session);

/**
 * @brief Close a session instance
 *
 * After called, access to closed session by application will be invalid
 *
 * @param[in] session The session to be closed
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_close_session(nnfw_session *session);

/**
 * @brief     Load model from nnpackage file or directory
 *
 * The length of \p package_file_path must not exceed 1024 bytes including zero at the end.
 *
 * @param[in] session           nnfw_session loading the given nnpackage file/dir
 * @param[in] package_file_path Path to the nnpackage file or unzipped directory to be loaded
 *
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_load_model_from_file(nnfw_session *session, const char *package_file_path);

/**
 * @brief     Apply i-th input's tensor info to resize input tensor
 *
 * This function should be called before {@link nnfw_prepare} is invoked, and
 * should be called after {@link nnfw_load_model_from_file} is invoked
 * See {@link nnfw_prepare} for information applying updated tensor info
 * If this function is called many times for same index, tensor info is overwritten
 *
 * @deprecated Deprecated since 1.7.0. Use {@link nnfw_set_input_tensorinfo} instead.
 *
 * @param[in] session     Session to the input tensor info is to be set
 * @param[in] index       Index of input to be applied (0-indexed)
 * @param[in] tensor_info Tensor info to be applied
 * @return    @c NNFW_STATUS_NO_ERROR if successful, otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_apply_tensorinfo(nnfw_session *session, uint32_t index,
                                  nnfw_tensorinfo tensor_info);

/**
 * @brief    Set input model's tensor info for resizing
 *
 * This function can be called at any time after calling {@link nnfw_load_model_from_file}. Changing
 * input tensor's shape will cause shape inference for the model. There are two different types of
 * shape inference - static and dynamic. Which one to use is depend on the current state of the
 * session.
 * When it is called after calling {@link nnfw_load_model_from_file} and before calling {@link
 * nnfw_prepare}, this info will be used when {@link nnfw_prepare}. And it will perform static shape
 * inference for all tensors.
 * When it is called after calling {@link nnfw_prepare} or even after {@link nnfw_run}, this info
 * will be used when {@link nnfw_run}. And the shapes of the tensors are determined on the fly.
 * If this function is called many times for the same index, it is overwritten.
 *
 * @param[in] session     Session to the input tensor info is to be set
 * @param[in] index       Index of input to be set (0-indexed)
 * @param[in] tensor_info Tensor info to be set
 * @return    @c NNFW_STATUS_NO_ERROR if successful, otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_set_input_tensorinfo(nnfw_session *session, uint32_t index,
                                      const nnfw_tensorinfo *tensor_info);

/**
 * @brief     Prepare session to be ready for inference
 *
 * This phase may finalize model compilation, scheduling, and additional settings.
 * If {@link nnfw_apply_tensorinfo} is called to apply input tensor info different with model
 * before this function, tries to resize all tensors.
 *
 * @param[in] session the session to be prepared
 * @return    @c NNFW_STATUS_NO_ERROR if successful, otherwise return @c NNFW_STATUS_ERROR
 */
NNFW_STATUS nnfw_prepare(nnfw_session *session);

/**
 * @brief     Run inference
 *
 * <p>This function should be called after model is loaded by {@link nnfw_load_model_from_file},
 * session is prepared for inference by {@link nnfw_prepare}, set input and output buffers
 * by {@link nnfw_set_input} and {@link nnfw_set_output}.</p>
 *
 * <p>This function return after inference is finished.</p>
 *
 * @param[in] session The session to run inference
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_run(nnfw_session *session);

/**
 * @brief     Run inference asynchronously
 *
 * <p>This function must be called after model is loaded by {@link nnfw_load_model_from_file},
 * session is prepared for inference by {@link nnfw_prepare}, set input and output buffers
 * by {@link nnfw_set_input} and {@link nnfw_set_output}.</p>
 *
 * <p>This function returns immediately after starting a thread to run the inference.
 * To get the result of it or to do the next inference with {@link nnfw_run} or
 * {@link nnfw_run_async}, {@link nnfw_await} must be called to ensure the current asynchronous
 * inference has finished. Only one asynchronous inference is allowed at a time for a session.
 * If this function is called while the previous one is still running, it returns an error.</p>
 *
 * @param[in] session The session to run inference
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_run_async(nnfw_session *session);

/**
 * @brief     Wait for asynchronous run to finish
 *
 * <p>This function must be called after calling {@link nnfw_run_async}, and can be called only once
 * for a {@link nnfw_run_async} call.
 *
 * <p>When this function returns, it means that this session has finished the asynchronous run. Then
 * the user can safely use the output data.</p>
 *
 * <p>This function returns after the asynchronous inference is finished.</p>
 *
 * @param[in] session The session to run inference
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_await(nnfw_session *session);

/**
 * @brief     Set input buffer
 *
 * This function must be called after {@link nnfw_prepare}, \p buffer given to this function can be
 * reused for many inferences. \p length must be greater or equal than the operand requires. To
 * specify an optional input, you can either not call this for that input or call this with \p
 * buffer of NULL and \p length of 0.
 *
 * @param[in] session Session to the input is to be set
 * @param[in] index   Index of input to be set (0-indexed)
 * @param[in] type    Type of the input
 * @param[in] buffer  Raw buffer for input
 * @param[in] length  Size of bytes of input buffer
 *
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_input(nnfw_session *session, uint32_t index, NNFW_TYPE type,
                           const void *buffer, size_t length);

/**
 * @brief       Set output buffer
 *
 * This function must be called after {@link nnfw_prepare}, \p buffer given to this function can be
 * reused for many inferences. \p length must be greater or equal than the operand requires. An
 * output operand can have unspecified shape and deduced dynamically during the execution. You must
 * provide \p buffer large enough.
 *
 * @param[in]   session Session from inference output is to be extracted
 * @param[in]   index   Index of output to be set (0-indexed)
 * @param[in]   type    Type of the output
 * @param[out]  buffer  Raw buffer for output
 * @param[in]   length  Size of bytes of output buffer
 *
 * @return      @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_output(nnfw_session *session, uint32_t index, NNFW_TYPE type, void *buffer,
                            size_t length);

/**
 * @brief       Get the number of inputs
 *
 * Application can call this function to get number of inputs defined in loaded model.
 * This function should be called after {@link nnfw_load_model_from_file} is invoked to load model
 *
 * @param[in]   session Session from input information is to be extracted
 * @param[out]  number  Variable which the number of inputs is put into
 *
 * @return      @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_input_size(nnfw_session *session, uint32_t *number);

/**
 * @brief       Get the number of outputs
 *
 * Application can call this function to get number of outputs defined in loaded model.
 * This function should be called after {@link nnfw_load_model_from_file} is invoked to load model
 *
 * @param[in]   session Session from output information is to be extracted
 * @param[out]  number  Variable which the number of outputs is put into
 *
 * @return      @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_output_size(nnfw_session *session, uint32_t *number);

/**
 * @brief Set the layout of an input
 *
 * The input that does not call this has NNFW_LAYOUT_NHWC layout
 *
 * @param[in] session session from inference input is to be extracted
 * @param[in] index   index of input to be set (0-indexed)
 * @param[in] layout  layout to set to target input
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_input_layout(nnfw_session *session, uint32_t index, NNFW_LAYOUT layout);

/**
 * @brief Set the layout of an output
 *
 * The output that does not call this has NNFW_LAYOUT_NHWC layout
 *
 * @param[in] session session from inference output is to be extracted
 * @param[in] index   index of output to be set (0-indexed)
 * @param[in] layout  layout to set to target output
 *
 * @return NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_output_layout(nnfw_session *session, uint32_t index, NNFW_LAYOUT layout);

/**
 * @brief       Get i-th input tensor info
 *
 * <p>Before {@link nnfw_prepare} is invoked, this function return tensor info in model,
 * so updated tensor info by {@link nnfw_apply_tensorinfo} is not returned.</p>
 *
 * <p>After {@link nnfw_prepare} is invoked, this function return updated tensor info
 * if tensor info is updated by {@link nnfw_apply_tensorinfo}.</p>
 *
 * @param[in]   session     Session from input information is to be extracted
 * @param[in]   index       Index of input
 * @param[out]  tensor_info Tensor info (shape, type, etc)
 *
 * @return      @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_input_tensorinfo(nnfw_session *session, uint32_t index,
                                  nnfw_tensorinfo *tensor_info);

/**
 * @brief     Get i-th output tensor info
 *
 * <p>After {@link nnfw_load_model_from_file} and before {@link nnfw_prepare} is invoked, it returns
 * tensor info in the model.</p>
 *
 * <p>After {@link nnfw_prepare} and before {@link nnfw_run} is invoked, this function returns
 * updated tensor info if tensor info is updated by {@link nnfw_set_input_tensorinfo}.</p>
 *
 * <p>After {@link nnfw_run} is invoked(at least once), it returns the updated tensor info during
 * the latest execution.</p>
 *
 * @param[in]   session     Session from output information is to be extracted
 * @param[in]   index       Index of output
 * @param[out]  tensor_info Tensor info (shape, type, etc)
 *
 * @return      @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_output_tensorinfo(nnfw_session *session, uint32_t index,
                                   nnfw_tensorinfo *tensor_info);

/**
 * @brief     Set available backends
 *
 * This function should be called before {@link nnfw_prepare} is invoked.
 *
 * <p>Supported backends differs on each platforms.
 * For example, `x86_64` supports "cpu" only.
 * Multiple backends can be set and they must be separated by a semicolon (ex: "acl_cl;cpu").
 * For each backend string, `libbackend_{backend}.so` will be dynamically loaded during
 * {@link nnfw_prepare}.
 * Among the multiple backends, the 1st element is used as the default backend.</p>
 *
 * @param[in] session session to which avilable backends are set
 * @param[in] backends available backends on which nnfw uses
 *
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_available_backends(nnfw_session *session, const char *backends);

/**
 * @brief     Set the operation's backend
 *
 * This function should be called before {@link nnfw_prepare} is invoked.
 *
 * <p>The backend for op has higher priority than available backends specified by
 * {@link nnfw_set_available_backends}.</p>
 *
 * @deprecated Deprecated since 1.8.0.
 *
 * @param[in] session session to be modified
 * @param[in] op operation to be set
 * @param[in] backend bakcend on which operation run
 *
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_op_backend(nnfw_session *session, const char *op, const char *backend);

/**
 * @brief     Retrieve uint32 type of nnfw information for given information ID.
 *
 * <p>Retrieves the information of property given by information id </p>
 *
 * @note: The input session could be null for global information (e.g. runtime version).*
 *
 * @param[in] session session to be queried on.
 * @param[in] id ID to be queried
 * @param[out] val uint32 value to be returned.
 *
 * @return @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_query_info_u32(nnfw_session *session, NNFW_INFO_ID id, uint32_t *val);

/**
 * @brief     Set runtime's workspace directory
 *
 * <p>This function sets the directory to be used as a workspace.
 * System should allow read and write access to the directory for the runtime.
 * Default workspace is running directory of the application.
 * This function should be called before {@link nnfw_prepare} is invoked.</p>
 *
 * @param[in] session session to be queried on.
 * @param[in] dir     workspace directory path
 * @return    @c NNFW_STATUS_NO_ERROR if successful
 */
NNFW_STATUS nnfw_set_workspace(nnfw_session *session, const char *dir);

#ifdef __cplusplus
}
#endif

#endif
