#ifndef __NNFW_TFLITE_COMPARATOR_TENSOR_VIEW_H__
#define __NNFW_TFLITE_COMPARATOR_TENSOR_VIEW_H__

#include "nnfw.h"

#include "misc/tensor/Shape.h"
#include "misc/tensor/Index.h"
#include "misc/tensor/Reader.h"
#include "misc/tensor/NonIncreasingStride.h"

namespace nnfw
{
namespace onert_cmp
{

/**
 * @brief Class to define TensorView which is inherited from nnfw::misc::tensor::Reader<T> class
 */
template <typename T> class TensorView final : public misc::tensor::Reader<T>
{
public:
  /**
   * @brief Construct a TensorView object with base and shape informations
   * @param[in] shape The shape of a tensor
   * @param[in] base The base address of a tensor
   */
  TensorView(const misc::tensor::Shape &shape, T *base) : _shape{shape}, _base{base}
  {
    // Set 'stride'
    _stride.init(_shape);
  }

public:
  /**
   * @brief Get shape of tensor
   * @return Reference of shape
   */
  const misc::tensor::Shape &shape(void) const override { return _shape; }

  /**
   * @brief Get value of tensor index
   * @param[in] index The tensor index
   * @return The value at the index
   */
  T at(const misc::tensor::Index &index) const override
  {
    const auto offset = _stride.offset(index);
    return *(_base + offset);
  }

  /**
   * @brief Get reference value of tensor index
   * @param[in] index The tensor index
   * @return The reference value at the index
   */
  T &at(const misc::tensor::Index &index)
  {
    const auto offset = _stride.offset(index);
    return *(_base + offset);
  }

private:
  misc::tensor::Shape _shape; /**< The tensor shape */

public:
  T *_base;                                  /**< The base address of tensor */
  misc::tensor::NonIncreasingStride _stride; /**< The NonIncreasingStride object */
};

} // namespace onert_cmp
} // namespace nnfw

#endif // __NNFW_TFLITE_COMPARATOR_TENSOR_VIEW_H__
