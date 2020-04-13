#ifndef __TESTER_MODEL_H__
#define __TESTER_MODEL_H__

#include <mio/circle/schema_generated.h>

#include <memory>

namespace luci
{

struct Model
{
  virtual ~Model() = default;

  virtual const ::circle::Model *model(void) = 0;
};

/**
 * @brief Load Circle model (as a raw Model) from a given path
 *
 * @note May return a nullptr
 */
std::unique_ptr<Model> load_model(const std::string &path);

} // namespace luci

#endif // __TESTER_MODEL_H__
