#include "Shape.h"

#include <pybind11.h>

namespace py = pybind11;
using namespace circle_resizer;

PYBIND11_MODULE(circle_resizer_python_api, m)
{
  m.doc() = "circle_resizer::Shape";

  py::class_<Dim>(m, "Dim").def(py::init<int32_t>()).def("value", &Dim::value);
}
