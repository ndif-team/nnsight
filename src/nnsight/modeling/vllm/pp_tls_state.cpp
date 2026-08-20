// Per-greenlet torch thread-local state capture/restore.
//
// A bundle captures the full at::ThreadLocalState (dispatch key sets, grad
// mode, function/dispatch mode stacks, debug info, observer callbacks) plus
// the c10 warning handler, which every pybind torch binding points at a
// handler object on its caller's C stack for the duration of the call. A
// greenlet park inside such a call strands that pointer in the OS thread's
// state while the stack it points into is sliced away; restoring the
// arriving greenlet's bundle keeps every pointer paired with the stack that
// owns it.
//
// Built at runtime with torch.utils.cpp_extension.load against the installed
// torch, so there is no link configuration to maintain here.

#include <torch/extension.h>

#include <ATen/ThreadLocalState.h>
#include <c10/util/Exception.h>

namespace {

struct Bundle {
  at::ThreadLocalState state;      // captures on construction
  c10::WarningHandler* handler;

  Bundle() : state(), handler(c10::WarningUtils::get_warning_handler()) {}

  void restore() const {
    at::ThreadLocalState::setThreadLocalState(state);
    c10::WarningUtils::set_warning_handler(handler);
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Bundle>(m, "Bundle")
      .def(py::init<>())
      .def("restore", &Bundle::restore);
}
