#include <torch/extension.h>
#include <iostream>

// Simple CPU implementation that prints and returns input + 1.
torch::Tensor hello_native(torch::Tensor input) {
  // Print only once per call; ok for demo purposes.
  std::cout << "Hello World from C++ extension!" << std::endl;

  // A tiny deterministic transform for correctness checking.
  return input + 1;
}

// Register operator schema under the "teal" namespace.
TORCH_LIBRARY(teal, m) {
  m.def("hello_native(Tensor input) -> Tensor");
}

// Bind the CPU implementation.
TORCH_LIBRARY_IMPL(teal, CPU, m) {
  m.impl("hello_native", hello_native);
}

// Provide a Python module entry so torch.utils.cpp_extension.load can import.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // No Python bindings needed; operator already registered above.
}
