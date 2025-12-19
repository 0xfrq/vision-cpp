#include <onnxruntime_cxx_api.h>
#include <iostream>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-test");
    std::cout << "ONNX Runtime 1.23.2 works!" << std::endl;
    return 0;
}

