#include "yolo_onnx.hpp"

YoloONNX::YoloONNX(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo")
{
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC);
    session = Ort::Session(env, model_path.c_str(), session_options);
    
    printModelInfo();  
}

void YoloONNX::printModelInfo() {
    Ort::AllocatorWithDefaultOptions allocator;

    size_t num_inputs = session.GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        char* input_name = session.GetInputName(i, allocator);

        std::cout << "Input " << i << " name: " << input_name << std::endl;

        allocator.Free(input_name);
    }

    size_t num_outputs = session.GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        char* output_name = session.GetOutputName(i, allocator);

        std::cout << "Output " << i << " name: " << output_name << std::endl;

        allocator.Free(output_name);
    }
}



std::vector<Detection> YoloONNX::infer(const cv::Mat& image)
{
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width, input_height));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // BGR → RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // HWC → CHW
    std::vector<float> input_tensor_values(1 * 3 * input_width * input_height);
    int idx = 0;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < input_height; y++) {
            for (int x = 0; x < input_width; x++) {
                input_tensor_values[idx++] =
                    resized.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    std::array<int64_t, 4> input_shape{1, 3, input_height, input_width};
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output"};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    float* output = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    int num_boxes = shape[1];
    int elements = shape[2];

    std::vector<Detection> detections;

    for (int i = 0; i < num_boxes; i++) {
        float conf = output[i * elements + 4];
        if (conf < 0.5) continue;

        float cx = output[i * elements + 0] * image.cols;
        float cy = output[i * elements + 1] * image.rows;
        float w  = output[i * elements + 2] * image.cols;
        float h  = output[i * elements + 3] * image.rows;

        int x = int(cx - w / 2);
        int y = int(cy - h / 2);

        detections.push_back({
            cv::Rect(x, y, int(w), int(h)),
            conf,
            0
        });
    }

    return detections;
}

