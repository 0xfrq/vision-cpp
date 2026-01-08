#include "yolo_onnx.hpp"

using namespace std;
using namespace Ort;

YoloONNX::YoloONNX(const string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "yolo"),
      session(nullptr)
{
    // set thread count untuk inferensi lebih cepat
    session_options.SetIntraOpNumThreads(1);
    // enable optimasi basic untuk peningkatan performa
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_BASIC);
    
    // load session model yolo
    session = Session(env, model_path.c_str(), session_options);
    
    // print info model saat pertama kali load
    printModelInfo();
}

void YoloONNX::printModelInfo() {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session.GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        char* input_name = session.GetInputName(i, allocator);
        cout << "Input " << i << " name: " << input_name << endl;
        allocator.Free(input_name);
    }
    size_t num_outputs = session.GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        char* output_name = session.GetOutputName(i, allocator);
        cout << "Output " << i << " name: " << output_name << endl;
        allocator.Free(output_name);
    }
}

vector<Detection> YoloONNX::infer(const cv::Mat& image)
{
    // Store original dimensions
    int orig_width = image.cols;
    int orig_height = image.rows;
    
    // resize dan normalize ke float
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width, input_height));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);
    
    // BGR to RGB conversion dengan reshape untuk HWC to CHW lebih cepat
    vector<cv::Mat> channels(3);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    cv::split(resized, channels);
    
    // HWC ke CHW dengan memcpy (paling cepat)
    int channel_size = input_width * input_height;
    vector<float> input_tensor_values(3 * channel_size);
    
    // copy R channel
    memcpy(input_tensor_values.data(), channels[0].ptr<float>(), channel_size * sizeof(float));
    
    // copy G channel
    memcpy(input_tensor_values.data() + channel_size, channels[1].ptr<float>(), channel_size * sizeof(float));
    
    // copy B channel
    memcpy(input_tensor_values.data() + 2 * channel_size, channels[2].ptr<float>(), channel_size * sizeof(float));
    
    array<int64_t, 4> input_shape{1, 3, input_height, input_width};
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
    const char* output_names[] = {"output0"};
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
    
    // hitung faktor skala untuk konversi koordinat
    float scale_x = static_cast<float>(orig_width) / input_width;
    float scale_y = static_cast<float>(orig_height) / input_height;
    
    vector<Detection> detections;
    
    // loop semua deteksi dari output model 
    for (int i = 0; i < num_boxes; i++) {
        // format yolo: [x_center, y_center, width, height, confidence, class_scores...]
        float cx = output[i * elements + 0];
        float cy = output[i * elements + 1];
        float w  = output[i * elements + 2];
        float h  = output[i * elements + 3];
        float conf = output[i * elements + 4];
        
        // filter confidence threshold yang lebih tinggi untuk skip deteksi lemah
        if (conf < 0.55) continue;
        
        // scale koordinat ke ukuran frame asli tanpa clamp (lebih cepat)
        int x = (int)(cx * scale_x - (w * scale_x) / 2);
        int y = (int)(cy * scale_y - (h * scale_y) / 2);
        int width = (int)(w * scale_x);
        int height = (int)(h * scale_y);
        
        // simpan deteksi tanpa boundary check
        Detection det;
        det.box = cv::Rect(x, y, width, height);
        det.conf = conf;
        det.class_id = 0;
        detections.push_back(det);
    }
    
    return detections;
}
