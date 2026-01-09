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
    // ukuran asli frame
    int orig_width = image.cols;
    int orig_height = image.rows;
    
    // step 1: resize ke blobsize (seperti python size=blobsize)
    cv::Mat blob_resized;
    float scale = (float)blob_size / max(orig_width, orig_height);
    int new_w = (int)(orig_width * scale);
    int new_h = (int)(orig_height * scale);
    cv::resize(image, blob_resized, cv::Size(new_w, new_h));
    
    // step 2: letterbox padding ke 416x416 untuk onnx
    cv::Mat padded = cv::Mat::zeros(input_height, input_width, CV_8UC3);
    int pad_x = (input_width - new_w) / 2;
    int pad_y = (input_height - new_h) / 2;
    blob_resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));
    
    // normalize ke float
    cv::Mat resized;
    padded.convertTo(resized, CV_32F, 1.0 / 255.0);
    
    // BGR to RGB dan split channels
    vector<cv::Mat> channels(3);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    cv::split(resized, channels);
    
    // HWC ke CHW dengan memcpy
    int channel_size = input_width * input_height;
    vector<float> input_tensor_values(3 * channel_size);
    memcpy(input_tensor_values.data(), channels[0].ptr<float>(), channel_size * sizeof(float));
    memcpy(input_tensor_values.data() + channel_size, channels[1].ptr<float>(), channel_size * sizeof(float));
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
    
    vector<Detection> detections;
    
    // loop semua deteksi dari output model 
    for (int i = 0; i < num_boxes; i++) {
        float cx = output[i * elements + 0];
        float cy = output[i * elements + 1];
        float w  = output[i * elements + 2];
        float h  = output[i * elements + 3];
        float conf = output[i * elements + 4];
        
        // filter confidence
        if (conf < 0.40) continue;
        
        // konversi dari 416x416 letterbox ke koordinat asli
        // 1. hapus padding
        float cx_unpad = cx - pad_x;
        float cy_unpad = cy - pad_y;
        float w_unpad = w;
        float h_unpad = h;
        
        // 2. scale balik ke ukuran asli
        float inv_scale = 1.0f / scale;
        int x = (int)((cx_unpad - w_unpad/2) * inv_scale);
        int y = (int)((cy_unpad - h_unpad/2) * inv_scale);
        int width = (int)(w_unpad * inv_scale);
        int height = (int)(h_unpad * inv_scale);
        
        // clamp ke batas frame
        x = max(0, min(x, orig_width - 1));
        y = max(0, min(y, orig_height - 1));
        width = min(width, orig_width - x);
        height = min(height, orig_height - y);
        
        if(width > 0 && height > 0) {
            Detection det;
            det.box = cv::Rect(x, y, width, height);
            det.conf = conf;
            det.class_id = 0;
            detections.push_back(det);
        }
    }
    
    return detections;
}

void YoloONNX::setInputSize(int size) {
    blob_size = size;
}
