// deteksi bola yolov5 + hsv tracking untuk robot sepak bola
// EXACT port dari python gandamana_vision.py ball_detect()
#include <ros/ros.h>
#include <ros/package.h>
#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cmath>
#include "yolo_onnx.hpp"

// ============ GLOBAL STATE (EXACT PYTHON MIRROR) ============
int min_h = 0, min_s = 0, min_v = 0;
int max_h = 0, max_s = 0, max_v = 0;
int x_ball = 0, y_ball = 0, w_ball = 0, h_ball = 0;
float x_center_ball = 0, y_center_ball = 0;
int ball_area = 0;

std::string detect_status = "NOTFOUND";  // NOTFOUND atau FOUND
double waktu_sebelum = 0.0;
std::vector<float> scan_area = {0, 0};
const int framesize[2] = {320, 240};
int blobsize = 416;
const float g_fisheye = 1.0f;  // 0.34 jika fisheye, 1.0 normal
double fps = 0.0;
int frame_counter = 0;
double fps_start_time = 0.0;

// ============ UTILITY FUNCTIONS ============
int min_value(int a, int b) {
    return (a <= b) ? a : b;
}

int max_value(int a, int b) {
    return (a >= b) ? a : b;
}

double map_value(double source_val, double source_min, double source_max, double target_min, double target_max) {
    source_val = (double)min_value(source_val, max_value(source_min, source_max));
    source_val = (double)max_value(source_val, min_value(source_min, source_max));
    return target_min + ((source_val - source_min) * ((target_max - target_min) / (source_max - source_min)));
}

// ============ THREADED CAMERA ============
class ThreadedCapture {
private:
    cv::VideoCapture cap;
    cv::Mat frame;
    bool stopped;
    std::mutex frameMutex;
    std::thread captureThread;
    
    void update() {
        while(!stopped) {
            cv::Mat temp;
            cap >> temp;
            if(!temp.empty()) {
                std::lock_guard<std::mutex> lock(frameMutex);
                frame = temp.clone();
            }
        }
    }
    
public:
    ThreadedCapture(int src) : stopped(false) {
        cap.open(src, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FPS, 60);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, framesize[0]);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, framesize[1]);
        if(!cap.isOpened()) {
            ROS_ERROR("kamera gagal dibuka");
            return;
        }
        cap >> frame;
        captureThread = std::thread(&ThreadedCapture::update, this);
    }
    
    cv::Mat read() {
        std::lock_guard<std::mutex> lock(frameMutex);
        return frame.clone();
    }
    
    void stop() {
        stopped = true;
        if(captureThread.joinable()) captureThread.join();
        cap.release();
    }
    
    ~ThreadedCapture() { stop(); }
};

// ============ CALCULATE FPS ============
void calculate_fps() {
    frame_counter++;
    double current_time = ros::Time::now().toSec();
    double elapsed_time = current_time - fps_start_time;
    if(elapsed_time >= 0.2) {
        fps = frame_counter / elapsed_time;
        frame_counter = 0;
        fps_start_time = current_time;
    }
}

// ============ EXTRACT FIELD (GREEN AREA) ============
cv::Mat extractField(const cv::Mat& img) {
    cv::Mat hsv, mask, result;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    
    cv::inRange(hsv, cv::Scalar(35, 40, 40), cv::Scalar(85, 255, 255), mask);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::erode(mask, mask, kernel, cv::Point(-1,-1), 2);
    cv::dilate(mask, mask, kernel, cv::Point(-1,-1), 5);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    
    cv::Mat fieldMask = cv::Mat::zeros(img.size(), CV_8UC1);
    if(!contours.empty()) {
        auto maxContour = *std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>&a, const std::vector<cv::Point>&b){ return cv::contourArea(a) < cv::contourArea(b); });
        std::vector<cv::Point> hull;
        cv::convexHull(maxContour, hull);
        cv::fillConvexPoly(fieldMask, hull, cv::Scalar(255));
    }
    
    cv::bitwise_and(img, img, result, fieldMask);
    return result;
}

// ============ GET HSV VALUE - EXACT PYTHON PORT ============
void get_hsv_val(const cv::Mat& img) {
    int x = x_ball;
    int y = y_ball;
    int w = w_ball;
    int h = h_ball;
    
    if(x == 0 && y == 0 && w == 0 && h == 0) return;
    
    int x1 = x;
    int y1 = y;
    int x2 = x + w;
    int y2 = y + h;
    
    ball_area = w * h;
    ROS_INFO("LUAS BOLA : %d", ball_area);
    
    // compute dot positions (EXACT PYTHON LOGIC)
    int dot_tengah_x = (x1 + x2) / 2;
    int dot_tengah_y = (y1 + y2) / 2;
    
    // collect sampling points
    std::vector<std::pair<int,int>> dots;
    dots.push_back({dot_tengah_x, dot_tengah_y});                                      // dot_tengah
    dots.push_back({(x1 + dot_tengah_x) / 2, (y1 + dot_tengah_y) / 2});               // dot_sepertiga_kanan
    dots.push_back({(x2 + dot_tengah_x) / 2, (y2 + dot_tengah_y) / 2});               // dot_duapertiga_kanan
    dots.push_back({(dot_tengah_x + (x1 + (x2-x1))) / 2, (dot_tengah_y + y1) / 2});   // dot_sepertiga_kiri
    dots.push_back({(x1 + dot_tengah_x) / 2, (y1 + (y2-y1) + dot_tengah_y) / 2});     // dot_duapertiga_kiri
    
    // sample HSV at each point
    std::vector<int> H, S, V;
    for(auto& dot : dots) {
        int px = std::max(0, std::min(img.cols-1, dot.first));
        int py = std::max(0, std::min(img.rows-1, dot.second));
        cv::Vec3b bgr = img.at<cv::Vec3b>(py, px);
        
        cv::Mat hsv_mat;
        cv::cvtColor(cv::Mat(1, 1, CV_8UC3, cv::Scalar(bgr[0], bgr[1], bgr[2])), hsv_mat, cv::COLOR_BGR2HSV);
        cv::Vec3b hsv_val = hsv_mat.at<cv::Vec3b>(0, 0);
        
        H.push_back(hsv_val[0]);
        S.push_back(hsv_val[1]);
        V.push_back(hsv_val[2]);
    }
    
    // sample atas_tengah point (extra point like Python - using h//5 not h//4)
    int atas_py = (dot_tengah_y + y1 + h/5) / 2;
    int atas_px = dot_tengah_x;
    atas_py = std::max(0, std::min(img.rows-1, atas_py));
    atas_px = std::max(0, std::min(img.cols-1, atas_px));
    cv::Vec3b bgr_atas = img.at<cv::Vec3b>(atas_py, atas_px);
    cv::Mat hsv_atas;
    cv::cvtColor(cv::Mat(1, 1, CV_8UC3, cv::Scalar(bgr_atas[0], bgr_atas[1], bgr_atas[2])), hsv_atas, cv::COLOR_BGR2HSV);
    cv::Vec3b hsv_val_atas = hsv_atas.at<cv::Vec3b>(0, 0);
    H.push_back(hsv_val_atas[0]);
    S.push_back(hsv_val_atas[1]);
    V.push_back(hsv_val_atas[2]);
    
    // sample bawah_tengah point
    int bawah_py = (dot_tengah_y + y2) / 2;
    int bawah_px = dot_tengah_x;
    bawah_py = std::max(0, std::min(img.rows-1, bawah_py));
    bawah_px = std::max(0, std::min(img.cols-1, bawah_px));
    cv::Vec3b bgr_bawah = img.at<cv::Vec3b>(bawah_py, bawah_px);
    cv::Mat hsv_bawah;
    cv::cvtColor(cv::Mat(1, 1, CV_8UC3, cv::Scalar(bgr_bawah[0], bgr_bawah[1], bgr_bawah[2])), hsv_bawah, cv::COLOR_BGR2HSV);
    cv::Vec3b hsv_val_bawah = hsv_bawah.at<cv::Vec3b>(0, 0);
    H.push_back(hsv_val_bawah[0]);
    S.push_back(hsv_val_bawah[1]);
    V.push_back(hsv_val_bawah[2]);
    
    // compute min/max
    min_h = *std::min_element(H.begin(), H.end());
    min_s = *std::min_element(S.begin(), S.end());
    min_v = *std::min_element(V.begin(), V.end());
    max_h = *std::max_element(H.begin(), H.end());
    max_s = *std::max_element(S.begin(), S.end());
    max_v = *std::max_element(V.begin(), V.end());
    
    // apply thresholds EXACTLY like Python
    int nilai_maksimum_h = 33;
    if(max_h >= nilai_maksimum_h) max_h = nilai_maksimum_h;
    
    int nilai_minimum_s = 160;
    if(min_s <= nilai_minimum_s) min_s = nilai_minimum_s;
    
    ROS_INFO("min hue %d min sat %d min val %d max hue %d max sat %d max val %d", 
             min_h, min_s, min_v, max_h, max_s, max_v);
}

// ============ MAIN ============
int main(int argc, char** argv) {
    ros::init(argc, argv, "vision_yolo_cpp");
    ros::NodeHandle nh;

    auto pub_state = nh.advertise<v2_detection::BallState>("/DEWO/image_processing/deteksi_bola/ball_state", 10);
    auto pub_coord = nh.advertise<v2_detection::BallCoordinate>("/DEWO/image_processing/deteksi_bola/coordinate", 10);
    auto pub_area = nh.advertise<v2_detection::Ballarea>("/DEWO/image_processing/deteksi_bola/ball_area", 10);

    std::string pkg = ros::package::getPath("vision_cpp");
    YoloONNX yolo(pkg + "/src/best.onnx");

    ROS_INFO("memulai kamera...");
    ThreadedCapture capture(0);
    
    fps_start_time = ros::Time::now().toSec();
    waktu_sebelum = ros::Time::now().toSec();
    
    ROS_INFO("sistem siap");

    while(ros::ok()) {
        cv::Mat img = capture.read();
        if(img.empty()) { ros::spinOnce(); continue; }
        
        cv::Mat img_result = img.clone();

        // ============ FOUND STATE - HSV TRACKING ============
        if(detect_status == "FOUND") {
            // Extract field EVERY frame (like Python) with proper HSV values
            cv::Mat field_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
            cv::Mat hsv_image;
            cv::cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
            
            // Field HSV values (calibrated for green field)
            cv::Mat field_binary;
            cv::inRange(hsv_image, cv::Scalar(35, 40, 40), cv::Scalar(85, 255, 255), field_binary);
            cv::erode(field_binary, field_binary, field_kernel, cv::Point(-1,-1), 2);
            cv::dilate(field_binary, field_binary, field_kernel, cv::Point(-1,-1), 5);
            
            cv::Mat field_mask = cv::Mat::zeros(img.size(), CV_8UC1);
            std::vector<std::vector<cv::Point>> field_contours;
            cv::findContours(field_binary, field_contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
            
            cv::Mat field_img = img.clone();
            if(field_contours.size() > 0) {
                auto field_cntr = *std::max_element(field_contours.begin(), field_contours.end(),
                    [](const std::vector<cv::Point>&a, const std::vector<cv::Point>&b){ 
                        return cv::contourArea(a) < cv::contourArea(b); 
                    });
                std::vector<cv::Point> hull;
                cv::convexHull(field_cntr, hull);
                cv::fillConvexPoly(field_mask, hull, cv::Scalar(255));
                cv::bitwise_and(img, img, field_img, field_mask);
            }
            
            // Now apply ball HSV tracking on field-masked image
            cv::Mat frame_copy = field_img.clone();
            cv::Mat hsv;
            cv::cvtColor(frame_copy, hsv, cv::COLOR_BGR2HSV);
            
            cv::Mat binary_ball;
            cv::inRange(hsv, cv::Scalar(min_h, min_s, min_v), cv::Scalar(max_h, max_s, max_v), binary_ball);
            
            cv::Mat kernel_bball = cv::Mat::ones(5, 5, CV_8U);
            cv::morphologyEx(binary_ball, binary_ball, cv::MORPH_CLOSE, kernel_bball);
            cv::morphologyEx(binary_ball, binary_ball, cv::MORPH_OPEN, kernel_bball);
            
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(binary_ball, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            bool objek_ditemukan = false;
            int contourLength = contours.size();
            
            for(auto& contour : contours) {
                double area = cv::contourArea(contour);
                
                // EXACT PYTHON FILTER
                if(area > ball_area/5 && area < ball_area*1.1 && contourLength > 0) {
                    cv::Rect r = cv::boundingRect(contour);
                    float x_center = r.x + r.width/2.0f;
                    int area_bola = r.width * r.height;
                    
                    // check scan_area and min area
                    if(scan_area[0] <= x_center && x_center <= scan_area[1] && area_bola > (3000*g_fisheye)) {
                        float x_center_rect = r.x + r.width/2.0f;
                        float y_center_rect = r.y + r.height/2.0f;
                        
                        cv::rectangle(img_result, r, cv::Scalar(0, 255, 255), 2);
                        
                        // SEND ROS
                        v2_detection::BallCoordinate bc;
                        bc.pos_x = (float)(x_center_rect/framesize[0]*2 - 1);
                        bc.pos_y = (float)(y_center_rect/framesize[1]*2 - 1);
                        bc.obj_size = area_bola;
                        pub_coord.publish(bc);
                        
                        v2_detection::Ballarea ba;
                        ba.ballarea = area_bola;
                        pub_area.publish(ba);
                        
                        v2_detection::BallState bs;
                        bs.ball_status = "FOUND";
                        pub_state.publish(bs);
                        
                        objek_ditemukan = true;
                        
                        // check timing to reset to yolo
                        double waktu_detect = map_value(area_bola, 0, 76800, 0.5, 80);
                        double waktu_sesudah = ros::Time::now().toSec();
                        double delta = waktu_sesudah - waktu_sebelum;
                        
                        ROS_INFO("Ball Area Result : %d", area_bola);
                        
                        if(delta >= waktu_detect) {
                            waktu_sebelum = ros::Time::now().toSec();
                            detect_status = "NOTFOUND";
                        }
                        break;
                    }
                    break;
                }
            }
            
            if(!objek_ditemukan) {
                detect_status = "NOTFOUND";
            }
        }

        // ============ NOTFOUND STATE - YOLO DETECTION ============
        if(detect_status == "NOTFOUND") {
            yolo.setInputSize(blobsize);
            auto dets = yolo.infer(img);
            
            if(dets.empty()) {
                ROS_INFO("YOLO SEAQRCHING");
                v2_detection::BallState bs;
                bs.ball_status = "NOTFOUND";
                pub_state.publish(bs);
                blobsize = 416;
                ROS_INFO("Did not detect ball, setting blobsize to %d", blobsize);
            } else {
                bool jika_ditemukan = false;
                
                // ITERATE DETECTIONS - EXACT PYTHON LOGIC
                for(auto& det : dets) {
                    float conf = det.conf;
                    int cls = det.class_id;
                    
                    // Skip if confidence < 0.5 (model.conf)
                    if(conf < 0.5f) continue;
                    
                    int x = det.box.x;
                    int y = det.box.y;
                    int w = det.box.width;
                    int h = det.box.height;
                    
                    // Find class 0 (ball)
                    if(cls == 0) {
                        x_ball = x;
                        y_ball = y;
                        w_ball = w;
                        h_ball = h;
                        
                        x_center_ball = x_ball + w_ball/2.0f;
                        y_center_ball = y_ball + h_ball/2.0f;
                        int obj_size_ball = w_ball * h_ball;
                        
                        // compute scan_area EXACTLY like Python
                        int in_area_ball;
                        if(ball_area <= (5000*g_fisheye)) {
                            in_area_ball = w_ball * 3;
                        } else {
                            in_area_ball = w_ball + (int)(35*g_fisheye);
                        }
                        scan_area = {x_center_ball - (float)in_area_ball, x_center_ball + (float)in_area_ball};
                        
                        // draw box
                        cv::rectangle(img_result, det.box, cv::Scalar(255, 0, 0), 1);
                        char text[50];
                        snprintf(text, sizeof(text), "bola: %.0f%%", conf * 100);
                        cv::putText(img_result, text, cv::Point(x, y-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0), 1);
                        
                        // SEND ROS
                        v2_detection::BallCoordinate bc;
                        bc.pos_x = (float)(x_center_ball/framesize[0]*2 - 1);
                        bc.pos_y = (float)(y_center_ball/framesize[1]*2 - 1);
                        bc.obj_size = obj_size_ball;
                        pub_coord.publish(bc);
                        
                        v2_detection::Ballarea ba;
                        ba.ballarea = obj_size_ball;
                        pub_area.publish(ba);
                        
                        v2_detection::BallState bs;
                        bs.ball_status = "FOUND";
                        pub_state.publish(bs);
                        
                        ROS_INFO("bola [%.0f%%]", conf * 100);
                        
                        // extract HSV
                        get_hsv_val(img);
                        detect_status = "FOUND";
                        waktu_sebelum = ros::Time::now().toSec();
                        
                        // update blobsize
                        if(obj_size_ball <= (2800*g_fisheye)) {
                            blobsize = 320;
                        } else if(obj_size_ball > (2800*g_fisheye)) {
                            blobsize = 224;
                        }
                        
                        jika_ditemukan = true;
                        break;  // CRITICAL: break after finding ball
                    }
                }
                
                // if no ball found
                if(!jika_ditemukan) {
                    v2_detection::BallCoordinate bc;
                    bc.pos_x = 0;
                    bc.pos_y = 0;
                    bc.obj_size = 0;
                    pub_coord.publish(bc);
                    
                    v2_detection::BallState bs;
                    bs.ball_status = "NOTFOUND";
                    pub_state.publish(bs);
                    ROS_INFO("bola not found (after yolov5)");
                }
            }
        }

        // ============ DISPLAY ============
        calculate_fps();
        
        char fps_str[30];
        snprintf(fps_str, sizeof(fps_str), "FPS: %.2f", fps);
        cv::putText(img_result, fps_str, cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        
        char blob_str[20];
        snprintf(blob_str, sizeof(blob_str), "%d", blobsize);
        cv::putText(img_result, blob_str, cv::Point(280, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("VISION_CPP", img_result);
        cv::waitKey(1);
        
        std::cout << std::endl << detect_status << std::endl;
        ros::spinOnce();
    }
    
    capture.stop();
    ROS_INFO("sistem berhenti");
    return 0;
}
