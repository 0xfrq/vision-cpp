#include <ros/ros.h>
#include <ros/package.h>

#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <chrono>

#include "yolo_onnx.hpp"

using namespace cv;
using namespace std;

/* =========================
   UTILITIES
   ========================= */
template<typename T>
inline T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

double map_value(double v, double smin, double smax, double tmin, double tmax) {
    v = clamp(v, min(smin, smax), max(smin, smax));
    return tmin + (v - smin) * (tmax - tmin) / (smax - smin);
}

/* =========================
   SIMPLE 2-STATE (PYTHON STYLE)
   ========================= */
enum DetectState {
    NOTFOUND = 0,  // Use YOLO to find ball
    FOUND = 1      // Use HSV to track ball
};

DetectState state = NOTFOUND;

/* =========================
   BALL STATE
   ========================= */
struct BallInfo {
    Point2f center;
    int area;
    int width, height;
    
    // HSV ranges from 7-point sampling
    int min_h, max_h;
    int min_s, max_s;
    int min_v, max_v;
    
    // Timeout (like Python map_value based on area)
    double last_seen_time;
    double timeout_seconds;
    
    // Scan area bounds
    int scan_x_min, scan_x_max;
    
    bool valid = false;
};

BallInfo ball;

/* =========================
   CONSTANTS
   ========================= */
constexpr double FISHEYE = 1.0;
constexpr int MIN_BALL_AREA = (int)(2500 * FISHEYE);

/* =========================
   FPS TRACKING
   ========================= */
int frame_counter = 0;
double fps_start_time = 0.0;
double fps = 0.0;

void calculate_fps() {
    frame_counter++;
    double current_time = ros::Time::now().toSec();
    double elapsed = current_time - fps_start_time;
    
    if(elapsed >= 0.3) {
        fps = frame_counter / elapsed;
        frame_counter = 0;
        fps_start_time = current_time;
    }
}

/* =========================
   THREADED VIDEO CAPTURE
   ========================= */
class ThreadedCapture {
private:
    VideoCapture cap;
    Mat frame;
    bool stopped;
    mutex frameMutex;
    thread captureThread;
    
    void update() {
        while(!stopped) {
            Mat temp;
            cap >> temp;
            if(!temp.empty()) {
                lock_guard<mutex> lock(frameMutex);
                frame = temp.clone();
            }
        }
    }
    
public:
    ThreadedCapture(int src) : stopped(false) {
        cap.open(src);
        cap.set(CAP_PROP_FPS, 60);
        cap.set(CAP_PROP_FRAME_WIDTH, 320);
        cap.set(CAP_PROP_FRAME_HEIGHT, 240);
        
        if(!cap.isOpened()) {
            ROS_ERROR("Camera failed to open");
            return;
        }
        
        cap >> frame;
        captureThread = thread(&ThreadedCapture::update, this);
    }
    
    Mat read() {
        lock_guard<mutex> lock(frameMutex);
        return frame.clone();
    }
    
    void stop() {
        stopped = true;
        if(captureThread.joinable()) {
            captureThread.join();
        }
        cap.release();
    }
    
    ~ThreadedCapture() {
        stop();
    }
};

/* =========================
   GET HSV VALUES (7-POINT SAMPLING)
   Like Python get_hsv_val()
   ========================= */
void get_hsv_val(const Mat& img, int x1, int y1, int x2, int y2, BallInfo& ball_out) {
    // Calculate 7 sample points (Python algorithm)
    int mid_x = (x1 + x2) / 2;
    int mid_y = (y1 + y2) / 2;
    
    vector<Point> sample_points = {
        {mid_x, mid_y},                                    // Center
        {(x1 + mid_x)/2, (y1 + mid_y)/2},                 // Top-left third
        {(x2 + mid_x)/2, (y2 + mid_y)/2},                 // Bottom-right third
        {(mid_x + x2)/2, (mid_y + y1)/2},                 // Top-right third
        {(x1 + mid_x)/2, (y1 + y2 + mid_y)/2},            // Bottom-left third
        {mid_x, (mid_y + y1 + (y2-y1)/5)/2},              // Upper center
        {mid_x, (mid_y + y2)/2}                            // Lower center
    };
    
    vector<int> H_vals, S_vals, V_vals;
    
    for(auto& pt : sample_points) {
        int px = clamp(pt.x, 0, img.cols-1);
        int py = clamp(pt.y, 0, img.rows-1);
        
        Vec3b bgr = img.at<Vec3b>(py, px);
        Mat bgr_mat(1, 1, CV_8UC3, Scalar(bgr[0], bgr[1], bgr[2]));
        Mat hsv_mat;
        cvtColor(bgr_mat, hsv_mat, COLOR_BGR2HSV);
        Vec3b hsv = hsv_mat.at<Vec3b>(0, 0);
        
        H_vals.push_back(hsv[0]);
        S_vals.push_back(hsv[1]);
        V_vals.push_back(hsv[2]);
    }
    
    // Python constraints: max_h <= 33, min_s >= 160
    ball_out.min_h = *min_element(H_vals.begin(), H_vals.end());
    ball_out.max_h = min(*max_element(H_vals.begin(), H_vals.end()), 33);
    ball_out.min_s = max(*min_element(S_vals.begin(), S_vals.end()), 160);
    ball_out.max_s = *max_element(S_vals.begin(), S_vals.end());
    ball_out.min_v = *min_element(V_vals.begin(), V_vals.end());
    ball_out.max_v = *max_element(V_vals.begin(), V_vals.end());
}

/* =========================
   EXTRACT FIELD (GREEN FIELD MASKING)
   Like Python extract_field()
   ========================= */
Mat extract_field(const Mat& img) {
    Mat hsv, mask;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // Green field detection
    Scalar lower(35, 40, 40);
    Scalar upper(85, 255, 255);
    inRange(hsv, lower, upper, mask);
    
    // Morphological operations
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    erode(mask, mask, kernel, Point(-1,-1), 2);
    dilate(mask, mask, kernel, Point(-1,-1), 5);
    
    // Find largest contour (field)
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    
    Mat field_mask = Mat::zeros(img.size(), CV_8UC1);
    if(!contours.empty()) {
        auto max_contour = *max_element(contours.begin(), contours.end(),
            [](const vector<Point>&a, const vector<Point>&b) {
                return contourArea(a) < contourArea(b);
            });
        
        // Convex hull
        vector<Point> hull;
        convexHull(max_contour, hull);
        fillConvexPoly(field_mask, hull, Scalar(255));
    }
    
    Mat result;
    bitwise_and(img, img, result, field_mask);
    return result;
}

/* =========================
   YOLO DETECTION
   ========================= */
bool yolo_detect(YoloONNX& yolo, const Mat& frame, BallInfo& ball_out) {
    auto detections = yolo.infer(frame);
    
    // Find best ball detection (class_id = 0)
    float best_conf = 0.0f;
    Rect best_box;
    
    for(auto& det : detections) {
        if(det.class_id == 0 && det.conf > 0.5f && det.conf > best_conf) {
            best_conf = det.conf;
            best_box = det.box;
        }
    }
    
    if(best_conf > 0.0f) {
        // Found ball
        ball_out.center = Point2f(best_box.x + best_box.width/2.0f,
                                  best_box.y + best_box.height/2.0f);
        ball_out.area = best_box.area();
        ball_out.width = best_box.width;
        ball_out.height = best_box.height;
        
        // Extract HSV from 7 points
        get_hsv_val(frame, best_box.x, best_box.y,
                    best_box.x + best_box.width,
                    best_box.y + best_box.height, ball_out);
        
        // Calculate timeout (Python: map_value(area, 0, 76800, 0.5, 80))
        ball_out.timeout_seconds = map_value(ball_out.area, 0, 76800, 0.5, 80.0);
        ball_out.last_seen_time = ros::Time::now().toSec();
        
        // Calculate scan area (Python algorithm)
        int area_threshold = (int)(5000 * FISHEYE);
        int scan_width;
        if(ball_out.area < area_threshold) {
            scan_width = ball_out.width * 3;  // Small ball
        } else {
            scan_width = ball_out.width + 35; // Large ball
        }
        
        ball_out.scan_x_min = max(0, (int)(ball_out.center.x - scan_width));
        ball_out.scan_x_max = min(frame.cols, (int)(ball_out.center.x + scan_width));
        
        ball_out.valid = true;
        return true;
    }
    
    return false;
}

/* =========================
   HSV TRACKING
   ========================= */
bool hsv_track(const Mat& frame, BallInfo& ball_out) {
    if(!ball_out.valid) return false;
    
    // Extract field (Python applies this during HSV tracking)
    Mat field_img = extract_field(frame);
    
    // Convert to HSV
    Mat hsv;
    cvtColor(field_img, hsv, COLOR_BGR2HSV);
    
    // Apply HSV range from ball
    Mat mask;
    inRange(hsv, Scalar(ball_out.min_h, ball_out.min_s, ball_out.min_v),
                 Scalar(ball_out.max_h, ball_out.max_s, ball_out.max_v), mask);
    
    // Morphological operations
    Mat kernel = Mat::ones(5, 5, CV_8U);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    
    // Find contours
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // Python area filter: area > ball_area//5 and area < ball_area*1.1
    int min_area = ball_out.area / 5;
    int max_area = (int)(ball_out.area * 1.1);
    
    Point2f best_center;
    int best_area = 0;
    int best_width = 0;
    int best_height = 0;
    bool found = false;
    
    for(auto& contour : contours) {
        int area = (int)contourArea(contour);
        
        // Area filter
        if(area < min_area || area > max_area) continue;
        
        // Get bounding box
        Rect bbox = boundingRect(contour);
        Point2f center(bbox.x + bbox.width/2.0f, bbox.y + bbox.height/2.0f);
        
        // Check if in scan area (Python: x_center check)
        if(center.x < ball_out.scan_x_min || center.x > ball_out.scan_x_max) continue;
        
        // Found valid contour - take largest
        if(area > best_area) {
            best_area = area;
            best_center = center;
            best_width = bbox.width;
            best_height = bbox.height;
            found = true;
        }
    }
    
    if(found) {
        // Update ball position
        ball_out.center = best_center;
        ball_out.area = best_area;
        ball_out.width = best_width;
        ball_out.height = best_height;
        
        // Update last seen time
        ball_out.last_seen_time = ros::Time::now().toSec();
        
        // Update scan area for next frame
        int area_threshold = (int)(5000 * FISHEYE);
        int scan_width;
        if(ball_out.area < area_threshold) {
            scan_width = ball_out.width * 3;
        } else {
            scan_width = ball_out.width + 35;
        }
        
        ball_out.scan_x_min = max(0, (int)(ball_out.center.x - scan_width));
        ball_out.scan_x_max = min(frame.cols, (int)(ball_out.center.x + scan_width));
        
        return true;
    }
    
    return false;
}

/* =========================
   MAIN
   ========================= */
int main(int argc, char** argv) {
    ros::init(argc, argv, "vision_yolo_cpp");
    ros::NodeHandle nh;

    auto pub_state = nh.advertise<v2_detection::BallState>(
        "/DEWO/image_processing/deteksi_bola/ball_state", 10);
    auto pub_coord = nh.advertise<v2_detection::BallCoordinate>(
        "/DEWO/image_processing/deteksi_bola/coordinate", 10);
    auto pub_area = nh.advertise<v2_detection::Ballarea>(
        "/DEWO/image_processing/deteksi_bola/ball_area", 10);

    string pkg = ros::package::getPath("vision_cpp");
    YoloONNX yolo(pkg + "/src/best.onnx");

    ROS_INFO("=== Simple Ball Tracking (Python Algorithm) ===");
    ROS_INFO("States: NOTFOUND (YOLO) / FOUND (HSV)");
    ROS_INFO("Timeout-based state switching");
    ROS_INFO("================================================");

    ThreadedCapture capture(0);
    fps_start_time = ros::Time::now().toSec();

    while(ros::ok()) {
        Mat frame = capture.read();
        if(frame.empty()) {
            ros::spinOnce();
            continue;
        }
        
        Mat display_frame = frame.clone();
        bool detected = false;
        double current_time = ros::Time::now().toSec();

        // ==== STATE MACHINE (PYTHON STYLE) ====
        if(state == NOTFOUND) {
            // Use YOLO to find ball
            if(yolo_detect(yolo, frame, ball)) {
                ROS_INFO("YOLO found ball! Area:%d Timeout:%.1fs", 
                         ball.area, ball.timeout_seconds);
                state = FOUND;
                detected = true;
            }
        }
        else if(state == FOUND) {
            // Use HSV tracking
            if(hsv_track(frame, ball)) {
                detected = true;
            } else {
                // Check timeout (Python style)
                double elapsed = current_time - ball.last_seen_time;
                if(elapsed > ball.timeout_seconds) {
                    ROS_INFO("Timeout %.1fs exceeded, back to YOLO search", 
                             ball.timeout_seconds);
                    state = NOTFOUND;
                    ball.valid = false;
                }
            }
        }
        
        // ==== PUBLISH ROS MESSAGES ====
        v2_detection::BallState bs;
        v2_detection::BallCoordinate bc;
        v2_detection::Ballarea ba;
        
        if(detected && ball.valid) {
            bs.ball_status = "FOUND";
            
            // Normalize coordinates to [-1, 1]
            bc.pos_x = clamp((double)ball.center.x / frame.cols * 2 - 1, -1.0, 1.0);
            bc.pos_y = clamp((double)ball.center.y / frame.rows * 2 - 1, -1.0, 1.0);
            bc.obj_size = ball.area;
            
            ba.ballarea = ball.area;
            
            pub_state.publish(bs);
            pub_coord.publish(bc);
            pub_area.publish(ba);
            
            ROS_INFO_THROTTLE(0.5, "FOUND | Area:%d | Pos:(%.1f, %.1f)", 
                             ball.area, ball.center.x, ball.center.y);
        } else {
            bs.ball_status = "NOTFOUND";
            pub_state.publish(bs);
            
            ROS_INFO_THROTTLE(1.0, "NOTFOUND - YOLO searching...");
        }
        
        // ==== VISUALIZATION ====
        if(detected && ball.valid) {
            // Bounding box
            Rect bbox(ball.center.x - ball.width/2, ball.center.y - ball.height/2,
                     ball.width, ball.height);
            rectangle(display_frame, bbox, Scalar(0, 255, 255), 2);
            
            // Center dot
            circle(display_frame, Point(ball.center), 5, Scalar(0, 0, 255), -1);
            
            // Scan area
            rectangle(display_frame,
                     Point(ball.scan_x_min, 0),
                     Point(ball.scan_x_max, frame.rows),
                     Scalar(255, 255, 0), 1);
            
            // Info text
            char info[128];
            snprintf(info, sizeof(info), "Area:%d Timeout:%.1fs", 
                     ball.area, ball.timeout_seconds);
            putText(display_frame, info, Point(5, 45),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
            
            char hsv_info[128];
            snprintf(hsv_info, sizeof(hsv_info), "HSV: H[%d,%d] S[%d,%d] V[%d,%d]",
                     ball.min_h, ball.max_h, ball.min_s, ball.max_s,
                     ball.min_v, ball.max_v);
            putText(display_frame, hsv_info, Point(5, 60),
                    FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255), 1);
        }
        
        // FPS and state
        calculate_fps();
        char fps_text[64];
        snprintf(fps_text, sizeof(fps_text), "FPS:%.1f | State:%s", 
                 fps, state == NOTFOUND ? "NOTFOUND" : "FOUND");
        putText(display_frame, fps_text, Point(5, 15),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
        
        // State color indicator
        putText(display_frame, state == NOTFOUND ? "YOLO" : "HSV", Point(5, 30),
                FONT_HERSHEY_SIMPLEX, 0.4,
                state == NOTFOUND ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 1);
        
        imshow("VISION_CPP", display_frame);
        waitKey(1);

        ros::spinOnce();
    }
    
    capture.stop();
    ROS_INFO("Vision system shutdown");
    return 0;
}
