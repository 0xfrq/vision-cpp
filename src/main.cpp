#include <ros/ros.h>
#include <ros/package.h>

#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>

#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <deque>

#include "yolo_onnx.hpp"

using namespace cv;
using namespace std;

/* =========================
   UTIL
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
   TRACKING STATES
   ========================= */
enum DetectState { 
    SEARCH_YOLO = 0,      // Full YOLO search
    TRACK_HSV_STRICT = 1, // HSV tracking with strict parameters
    TRACK_HSV_RELAXED = 2,// HSV tracking with relaxed parameters
    TRACK_PREDICT = 3     // Kalman prediction when temporarily lost
};

DetectState state = SEARCH_YOLO;
DetectState prev_state = SEARCH_YOLO;

/* =========================
   DETECTION CONFIDENCE
   ========================= */
struct DetectionQuality {
    float confidence = 0.0f;
    int stable_frames = 0;
    bool is_reliable = false;
    double last_update = 0.0;
};

DetectionQuality detection_quality;

/* =========================
   HSV TRACKING PARAMETERS
   ========================= */
struct HSVParams {
    int min_h, min_s, min_v;
    int max_h, max_s, max_v;
    bool valid = false;
    int update_count = 0;
};

HSVParams hsv_strict;   // Tight HSV range
HSVParams hsv_relaxed;  // Wider HSV range for recovery

/* =========================
   BALL STATE
   ========================= */
struct BallState {
    Point2f center;
    Point2f velocity;
    int area;
    Rect bbox;
    bool initialized = false;
    deque<Point2f> history;  // Position history for smoothing
    deque<int> area_history; // Area history
    
    void addHistory(Point2f pos, int a) {
        history.push_back(pos);
        area_history.push_back(a);
        if(history.size() > 5) {
            history.pop_front();
            area_history.pop_front();
        }
    }
    
    Point2f getSmoothedPosition() {
        if(history.empty()) return center;
        Point2f sum(0,0);
        for(auto& p : history) sum += p;
        return sum * (1.0f / history.size());
    }
    
    int getSmoothedArea() {
        if(area_history.empty()) return area;
        int sum = 0;
        for(auto& a : area_history) sum += a;
        return sum / area_history.size();
    }
};

BallState ball;

/* =========================
   KALMAN FILTER
   ========================= */
KalmanFilter kalman(4, 2, 0);
bool kalman_initialized = false;

void initKalman(Point2f pos) {
    kalman.transitionMatrix = (Mat_<float>(4, 4) << 
        1,0,1,0,  // x' = x + vx
        0,1,0,1,  // y' = y + vy
        0,0,1,0,  // vx' = vx
        0,0,0,1); // vy' = vy
    
    kalman.measurementMatrix = (Mat_<float>(2, 4) << 
        1,0,0,0,
        0,1,0,0);
    
    setIdentity(kalman.processNoiseCov, Scalar::all(1e-2));
    setIdentity(kalman.measurementNoiseCov, Scalar::all(1e-1));
    setIdentity(kalman.errorCovPost, Scalar::all(1));
    
    kalman.statePost.at<float>(0) = pos.x;
    kalman.statePost.at<float>(1) = pos.y;
    kalman.statePost.at<float>(2) = 0;
    kalman.statePost.at<float>(3) = 0;
    
    kalman_initialized = true;
}

Point2f predictKalman() {
    if(!kalman_initialized) return ball.center;
    Mat prediction = kalman.predict();
    return Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

void updateKalman(Point2f measured) {
    if(!kalman_initialized) {
        initKalman(measured);
        return;
    }
    Mat measurement = (Mat_<float>(2, 1) << measured.x, measured.y);
    kalman.correct(measurement);
}

/* =========================
   SCAN AREA
   ========================= */
struct ScanArea {
    int x_min, x_max;
    int y_min, y_max;
    
    void expandFromBall(Point2f center, int ball_area, Size frame_size) {
        int expansion = (ball_area < 5000) ? 120 : 80;
        x_min = max(0, (int)(center.x - expansion));
        x_max = min(frame_size.width, (int)(center.x + expansion));
        y_min = max(0, (int)(center.y - expansion/2));
        y_max = min(frame_size.height, (int)(center.y + expansion/2));
    }
    
    void expandForRecovery(Size frame_size) {
        x_min = max(0, x_min - 50);
        x_max = min(frame_size.width, x_max + 50);
        y_min = max(0, y_min - 30);
        y_max = min(frame_size.height, y_max + 30);
    }
    
    bool contains(Point2f pt) {
        return pt.x >= x_min && pt.x <= x_max && 
               pt.y >= y_min && pt.y <= y_max;
    }
};

ScanArea scan_area;

/* =========================
   TRACKING COUNTERS
   ========================= */
int frames_tracked = 0;
int frames_lost = 0;
int yolo_attempts = 0;
int hsv_success_streak = 0;

/* =========================
   PERFORMANCE PARAMS
   ========================= */
constexpr double FISHEYE = 1.0;
constexpr int MIN_BALL_AREA = (int)(2500 * FISHEYE);
constexpr int MAX_FRAMES_PREDICT = 25;      // Predict for up to 25 frames
constexpr int MIN_FRAMES_FOR_YOLO = 35;     // Only go to YOLO after 35 lost frames
constexpr float YOLO_CONF_THRESHOLD = 0.35f; // Lower for faster detection
constexpr int MIN_STABLE_FRAMES = 3;        // Frames needed for reliable lock

/* =========================
   FPS TRACKING
   ========================= */
int frame_counter = 0;
double fps_start_time = 0.0;
double fps = 0.0;
constexpr double FPS_DISPLAY_INTERVAL = 0.3;

/* =========================
   FPS CALCULATOR
   ========================= */
void calculate_fps() {
    frame_counter++;
    double current_time = ros::Time::now().toSec();
    double elapsed = current_time - fps_start_time;
    
    if(elapsed >= FPS_DISPLAY_INTERVAL) {
        fps = frame_counter / elapsed;
        frame_counter = 0;
        fps_start_time = current_time;
    }
}

/* =========================
   HSV EXTRACTION - MULTI-POINT SAMPLING
   ========================= */
void extractHSV(const Mat& img, const Rect& bbox, HSVParams& params, bool relaxed = false) {
    vector<Point> sample_points;
    int x1 = bbox.x;
    int y1 = bbox.y;
    int x2 = bbox.x + bbox.width;
    int y2 = bbox.y + bbox.height;
    Point mid((x1+x2)/2, (y1+y2)/2);

    // 7-point sampling pattern
    sample_points = {
        mid,
        {(x1+mid.x)/2, (y1+mid.y)/2},
        {(x2+mid.x)/2, (y2+mid.y)/2},
        {(mid.x+x2)/2, (mid.y+y1)/2},
        {(x1+mid.x)/2, (y1+y2+mid.y)/2},
        {mid.x, (mid.y+y1+bbox.height/5)/2},
        {mid.x, (mid.y+y2)/2}
    };

    vector<int> H, S, V;
    for(auto& pt : sample_points) {
        int px = clamp(pt.x, 0, img.cols-1);
        int py = clamp(pt.y, 0, img.rows-1);
        Vec3b bgr = img.at<Vec3b>(py, px);
        
        Mat hsv_sample;
        Mat bgr_mat(1, 1, CV_8UC3, Scalar(bgr[0], bgr[1], bgr[2]));
        cvtColor(bgr_mat, hsv_sample, COLOR_BGR2HSV);
        Vec3b hsv_val = hsv_sample.at<Vec3b>(0, 0);
        
        H.push_back(hsv_val[0]);
        S.push_back(hsv_val[1]);
        V.push_back(hsv_val[2]);
    }

    if(relaxed) {
        // Relaxed parameters for recovery
        params.min_h = max(0, *min_element(H.begin(), H.end()) - 5);
        params.max_h = min(180, *max_element(H.begin(), H.end()) + 5);
        params.min_s = max(0, *min_element(S.begin(), S.end()) - 30);
        params.max_s = min(255, *max_element(S.begin(), S.end()) + 20);
        params.min_v = max(0, *min_element(V.begin(), V.end()) - 30);
        params.max_v = min(255, *max_element(V.begin(), V.end()) + 30);
    } else {
        // Strict parameters for tracking
        params.min_h = *min_element(H.begin(), H.end());
        params.max_h = min(*max_element(H.begin(), H.end()), 33);
        params.min_s = max(*min_element(S.begin(), S.end()), 160);
        params.max_s = *max_element(S.begin(), S.end());
        params.min_v = *min_element(V.begin(), V.end());
        params.max_v = *max_element(V.begin(), V.end());
    }
    
    params.valid = true;
    params.update_count++;
}

/* =========================
   FIELD MASKING - OPTIMIZED
   ========================= */
Mat field_mask_cache;
bool field_mask_valid = false;
int field_mask_frame_count = 0;

Mat extractField(const Mat& img, bool force_update = false) {
    // Cache field mask for 30 frames
    if(!field_mask_valid || force_update || field_mask_frame_count > 30) {
        Mat hsv, mask;
        cvtColor(img, hsv, COLOR_BGR2HSV);
        
        Scalar lower(35, 40, 40);
        Scalar upper(85, 255, 255);
        inRange(hsv, lower, upper, mask);
        
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5,5));
        erode(mask, mask, kernel, Point(-1,-1), 2);
        dilate(mask, mask, kernel, Point(-1,-1), 5);
        
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
        
        field_mask_cache = Mat::zeros(img.size(), CV_8UC1);
        if(!contours.empty()) {
            auto max_contour = *max_element(contours.begin(), contours.end(),
                [](const vector<Point>&a, const vector<Point>&b) {
                    return contourArea(a) < contourArea(b);
                });
            vector<Point> hull;
            convexHull(max_contour, hull);
            fillConvexPoly(field_mask_cache, hull, Scalar(255));
        }
        
        field_mask_valid = true;
        field_mask_frame_count = 0;
    }
    
    field_mask_frame_count++;
    
    Mat result;
    bitwise_and(img, img, result, field_mask_cache);
    return result;
}

/* =========================
   CONTOUR SCORING
   ========================= */
struct ContourScore {
    float total_score = 0.0f;
    Point2f center;
    int area = 0;
    Rect bbox;
    bool valid = false;
};

ContourScore scoreContour(const vector<Point>& contour, Point2f predicted_pos, 
                          int expected_area, const ScanArea& scan) {
    ContourScore score;
    
    double area = contourArea(contour);
    if(area < MIN_BALL_AREA) return score;
    
    Rect bbox = boundingRect(contour);
    Point2f center(bbox.x + bbox.width/2.0f, bbox.y + bbox.height/2.0f);
    
    // Check scan area
    if(!scan.contains(center)) return score;
    
    // Distance score (closer to prediction = better)
    float dist = norm(center - predicted_pos);
    float dist_score = 1.0f / (1.0f + dist/50.0f);
    
    // Area similarity score
    float area_ratio = (float)area / expected_area;
    float area_score = 1.0f / (1.0f + abs(area_ratio - 1.0f)*3.0f);
    
    // Circularity score (ball should be roundish)
    double perimeter = arcLength(contour, true);
    float circularity = (4 * M_PI * area) / (perimeter * perimeter);
    float circ_score = clamp(circularity, 0.0f, 1.0f);
    
    // Aspect ratio score (ball should be squarish)
    float aspect = (float)bbox.width / bbox.height;
    float aspect_score = 1.0f / (1.0f + abs(aspect - 1.0f)*2.0f);
    
    // Combined score
    score.total_score = dist_score * 0.4f + area_score * 0.3f + 
                       circ_score * 0.2f + aspect_score * 0.1f;
    score.center = center;
    score.area = (int)area;
    score.bbox = bbox;
    score.valid = true;
    
    return score;
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
   YOLO DETECTION WITH CACHING
   ========================= */
struct YoloResult {
    bool found = false;
    Rect bbox;
    float confidence = 0.0f;
    Point2f center;
    int area = 0;
};

YoloResult performYOLO(YoloONNX& yolo, const Mat& frame) {
    YoloResult result;
    auto dets = yolo.infer(frame);
    
    float best_conf = 0.0f;
    for(auto& d : dets) {
        if(d.class_id != 0 || d.conf < YOLO_CONF_THRESHOLD) continue;
        
        if(d.conf > best_conf) {
            best_conf = d.conf;
            result.bbox = d.box;
            result.confidence = d.conf;
            result.center = Point2f(d.box.x + d.box.width/2.0f, 
                                   d.box.y + d.box.height/2.0f);
            result.area = d.box.area();
            result.found = true;
        }
    }
    
    return result;
}

/* =========================
   HSV TRACKING
   ========================= */
struct HSVResult {
    bool found = false;
    Point2f center;
    int area = 0;
    Rect bbox;
    float confidence = 0.0f;
};

HSVResult performHSV(const Mat& frame, const HSVParams& params, 
                     const ScanArea& scan, Point2f predicted_pos, 
                     int expected_area, bool use_field_mask) {
    HSVResult result;
    
    if(!params.valid) return result;
    
    // Optional field masking
    Mat work_frame = use_field_mask ? extractField(frame) : frame.clone();
    
    Mat hsv, mask;
    cvtColor(work_frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, Scalar(params.min_h, params.min_s, params.min_v),
                 Scalar(params.max_h, params.max_s, params.max_v), mask);
    
    Mat kernel = Mat::ones(5, 5, CV_8U);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    ContourScore best_score;
    for(auto& c : contours) {
        ContourScore score = scoreContour(c, predicted_pos, expected_area, scan);
        if(score.valid && score.total_score > best_score.total_score) {
            best_score = score;
        }
    }
    
    if(best_score.valid && best_score.total_score > 0.3f) {
        result.found = true;
        result.center = best_score.center;
        result.area = best_score.area;
        result.bbox = best_score.bbox;
        result.confidence = best_score.total_score;
    }
    
    return result;
}

/* =========================
   STATE MACHINE LOGIC
   ========================= */
void updateDetectionQuality(bool detected, float confidence = 0.0f) {
    if(detected) {
        detection_quality.stable_frames++;
        detection_quality.confidence = max(detection_quality.confidence, confidence);
        detection_quality.is_reliable = (detection_quality.stable_frames >= MIN_STABLE_FRAMES);
    } else {
        detection_quality.stable_frames = max(0, detection_quality.stable_frames - 1);
        detection_quality.confidence *= 0.9f;
        if(detection_quality.stable_frames < MIN_STABLE_FRAMES/2) {
            detection_quality.is_reliable = false;
        }
    }
    detection_quality.last_update = ros::Time::now().toSec();
}

void transitionState(DetectState new_state, const string& reason = "") {
    if(state != new_state) {
        prev_state = state;
        state = new_state;
        if(!reason.empty()) {
            ROS_INFO("State: %d -> %d (%s)", prev_state, new_state, reason.c_str());
        }
    }
}

/* =========================
   MAIN DETECTION PIPELINE
   ========================= */
struct DetectionResult {
    bool found = false;
    Point2f center;
    int area = 0;
    Rect bbox;
    DetectState recommended_state;
    string debug_msg;
};

DetectionResult detectBall(YoloONNX& yolo, const Mat& frame) {
    DetectionResult result;
    result.recommended_state = state;
    
    // Get prediction from Kalman
    Point2f predicted_pos = kalman_initialized ? predictKalman() : ball.center;
    
    switch(state) {
        case SEARCH_YOLO: {
            // Full YOLO search
            YoloResult yolo_res = performYOLO(yolo, frame);
            if(yolo_res.found) {
                result.found = true;
                result.center = yolo_res.center;
                result.area = yolo_res.area;
                result.bbox = yolo_res.bbox;
                
                // Extract both strict and relaxed HSV
                extractHSV(frame, yolo_res.bbox, hsv_strict, false);
                extractHSV(frame, yolo_res.bbox, hsv_relaxed, true);
                
                // Initialize Kalman
                initKalman(yolo_res.center);
                
                // Setup scan area
                scan_area.expandFromBall(yolo_res.center, yolo_res.area, frame.size());
                
                // Transition to strict HSV tracking
                result.recommended_state = TRACK_HSV_STRICT;
                result.debug_msg = "YOLO->HSV_STRICT";
                
                frames_tracked = 0;
                frames_lost = 0;
                hsv_success_streak = 0;
            } else {
                result.debug_msg = "YOLO searching...";
            }
            yolo_attempts++;
            break;
        }
        
        case TRACK_HSV_STRICT: {
            // Try strict HSV first
            HSVResult hsv_res = performHSV(frame, hsv_strict, scan_area, predicted_pos,
                                          ball.area, frames_tracked > 10);
            
            if(hsv_res.found) {
                result.found = true;
                result.center = hsv_res.center;
                result.area = hsv_res.area;
                result.bbox = hsv_res.bbox;
                
                updateKalman(hsv_res.center);
                scan_area.expandFromBall(hsv_res.center, hsv_res.area, frame.size());
                
                frames_tracked++;
                frames_lost = 0;
                hsv_success_streak++;
                result.debug_msg = "HSV_STRICT OK";
                
                // Periodically update HSV from current detection
                if(frames_tracked % 15 == 0) {
                    extractHSV(frame, hsv_res.bbox, hsv_strict, false);
                }
            } else {
                frames_lost++;
                result.recommended_state = TRACK_HSV_RELAXED;
                result.debug_msg = "HSV_STRICT->RELAXED";
            }
            break;
        }
        
        case TRACK_HSV_RELAXED: {
            // Try relaxed HSV
            scan_area.expandForRecovery(frame.size());
            HSVResult hsv_res = performHSV(frame, hsv_relaxed, scan_area, predicted_pos,
                                          ball.area, true);
            
            if(hsv_res.found) {
                result.found = true;
                result.center = hsv_res.center;
                result.area = hsv_res.area;
                result.bbox = hsv_res.bbox;
                
                updateKalman(hsv_res.center);
                scan_area.expandFromBall(hsv_res.center, hsv_res.area, frame.size());
                
                frames_tracked++;
                frames_lost = 0;
                result.recommended_state = TRACK_HSV_STRICT;
                result.debug_msg = "HSV_RELAXED OK->STRICT";
                
                // Update HSV params
                extractHSV(frame, hsv_res.bbox, hsv_strict, false);
                extractHSV(frame, hsv_res.bbox, hsv_relaxed, true);
            } else {
                frames_lost++;
                if(frames_lost < MAX_FRAMES_PREDICT) {
                    result.recommended_state = TRACK_PREDICT;
                    result.debug_msg = "RELAXED->PREDICT";
                } else {
                    result.recommended_state = TRACK_PREDICT;
                    result.debug_msg = "RELAXED->PREDICT (long)";
                }
            }
            break;
        }
        
        case TRACK_PREDICT: {
            // Use Kalman prediction
            Point2f kalman_pred = predictKalman();
            
            // Try to find ball near prediction with very relaxed params
            scan_area.expandForRecovery(frame.size());
            HSVResult hsv_res = performHSV(frame, hsv_relaxed, scan_area, kalman_pred,
                                          ball.area, true);
            
            if(hsv_res.found) {
                result.found = true;
                result.center = hsv_res.center;
                result.area = hsv_res.area;
                result.bbox = hsv_res.bbox;
                
                updateKalman(hsv_res.center);
                scan_area.expandFromBall(hsv_res.center, hsv_res.area, frame.size());
                
                frames_lost = 0;
                result.recommended_state = TRACK_HSV_STRICT;
                result.debug_msg = "PREDICT->RECOVERED";
                
                extractHSV(frame, hsv_res.bbox, hsv_strict, false);
                extractHSV(frame, hsv_res.bbox, hsv_relaxed, true);
            } else {
                frames_lost++;
                
                // Keep using prediction
                result.found = true;
                result.center = kalman_pred;
                result.area = ball.area;
                result.bbox = Rect(kalman_pred.x - 20, kalman_pred.y - 20, 40, 40);
                result.debug_msg = "PREDICTING (" + to_string(frames_lost) + ")";
                
                // Only go to YOLO after many frames lost
                if(frames_lost > MIN_FRAMES_FOR_YOLO) {
                    result.recommended_state = SEARCH_YOLO;
                    result.debug_msg = "PREDICT->YOLO (timeout)";
                    result.found = false;
                }
            }
            break;
        }
    }
    
    return result;
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

    ROS_INFO("=== Advanced Ball Tracking System ===");
    ROS_INFO("YOLO Confidence: %.2f", YOLO_CONF_THRESHOLD);
    ROS_INFO("Max Prediction Frames: %d", MAX_FRAMES_PREDICT);
    ROS_INFO("Min Frames Before YOLO: %d", MIN_FRAMES_FOR_YOLO);
    ROS_INFO("=====================================");

    ThreadedCapture capture(0);
    
    fps_start_time = ros::Time::now().toSec();
    ROS_INFO("Vision system ready");

    while(ros::ok()) {
        Mat frame = capture.read();
        if(frame.empty()) {
            ros::spinOnce();
            continue;
        }
        
        Mat display_frame = frame.clone();

        // ==== CORE DETECTION ====
        DetectionResult detection = detectBall(yolo, frame);
        
        // Update state machine
        if(detection.recommended_state != state) {
            transitionState(detection.recommended_state, detection.debug_msg);
        }
        
        // Update ball state
        if(detection.found) {
            ball.center = detection.center;
            ball.area = detection.area;
            ball.bbox = detection.bbox;
            ball.initialized = true;
            
            // Calculate velocity
            if(!ball.history.empty()) {
                ball.velocity = detection.center - ball.history.back();
            }
            
            ball.addHistory(detection.center, detection.area);
            
            updateDetectionQuality(true, 0.8f);
        } else {
            updateDetectionQuality(false);
        }
        
        // ==== PUBLISH ROS MESSAGES ====
        v2_detection::BallState bs;
        v2_detection::BallCoordinate bc;
        v2_detection::Ballarea ba;
        
        if(ball.initialized && detection.found) {
            bs.ball_status = "FOUND";
            
            Point2f smooth_pos = ball.getSmoothedPosition();
            bc.pos_x = clamp(smooth_pos.x / frame.cols * 2 - 1, -1.0f, 1.0f);
            bc.pos_y = clamp(smooth_pos.y / frame.rows * 2 - 1, -1.0f, 1.0f);
            bc.obj_size = ball.area;
            
            ba.ballarea = ball.area;
            
            pub_state.publish(bs);
            pub_coord.publish(bc);
            pub_area.publish(ba);
            
            // Debug output
            if(state == TRACK_HSV_STRICT || state == TRACK_HSV_RELAXED) {
                ROS_INFO_THROTTLE(0.5, "%s | Area:%d | Tracked:%d frames", 
                                 detection.debug_msg.c_str(), ball.area, frames_tracked);
            }
        } else {
            bs.ball_status = "NOTFOUND";
            pub_state.publish(bs);
            
            if(state == SEARCH_YOLO) {
                ROS_INFO_THROTTLE(1.0, "YOLO Searching... (attempt %d)", yolo_attempts);
            } else if(state == TRACK_PREDICT) {
                ROS_INFO_THROTTLE(0.5, "%s", detection.debug_msg.c_str());
            }
        }
        
        // ==== VISUALIZATION ====
        if(ball.initialized && detection.found) {
            // Bounding box
            rectangle(display_frame, ball.bbox, Scalar(0, 255, 255), 2);
            
            // Center dot
            circle(display_frame, Point(ball.center), 5, Scalar(0, 0, 255), -1);
            
            // Velocity vector
            if(norm(ball.velocity) > 1.0) {
                Point2f vel_end = ball.center + ball.velocity * 5.0f;
                arrowedLine(display_frame, Point(ball.center), Point(vel_end),
                           Scalar(255, 0, 255), 2);
            }
            
            // Kalman prediction
            if(kalman_initialized) {
                Point2f kalman_pred = predictKalman();
                circle(display_frame, Point(kalman_pred), 3, Scalar(0, 255, 0), -1);
                line(display_frame, Point(ball.center), Point(kalman_pred),
                     Scalar(0, 255, 0), 1);
            }
            
            // Scan area
            rectangle(display_frame, 
                     Point(scan_area.x_min, scan_area.y_min),
                     Point(scan_area.x_max, scan_area.y_max),
                     Scalar(255, 255, 0), 1);
            
            // Position history trail
            if(ball.history.size() > 1) {
                for(size_t i = 1; i < ball.history.size(); i++) {
                    line(display_frame, Point(ball.history[i-1]), 
                         Point(ball.history[i]), Scalar(128, 128, 255), 1);
                }
            }
        }
        
        // ==== FPS AND STATUS ====
        calculate_fps();
        
        char fps_text[128];
        snprintf(fps_text, sizeof(fps_text), "FPS:%.1f | State:%d | Lost:%d | Track:%d", 
                 fps, state, frames_lost, frames_tracked);
        putText(display_frame, fps_text, Point(5, 15),
                FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
        
        // State info
        const char* state_names[] = {"YOLO", "HSV_STRICT", "HSV_RELAX", "PREDICT"};
        char state_text[64];
        snprintf(state_text, sizeof(state_text), "%s", state_names[state]);
        putText(display_frame, state_text, Point(5, 30),
                FONT_HERSHEY_SIMPLEX, 0.4, 
                state == SEARCH_YOLO ? Scalar(0, 0, 255) : Scalar(0, 255, 0), 1);
        
        // Detection quality
        if(detection_quality.is_reliable) {
            char qual_text[64];
            snprintf(qual_text, sizeof(qual_text), "RELIABLE (%d)", 
                     detection_quality.stable_frames);
            putText(display_frame, qual_text, Point(5, 45),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);
        }
        
        // HSV ranges (when tracking)
        if(hsv_strict.valid && state != SEARCH_YOLO) {
            char hsv_text[128];
            snprintf(hsv_text, sizeof(hsv_text), "HSV: H[%d,%d] S[%d,%d] V[%d,%d]",
                     hsv_strict.min_h, hsv_strict.max_h,
                     hsv_strict.min_s, hsv_strict.max_s,
                     hsv_strict.min_v, hsv_strict.max_v);
            putText(display_frame, hsv_text, Point(5, 60),
                    FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255), 1);
        }
        
        imshow("VISION_CPP", display_frame);
        waitKey(1);

        ros::spinOnce();
    }
    
    capture.stop();
    ROS_INFO("Vision system shutdown");
    return 0;
}
