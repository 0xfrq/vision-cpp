#include <ros/ros.h>
#include <ros/package.h>

#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

#include "yolo_onnx.hpp"

using namespace std;
using namespace cv;

/* =============================
   Ball Detection State
   ============================= */
enum BallStatus {
    NOTFOUND = 0,
    FOUND = 1
};

int main(int argc, char** argv) {

    /* =============================
       ROS INIT
       ============================= */
    ros::init(argc, argv, "vision_yolo_cpp");
    ros::NodeHandle nh;

    /* =============================
       LOAD MODEL
       ============================= */
    string pkg_path = ros::package::getPath("vision_cpp");
    string model_path = pkg_path + "/src/best.onnx";

    ROS_INFO("[VISION] Loading YOLO model: %s", model_path.c_str());
    YoloONNX yolo(model_path);

    /* =============================
       ROS PUBLISHERS
       ============================= */
    ros::Publisher pub_state =
        nh.advertise<v2_detection::BallState>(
            "/DEWO/image_processing/deteksi_bola/ball_state", 1);

    ros::Publisher pub_coord =
        nh.advertise<v2_detection::BallCoordinate>(
            "/DEWO/image_processing/deteksi_bola/coordinate", 1);

    ros::Publisher pub_area =
        nh.advertise<v2_detection::Ballarea>(
            "/DEWO/image_processing/deteksi_bola/ball_area", 1);

    /* =============================
       CAMERA
       ============================= */
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);

    if (!cap.isOpened()) {
        ROS_ERROR("[VISION] Kamera gagal dibuka");
        return -1;
    }

    /* =============================
       BALL MEMORY & STATE
       ============================= */
    BallStatus ball_status = NOTFOUND;

    Rect last_ball_box;
    int last_area = 0;

    int lost_counter = 0;
    int confirm_counter = 0;

    const int LOST_THRESHOLD = 5;
    const int CONFIRM_THRESHOLD = 2;

    double fps = 0.0;
    ros::Rate loop_rate(60);

    /* =============================
       MAIN LOOP
       ============================= */
    while (ros::ok()) {

        auto t0 = chrono::high_resolution_clock::now();

        Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        vector<Detection> detections;
        Rect search_roi;
        Mat input_frame;

        /* =============================
           ROI MODE (TRACKING)
           ============================= */
        if (ball_status == FOUND) {

            int pad = 40;
            search_roi = last_ball_box;

            search_roi.x = max(0, search_roi.x - pad);
            search_roi.y = max(0, search_roi.y - pad);
            search_roi.width = min(frame.cols - search_roi.x,
                                    search_roi.width + 2 * pad);
            search_roi.height = min(frame.rows - search_roi.y,
                                     search_roi.height + 2 * pad);

            input_frame = frame(search_roi);
            detections = yolo.infer(input_frame);

        } else {
            /* FULL SEARCH */
            input_frame = frame;
            detections = yolo.infer(frame);
        }

        bool detected = false;
        Rect best_box;

        /* =============================
           DETECTION FILTERING
           ============================= */
        if (!detections.empty()) {

            for (auto& det : detections) {

                Rect box = det.box;

                if (ball_status == FOUND) {
                    box.x += search_roi.x;
                    box.y += search_roi.y;
                }

                int area = box.width * box.height;

                /* area consistency */
                if (ball_status == FOUND) {
                    if (area < last_area * 0.5 || area > last_area * 1.5)
                        continue;
                }

                /* movement consistency */
                if (ball_status == FOUND) {
                    Point c_new(box.x + box.width / 2,
                                box.y + box.height / 2);
                    Point c_old(last_ball_box.x + last_ball_box.width / 2,
                                last_ball_box.y + last_ball_box.height / 2);

                    if (norm(c_new - c_old) > 80)
                        continue;
                }

                best_box = box;
                detected = true;
                break;
            }
        }

        /* =============================
           STATE MACHINE
           ============================= */
        if (detected) {

            confirm_counter++;
            lost_counter = 0;

            last_ball_box = best_box;
            last_area = best_box.width * best_box.height;

            if (confirm_counter >= CONFIRM_THRESHOLD)
                ball_status = FOUND;

        } else {

            confirm_counter = 0;
            lost_counter++;

            if (lost_counter > LOST_THRESHOLD)
                ball_status = NOTFOUND;
        }

        /* =============================
           ROS MESSAGE
           ============================= */
        v2_detection::BallState msg_state;
        v2_detection::BallCoordinate msg_coord;
        v2_detection::Ballarea msg_area;

        if (ball_status == FOUND) {

            int cx = last_ball_box.x + last_ball_box.width / 2;
            int cy = last_ball_box.y + last_ball_box.height / 2;

            msg_state.ball_status = "FOUND";

            msg_coord.pos_x = (float(cx) / frame.cols) * 2.0f - 1.0f;
            msg_coord.pos_y = (float(cy) / frame.rows) * 2.0f - 1.0f;
            msg_coord.obj_size = last_area;

            msg_area.ballarea = last_area;

            pub_coord.publish(msg_coord);
            pub_area.publish(msg_area);

            rectangle(frame, last_ball_box, Scalar(0, 255, 0), 2);

        } else {

            msg_state.ball_status = "NOTFOUND";
        }

        pub_state.publish(msg_state);

        /* =============================
           FPS & DISPLAY
           ============================= */
        auto t1 = chrono::high_resolution_clock::now();
        double dt = chrono::duration<double>(t1 - t0).count();
        if (dt > 0) fps = 0.9 * fps + 0.1 * (1.0 / dt);

        putText(frame,
                "FPS: " + to_string(int(fps)) +
                " | " + (ball_status == FOUND ? "FOUND" : "NOTFOUND"),
                Point(10, 20),
                FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(0, 255, 0), 1);

        imshow("vision_yolo_cpp", frame);
        waitKey(1);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
