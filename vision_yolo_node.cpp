#include <ros/ros.h>
#include <std_msgs/String.h>

#include <v2_detection/BallState.h>
#include <v2_detection/BallCoordinate.h>
#include <v2_detection/Ballarea.h>

#include <opencv2/opencv.hpp>
#include <chrono>

#include "yolo_onnx.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    // init ros node
    ros::init(argc, argv, "vision_yolo_cpp");
    ros::NodeHandle nh;

    // publisher
    ros::Publisher pub_state =
        nh.advertise<v2_detection::BallState>("/vision/ball_state", 1);

    ros::Publisher pub_coord =
        nh.advertise<v2_detection::BallCoordinate>("/vision/ball_coordinate", 1);

    ros::Publisher pub_area =
        nh.advertise<v2_detection::Ballarea>("/vision/ball_area", 1);

    // load model
    cout << "loading yolo model..." << endl;
    YoloONNX yolo("best.onnx");

    // open camera
    VideoCapture cap(0);
    cap.set(CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);

    if (!cap.isOpened()) {
        ROS_ERROR("kamera tidak bisa dibuka");
        return -1;
    }

    double fps = 0;
    ros::Rate loop_rate(60);

    while (ros::ok()) {

        auto t0 = chrono::high_resolution_clock::now();

        Mat frame;
        cap >> frame;
        if (frame.empty()) continue;

        auto detections = yolo.infer(frame);

        // message object
        v2_detection::BallState msg_state;
        v2_detection::BallCoordinate msg_coord;
        v2_detection::Ballarea msg_area;

        if (detections.empty()) {

            msg_state.ball_status = "NOTFOUND";
            pub_state.publish(msg_state);

        } else {

            // ambil deteksi pertama (bola)
            auto& det = detections[0];

            int cx = det.box.x + det.box.width / 2;
            int cy = det.box.y + det.box.height / 2;
            int area = det.box.width * det.box.height;

            // normalisasi -1 sampai 1 (seperti python)
            msg_coord.pos_x = (float(cx) / frame.cols) * 2.0f - 1.0f;
            msg_coord.pos_y = (float(cy) / frame.rows) * 2.0f - 1.0f;
            msg_coord.obj_size = area;

            msg_area.ballarea = area;
            msg_state.ball_status = "FOUND";

            pub_coord.publish(msg_coord);
            pub_area.publish(msg_area);
            pub_state.publish(msg_state);

            rectangle(frame, det.box, Scalar(0,255,0), 1);
        }

        // fps
        auto t1 = chrono::high_resolution_clock::now();
        double dt = chrono::duration<double>(t1 - t0).count();
        if (dt > 0) fps = 0.9 * fps + 0.1 * (1.0 / dt);

        putText(frame, "fps: " + to_string(int(fps)),
                Point(10,20), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(0,255,0), 1);

        imshow("vision_yolo_cpp", frame);
        waitKey(1);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}

