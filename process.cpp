#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() {
    cv::Mat img = cv::imread("test.png");
    if (img.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    // Convert to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Haar cascades for face and eye detection
    cv::CascadeClassifier faces, eyes;
    faces.load(cv::samples::findFile("haarcascades/haarcascade_frontalface_default.xml"));
    eyes.load(cv::samples::findFile("haarcascades/haarcascade_eye.xml"));

    std::vector<cv::Rect> faceRects, eyeRects;
    faces.detectMultiScale(img, faceRects, 1.1, 2, 0, cv::Size(30, 30));
    eyes.detectMultiScale(img, eyeRects, 1.1, 2, 0, cv::Size(30, 30));

    // Process each detected face
    for (const auto& face : faceRects) {
        cv::Mat faceROI = img(face);
        cv::Mat hsvFaceROI;
        cv::cvtColor(faceROI, hsvFaceROI, cv::COLOR_RGB2HSV);
        for (int y = 0; y < hsvFaceROI.rows; y++) {
            for (int x = 0; x < hsvFaceROI.cols; x++) {
                hsvFaceROI.at<cv::Vec3b>(y, x)[1] = std::min(255, static_cast<int>(hsvFaceROI.at<cv::Vec3b>(y, x)[1] * 1.5));
            }
        }
        cv::cvtColor(hsvFaceROI, faceROI, cv::COLOR_HSV2RGB);
    }

    // Process each detected eye
    for (const auto& eye : eyeRects) {
        cv::Mat eyeROI = img(eye);
        cv::Mat hsvEyeROI;
        cv::cvtColor(eyeROI, hsvEyeROI, cv::COLOR_RGB2HSV);
        for (int y = 0; y < hsvEyeROI.rows; y++) {
            for (int x = 0; x < hsvEyeROI.cols; x++) {
                hsvEyeROI.at<cv::Vec3b>(y, x)[1] = std::min(255, static_cast<int>(hsvEyeROI.at<cv::Vec3b>(y, x)[1] * 1.5));
            }
        }
        cv::cvtColor(hsvEyeROI, eyeROI, cv::COLOR_HSV2RGB);
    }

    // Convert back to BGR before saving
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imwrite("result.png", img);

    return 0;
}