// ObjectDetection.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/core/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utility.h"

using namespace cv;
using namespace std;


int main()
{
	ObjDetection obj("E:/Backup Windows/code_repository/Visual Studio Projects/OpenCV Projects/ObjectDetection/testImage.PNG");
	obj.loadNetwork();
	obj.processImage();
	/*Mat image = Mat::zeros(300, 600, CV_8UC3);
	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
	imshow("Display Window", image);*/
	waitKey(0);
	return 0;
}
