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
	ObjDetection obj("your_path_to_test_image/testImage.PNG");
	obj.loadNetwork();
	obj.processImage();
	waitKey(0);
	return 0;
}
