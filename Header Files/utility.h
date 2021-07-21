#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <string>
#include <vector>

constexpr float	CONF_THRESHOLD = 0.5f;
constexpr float NMS_THRESHOLD  = 0.4;   // maximum suppression threshold
constexpr int INP_WIDTH = 416;  // Width of network's input image
constexpr int INP_HEIGHT = 416;

class ObjDetection {
	std::string filePath;
	std::string modelWeights;
	std::string netConfig;
	std::string classFileName;
	cv::dnn::Net network;
	std::vector<std::string> class_categories; // Save all classes

	std::vector<std::string> getOutputLayersName(const cv::dnn::Net& net);
	void drawPredictedBox(float confidenceScore, cv::Mat& frame, cv::Rect curBox, int clsId);
	void postProcessing(cv::Mat& inpFrame, const std::vector<cv::Mat>& outputBlob);

public:
	ObjDetection(const std::string path) : filePath(path) {}
	void loadModelClasses();
	void loadNetwork();
	void processImage();
};

#endif // ! UTILITY_H

