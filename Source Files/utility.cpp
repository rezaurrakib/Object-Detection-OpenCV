#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "utility.h"


void ObjDetection::loadModelClasses() {
	classFileName = "coco.names";
	// parsing the 'coco.names' file and save the classes as a list
	std::ifstream ifs(classFileName.c_str()); // Open the className file
	std::string class_name;
	while (getline(ifs, class_name)) {
		class_categories.push_back(class_name);
	}

}

void ObjDetection::loadNetwork() {
	netConfig = "yolov3.cfg"; // Load the YOLO config file
	modelWeights = "yolov3.weights";
	network = cv::dnn::readNetFromDarknet(netConfig, modelWeights); // Load the network file
	if (network.empty() == false) {
		std::cout << "Network Status: Not Empty";
	}
	else {
		std::cout << "Empty....";
	}
	
}

std::vector<std::string> ObjDetection::getOutputLayersName(const cv::dnn::Net& net) {
	std::vector<cv::String> names;

	if (names.empty()) {
		std::vector<int> outLayers = net.getUnconnectedOutLayers();
		std::vector<cv::String> layesNames = net.getLayerNames();

		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); i++) {
			names[i] = layesNames[outLayers[i] - 1];
		}
	}
	return names;
}

/** @brief Draw predicted Boxes
 *  @param
 *  @param
 */
void ObjDetection::drawPredictedBox(float confidenceScore, cv::Mat& frame, cv::Rect curBox, int clsId) {
	int left = curBox.x;
	int top = curBox.y;
	int right = curBox.x + curBox.width;
	int bottom = curBox.y + curBox.height;

	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 190, 70), 3);
	// get the label and corresponding confidence
	std::string lbl = cv::format("%.3f", confidenceScore);
	if (!class_categories.empty()) {
		CV_Assert(clsId < (int)class_categories.size());
		lbl = class_categories[clsId] + ":" + lbl;
	}

	int baseLine;
	cv::Size lblSize = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = std::max(top, lblSize.height);
	cv::rectangle(frame, cv::Point(left, top - round(1.5 * lblSize.height)), cv::Point(left + round(1.5 * lblSize.width),
		top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
	cv::putText(frame, lbl, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);

}

/** @brief Remove all the bounding boxes with low confidence
 *  @param inpFrame input image to the network
 *  @param out
 */
void ObjDetection::postProcessing(cv::Mat& inpFrame, const std::vector<cv::Mat>& outputBlob) {
	std::vector<int> clsId;
	std::vector<float> confidenceScores;
	std::vector<cv::Rect> bBoxes;

	// Bounding box scanning and retrieval of high confidence boxes
	for (int i = 0; i < outputBlob.size(); i++) {
		float* blobData = (float*) outputBlob[i].data;
		for (int j = 0; j < outputBlob[i].rows; j++) {
			// The first 5 element specifies 'center_x, center_y, width, height & confidence of bBox encloses an Obj'
			// From 6th element till the end represents confidence score for each class
			cv::Mat scores = outputBlob[i].row(j).colRange(5, outputBlob[i].cols);
			cv::Point clsIdPoint;
			double confidence;
			cv::minMaxLoc(scores, 0, &confidence, 0, &clsIdPoint);
			if (confidence > CONF_THRESHOLD) {
				int centerX = (int)(blobData[0] * inpFrame.cols);
				int centerY = (int)(blobData[1] * inpFrame.rows);
				int width = (int)(blobData[2] * inpFrame.cols);
				int height = (int)(blobData[3] * inpFrame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				clsId.push_back(clsIdPoint.x);
				confidenceScores.push_back((float)confidence);
				bBoxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(bBoxes, confidenceScores, CONF_THRESHOLD, NMS_THRESHOLD, indices);
	for (size_t i = 0; i < indices.size(); i++) {
		int idx = indices[i];
		cv::Rect box = bBoxes[idx];
		drawPredictedBox(confidenceScores[i], inpFrame, box, clsId[i]);
	}
}

void ObjDetection::processImage() {
	cv::Mat blob;
	cv::Mat input = cv::imread(filePath);
	// Converting image to a blob
	cv::dnn::blobFromImage(input, blob, 1/255.0, cv::Size(INP_WIDTH, INP_HEIGHT), cv::Scalar(0, 0, 0), true, false);
	network.setInput(blob);
	
	// Runs the forward pass to get output of the output layers
	std::vector<cv::Mat> output;
	network.forward(output, getOutputLayersName(network));
	postProcessing(input, output);

	std::vector<double> layersTimes;
	double freq = cv::getTickFrequency() / 1000;
	double t = network.getPerfProfile(layersTimes) / freq;
	std::string label = cv::format("Inference time for a frame : %.3f ms", t);
	cv::putText(input, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
	
	cv::Mat detFrame;
	input.convertTo(detFrame, CV_8U);
	std::string outputFile = "yolo_out_py.jpg";
	cv::imwrite(outputFile, detFrame);
	cv::imshow("Face Detection Output", input);
	//}

}
