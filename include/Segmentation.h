#ifndef __SEGMENTATION_H
#define __SEGMENTATION_H

#include "System.h"

#include "torch/script.h"
#include "torch/torch.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>

namespace ORB_SLAM2
{

class Segmentation {

private:
torch::jit::script::Module module;
torch::Device device = torch::Device(torch::kCUDA);

public:
Segmentation(std::string model_dir);
~Segmentation();
cv::Mat GetSegmentation(cv::Mat img);

};

}

#endif
