#include "Segmentation.h"

namespace F = torch::nn::functional;

namespace ORB_SLAM2
{

Segmentation::Segmentation(std::string model_dir) {
    std::cout << "loading segmentation model..." << std::endl;
    try {
        this->module = torch::jit::load(model_dir);
    }
    catch (const c10::Error& ) {
        std::cerr << "error loading the model\n";
    }
    if (torch::cuda::is_available())
    {
        std::cout << "cudu support: true, now device is GPU" << std::endl;
        this->device = torch::Device(torch::kCUDA);
    }
    else
    {
        std::cout << "cudu support: false, now device is CPU" << std::endl;
        this->device = torch::Device(torch::kCPU);
    }
    this->module.to(this->device);
}

Segmentation::~Segmentation() {}

cv::Mat Segmentation::GetSegmentation(cv::Mat img) {
    cv::Mat input_image;
    cv::cvtColor(img, input_image, cv::COLOR_BGR2RGB);
    cv::Size picSize = input_image.size();
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);

    torch::Tensor tensor_image = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols,3}, torch::kFloat32).to(this->device);
    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
    tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
    tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
    // std::cout<<"preprocess finish" << "\n";

    torch::Tensor output = module.forward({tensor_image}).toTensor();
    // std::cout<<"predict finish"<<"\n";
    output = output.argmax(1).squeeze(0).to(torch::kU8).to(torch::kCPU);
    cv::Mat output_image(picSize, CV_8U, output.data_ptr());
    // cv::threshold(output_image, output_image, 15, 255, cv::THRESH_TOZERO_INV);
    // cv::threshold(output_image, output_image, 14, 1, cv::THRESH_BINARY);
    // std::cout << "------" << output_image.type() << std::endl;
    return output_image;
}
}