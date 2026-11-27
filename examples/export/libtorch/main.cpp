// Small explicit loader, not a C++ port of HYSo Analyzer. No downloads.
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/version.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

std::vector<int64_t> shape(const std::string& text) {
    std::stringstream stream(text); std::string token; std::vector<int64_t> dims;
    int64_t count=1;
    while (std::getline(stream,token,',')) {
        size_t consumed=0;auto value=std::stoll(token,&consumed);
        if(consumed!=token.size()) throw std::runtime_error("invalid shape integer");
        if(value<1 || value>1048576 || count>1048576/value) throw std::runtime_error("shape budget exceeded");
        count*=value; dims.push_back(value);
    }
    if(dims.empty() || dims.size()>8) throw std::runtime_error("invalid rank");
    return dims;
}

torch::Tensor load(const char* path,const std::vector<int64_t>& dims) {
    int64_t count=1;for(auto d:dims)count*=d;
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=static_cast<std::streamoff>(count*sizeof(float))) throw std::runtime_error("raw float32 input byte/shape mismatch");
    file.seekg(0);auto tensor=torch::empty(dims,torch::kFloat32);
    if(!file.read(reinterpret_cast<char*>(tensor.data_ptr<float>()),count*sizeof(float))) throw std::runtime_error("input read failed");
    if(!torch::isfinite(tensor).all().item<bool>()) throw std::runtime_error("nonfinite input/reference");
    return tensor;
}

int main(int argc,char** argv) {
    try {
        if(argc!=6 && argc!=7) throw std::runtime_error("usage: checker model.pt input.f32 expected.f32 input_shape output_shape [--cuda]");
        const bool cuda=argc==7 && std::string(argv[6])=="--cuda";
        if(argc==7 && !cuda) throw std::runtime_error("unknown device option");
        if(cuda && !torch::cuda::is_available()) throw std::runtime_error("CUDA LibTorch runtime unavailable");
        c10::InferenceMode guard;
        auto input=load(argv[2],shape(argv[4]));auto expected=load(argv[3],shape(argv[5]));
        torch::Device device(cuda?torch::kCUDA:torch::kCPU);
        auto module=torch::jit::load(argv[1],device);module.eval();
        auto ivalue=module.forward({input.to(device)});
        if(!ivalue.isTensor()) throw std::runtime_error("this example profile requires a tensor output");
        auto native=ivalue.toTensor();auto output=native.to(torch::kCPU);
        if(output.sizes()!=expected.sizes() || output.scalar_type()!=torch::kFloat32) throw std::runtime_error("output shape/dtype mismatch");
        if(!torch::isfinite(output).all().item<bool>()) throw std::runtime_error("nonfinite output");
        auto error=(output.to(torch::kFloat64)-expected.to(torch::kFloat64)).abs();
        auto absolute=error.max().item<double>();
        auto relative=(error/expected.to(torch::kFloat64).abs().clamp_min(1e-30)).max().item<double>();
        bool matched=torch::allclose(output,expected,1e-4,1e-5,false);
        std::cout << std::setprecision(17) << "{\"runtime\":\"LibTorch_C++\",\"device\":\"" << (native.is_cuda()?"cuda:0":"cpu")
                  << "\",\"finite\":true,\"dtype\":\"float32\",\"elements\":" << output.numel()
                  << ",\"torch_version\":\"" << TORCH_VERSION << "\",\"max_absolute_error\":" << absolute
                  << ",\"max_relative_error\":" << relative << ",\"atol\":0.00001,\"rtol\":0.0001,\"matched\":" << (matched?"true":"false") << "}" << std::endl;
        return matched?0:2;
    } catch(const c10::Error& error) {std::cerr << error.what() << std::endl;return 3;}
      catch(const std::exception& error) {std::cerr << error.what() << std::endl;return 3;}
}
