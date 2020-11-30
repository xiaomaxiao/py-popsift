

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstring>

#include <popsift/common/device_prop.h>
#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/sift_config.h>
#include <popsift/version.hpp>
#include <cuda_runtime.h>

namespace py = pybind11;


class KeyPoint
{
public:
    KeyPoint(std::tuple<float,float> _pt, float _size, float _angle=-1, float _response=0, \
                int _octave=0, int _class_id=-1)
    {
        pt = _pt;
        size = _size;
        angle = _angle;
        response = _response;
        octave = _octave;
        class_id = _class_id;
    }
    
public:
    float angle;
    int class_id;
    int octave;
    std::tuple<float,float> pt;
    float response;
    float size;
};


class ZHSift 
{

public:
    /*init pipsift and set which gpu to use  
    */
    ZHSift(int gpu_id = 0 , bool print_dev_info = true){
        cudaDeviceReset();
        popsift::cuda::device_prop_t deviceInfo;
        deviceInfo.set( 0, print_dev_info );
        if( print_dev_info ) deviceInfo.print( );  
        
        popsift::Config config;
        //config.setMode(popsift::Config::SiftMode::OpenCV);
        m_PopSift = new PopSift(config,popsift::Config::ProcessingMode::ExtractingMode);

    }

    ~ZHSift(){
        m_PopSift->uninit();
        delete m_PopSift;
    }

    /*the method is the same as opencv sift 
     *     std::tuple<std::vector<KeyPoint> , std::vector<Desc> > 
    */
    std::tuple<std::vector<KeyPoint>,py::array_t<float>> detectAndCompute(py::array_t<uint8_t,py::array::c_style> grayImg) {
        
        //get image info from numpy 
        py::buffer_info buffer_grayImg = grayImg.request();
        if(buffer_grayImg.ndim !=2)
            throw std::runtime_error("number of dims must be equal to 2");
        size_t h = buffer_grayImg.shape[0];
        size_t w = buffer_grayImg.shape[1];

        //insert current image to sift job 
        SiftJob * job ; 
        job = m_PopSift->enqueue(w,h,static_cast<const unsigned char*>(buffer_grayImg.ptr));

        //construct keypoint and descriptor 
        popsift::FeaturesHost * feature_list = job->get();
        popsift::Feature * features = feature_list->getFeatures();
        size_t num_features = feature_list->getFeatureCount();
        size_t num_descriptors = feature_list->getDescriptorCount();
        
        std::vector<KeyPoint> keypoints;
        // std::vector<Desc> descriptors;
        py::array_t<float> descriptors = py::array_t<float>(num_descriptors * 128);
        py::buffer_info buffer_descriptors = descriptors.request();

        size_t descriptor_id = 0 ;
        for (size_t i = 0; i < num_features; i++){
            popsift::Feature feature = features[i];
            for (size_t ori = 0; ori <feature.num_ori;ori++){
                KeyPoint kp(
                            std::make_tuple(feature.xpos,feature.ypos),
                            feature.sigma,
                            feature.orientation[ori],
                            0.0,
                            feature.debug_octave,
                            -1);
                keypoints.push_back(kp);

                memcpy(static_cast<float *>(buffer_descriptors.ptr) + descriptor_id * 128,feature.desc[ori]->features,128 * sizeof(float));
                descriptor_id +=1;
                
            }

        }
        
        delete feature_list;
        delete job;
        
        descriptors.resize({num_descriptors,(size_t)128});
        return std::make_tuple(keypoints,descriptors);
    }

private:
    PopSift *m_PopSift;

};



PYBIND11_MODULE(ZHSift,m){
    py::class_<ZHSift>(m,"ZHSift")
        .def(py::init<int,bool>())
        .def("detectAndCompute",&ZHSift::detectAndCompute);
    py::class_<KeyPoint>(m,"KeyPoint")
        .def(py::init<std::tuple<float,float>,float,float,float,int,int>())
        .def_readonly("angle",&KeyPoint::angle)
        .def_readonly("class_id",&KeyPoint::class_id)
        .def_readonly("octave",&KeyPoint::octave)
        .def_readonly("pt",&KeyPoint::pt)
        .def_readonly("response",&KeyPoint::response)
        .def_readonly("size",&KeyPoint::size);
}
