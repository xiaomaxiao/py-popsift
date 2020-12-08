

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

#include <opencv2/core/core.hpp>

namespace py = pybind11;


// namespace pybind11 { namespace detail {
//     template <>
//     struct type_caster<cv::KeyPoint>{

//         PYBIND11_TYPE_CASTER(cv::KeyPoint,_("cv2.KeyPoint"));

//         //python to c++
//         bool load(handle obj, bool){

//         }
//     };

// }}

// class KeyPoint
// {
// public:
//     KeyPoint(std::tuple<float,float> _pt, float _size, float _angle=-1, float _response=0, \
//                 int _octave=0, int _class_id=-1)
//     {
//         pt = _pt;
//         size = _size;
//         angle = _angle;
//         response = _response;
//         octave = _octave;
//         class_id = _class_id;
//     }
    
// public:
//     float angle;
//     int class_id;
//     int octave;
//     std::tuple<float,float> pt;
//     float response;
//     float size;
// };

namespace pybind11 { namespace detail{
//! 实现 cv::Point 和 tuple(x,y) 之间的转换。
template<>
struct type_caster<cv::Point2f>{

    //! 定义 cv::Point 类型名为 tuple_xy, 并声明类型为 cv::Point 的局部变量 value。
    PYBIND11_TYPE_CASTER(cv::Point2f, _("tuple_xy"));

    //! 步骤1：从 Python 转换到 C++。    
    //! 将 Python tuple 对象转换为 C++ cv::Point 类型, 转换失败则返回 false。    
    //! 其中参数2表示是否隐式类型转换。   
    bool load(handle obj, bool){        
        // 确保传参是 tuple 类型        
        if(!py::isinstance<py::tuple>(obj)){            
            std::logic_error("Point(x,y) should be a tuple!");            
            return false;       
        }       
 
        // 从 handle 提取 tuple 对象，确保其长度是2。        
        py::tuple pt = reinterpret_borrow<py::tuple>(obj);        
        if(pt.size()!=2){            
            std::logic_error("Point(x,y) tuple should be size of 2");            
            return false;        
        }       

        //! 将长度为2的 tuple 转换为 cv::Point。        
        value = cv::Point2f(pt[0].cast<float>(), pt[1].cast<float>());       
        return true;    
    }

    //! 步骤2： 从 C++ 转换到 Python。    
    //! 将 C++ cv::Mat 对象转换到 tuple，参数2和参数3常忽略。    
    static handle cast(const cv::Point2f& pt, return_value_policy, handle){       
        return py::make_tuple(pt.x, pt.y).release();   
    }
};
}} //!  end namespace pybind11::detail


namespace pybind11 { namespace detail {
template <> struct type_caster<cv::KeyPoint>{
    PYBIND11_TYPE_CASTER(cv::KeyPoint,_("KeyPoint"));
    bool load(handle obj,bool){
        if(!obj) return false;
        value.pt = obj.attr("pt").cast<cv::Point2f>();
        value.size = obj.attr("size").cast<float>();
        value.angle = obj.attr("angle").cast<float>();
        value.response = obj.attr("response").cast<float>();
        value.octave = obj.attr("octave").cast<int>();
        value.class_id = obj.attr("size").cast<int>();
        return true;
    }

    static handle cast(cv::KeyPoint v, return_value_policy , handle){
        py::object obj = py::module::import("cv2").attr("KeyPoint")();
        obj.attr("pt") = py::cast(v.pt);
        obj.attr("size") = py::cast(v.size);
        obj.attr("angle") = py::cast(v.angle);
        obj.attr("response") = py::cast(v.response);
        obj.attr("octave") = py::cast(v.octave);
        obj.attr("size") = py::cast(v.size);
        return obj.release();
    }


};
}}




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
    std::tuple<std::vector<cv::KeyPoint>,py::array_t<float>> detectAndCompute(py::array_t<uint8_t,py::array::c_style> grayImg) {
        
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
        
        std::vector<cv::KeyPoint> keypoints;
        // std::vector<Desc> descriptors;
        py::array_t<float> descriptors = py::array_t<float>(num_descriptors * 128);
        py::buffer_info buffer_descriptors = descriptors.request();

        size_t descriptor_id = 0 ;
        for (size_t i = 0; i < num_features; i++){
            popsift::Feature feature = features[i];
            for (size_t ori = 0; ori <feature.num_ori;ori++){
                cv::KeyPoint kp(
                            feature.xpos,feature.ypos,
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
    // py::class_<cv::KeyPoint>(m,"cv.KeyPoint")
    //     .def(py::init<float,float,float,float,float,int,int>())
    //     .def_readonly("angle",&cv::KeyPoint::angle)
    //     .def_readonly("class_id",&cv::KeyPoint::class_id)
    //     .def_readonly("octave",&cv::KeyPoint::octave)
    //     .def_readonly("pt",&cv::KeyPoint::pt)
    //     .def_readonly("response",&cv::KeyPoint::response)
    //     .def_readonly("size",&cv::KeyPoint::size);
}
