#include "HrgTofApi.h"
#include <iostream>
#include <stdio.h>
#include <helper.h>

//#define PCL_VISUALIZER /*** 打开该宏，需要配置pcl库，可将深度图转换为PCL点云 ***、
//#define OPENCV_VISUALIZER /*** 打开该宏，需要配置Opencv库，显示渲染后的深度图像及幅度图像 ***/

#ifdef OPENCV_VISUALIZER
#include <opencv/cv.hpp>
#endif

#ifdef PCL_VISUALIZER
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

#ifdef OPENCV_VISUALIZER
void onMouse_depth(int event, int x, int y, int flags, void *param)
{
    cv::Mat *im = reinterpret_cast<cv::Mat*>(param);
    std::cout << "depth(" << x << "," << y << ") :" << im->at<float>(cv::Point(x, y))<< "m" <<std::endl;
}

void onMouse_amplitude(int event, int x, int y, int flags, void *param)
{
    cv::Mat *im = reinterpret_cast<cv::Mat*>(param);
    std::cout << "amplitude(" << x << "," << y << ") :" << static_cast<unsigned int>(im->at<int16_t>(cv::Point(x, y))) << std::endl;
}
#endif

typedef std::array<AlgorithmOutput_F32, 10> AlgorithmOutputArray;

int main()
{
    Hrg_LogConfig(HRG_LOG_LEVEL_INFO);

    Hrg_Dev_Info dev;
    dev.type = Dev_Eth;
    dev.Info.eth.addr = "192.168.0.6";
    dev.Info.eth.port = 8567;
    dev.frameReady = NULL; //callback function

    Hrg_Dev_Handle handle;
    if(0 != Hrg_OpenDevice(&dev, &handle))
    {
        printf("open device failed!\n");
        return -1;
    }

    //Hrg_SetRangeMode(&handle, Mode_Range_S);

    //Hrg_SetDepthRange(&handle, 0, 5000);

    Hrg_StartStream(&handle);

    AlgorithmOutputArray output_data;
    int output_idx = 0;
    Hrg_Frame frame;
    uint8_t* depth_rgb = new uint8_t[IMAGE_HEIGHT*IMAGE_WIDTH*3];
    uint8_t* dst_ir = new uint8_t[IMAGE_HEIGHT*IMAGE_WIDTH];
    float* pcl = new float[IMAGE_HEIGHT*IMAGE_WIDTH*3];

#ifdef PCL_VISUALIZER
    /****** pcl viewer******/
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem (0.3, 0.3, 0.3, 0.3);
    viewer->initCameraParameters();
    float theta = M_PI; // The angle of rotation in radians
    transform (0,0) = cos (theta);
    transform (0,1) = -sin(theta);
    transform (1,0) = sin (theta);
    transform (1,1) = cos (theta);
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
#endif

    while(true)
    {
        if(0 == Hrg_GetFrame(&handle, &frame))
        {
            printf("frame index:%d\n", frame.index);
			/*** Get depth data and amplitude data from the a frame. ***/
            Hrg_GetDepthF32andAmplitudeData(&handle,
                                            &frame,
                                            output_data[output_idx].depth.get(),
                                            output_data[output_idx].amplitude.get());

            /* then you can process depth data and amplitude data */
#ifdef OPENCV_VISUALIZER
            cv::Mat tof_depth = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F, output_data[output_idx].depth.get()); //原始深度图
            cv::Mat tof_amplitude= cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_16S, output_data[output_idx].amplitude.get()); //原始幅度图

            /*** In order to obtain higher quality images, bilateral filtering is recommended. ***/
            cv::Mat img_tof_depth_filter;
            cv::bilateralFilter(tof_depth*1000, img_tof_depth_filter, 20, 40, 10);
            cv::Mat tof_depth_f =img_tof_depth_filter/1000;
            tof_depth_f.copyTo(tof_depth);
#endif
            /* decode one depth_f32 data to rgb */
            Hrg_DepthF32ToRGB(&handle, depth_rgb, IMAGE_HEIGHT*IMAGE_WIDTH*3, output_data[output_idx].depth.get(), IMAGE_HEIGHT*IMAGE_WIDTH, 0.300, 3.747);

            /* decode one amplitude data to gray */
            Hrg_AmplitudeToIR(&handle, dst_ir, IMAGE_HEIGHT*IMAGE_WIDTH, output_data[output_idx].amplitude.get(), IMAGE_HEIGHT*IMAGE_WIDTH, 1200);

#ifdef OPENCV_VISUALIZER
            cv::Mat tof_depth_RGB = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, depth_rgb); //渲染后的深度图
            cv::Mat tof_amplitude_IR = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, dst_ir);//渲染后的幅度图

            cv::namedWindow("depth", 0);
            cv::setMouseCallback("depth", onMouse_depth, reinterpret_cast<void *>(&tof_depth));
            cv::imshow("depth", tof_depth_RGB);
            cv::waitKey(1);

            cv::namedWindow("amplitude", 0);
            cv::setMouseCallback("amplitude", onMouse_amplitude, reinterpret_cast<void *>(&tof_amplitude));
            cv::imshow("amplitude", tof_amplitude_IR);
            cv::waitKey(1);
#endif
            /*** Get point cloud data from distance data. ***/
            Hrg_GetXYZDataF32_f(&handle, output_data[output_idx].depth.get(), pcl, IMAGE_HEIGHT*IMAGE_WIDTH);

#ifdef PCL_VISUALIZER
            viewer->removeAllPointClouds();
            point_cloud_ptr->clear();

            for(int i=0;i<IMAGE_HEIGHT;i++)
            {
                for(int j=0;j<IMAGE_WIDTH;j++)
                {
                    int index = i*IMAGE_WIDTH+j;
                    pcl::PointXYZ point;
                    point.x = pcl[index*3+0];
                    point.y = pcl[index*3+1];
                    point.z = pcl[index*3+2];

                    if(point.z > 0)
                    {
                        point_cloud_ptr->points.push_back(point);
                    }
                }
            }
            point_cloud_ptr->width = point_cloud_ptr->points.size();
            point_cloud_ptr->height = 1;

            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*point_cloud_ptr, *transformed_cloud, transform);
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(point_cloud_ptr, "z");//按照z字段进行渲染
            viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud, fildColor);//显示点云，其中fildColor为颜色显示
            viewer->spinOnce(1);
            boost::this_thread::sleep(boost::posix_time::microseconds(1));
#endif

            Hrg_FreeFrame(&handle, &frame);
            output_idx = output_idx < 9 ? output_idx + 1 : 0;
        }

    }

    Hrg_StopStream(&handle);
    Hrg_CloseDevice(&handle);

    delete[] depth_rgb;
    delete[] dst_ir;
    delete[] pcl;

    return 0;
}


