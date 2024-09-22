#include "CSF_filter.h"
#include <pcl/filters/filter.h>
#include <pcl/visualization/cloud_viewer.h>

void applyCSFFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                    pcl::PointCloud<pcl::PointXYZ>& ground_out,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground_out,
                    const CSFParams& params) {

    CSF csf;
    csf.setPointCloud(*cloud_in);

    // 设置CSF参数
    csf.params.bSloopSmooth = params.bSloopSmooth;
    csf.params.cloth_resolution = params.cloth_resolution;
    csf.params.rigidness = params.rigidness;
    csf.params.time_step = params.time_step;
    csf.params.class_threshold = params.class_threshold;
    csf.params.iterations = params.iterations;

    // std::cout << "iterations: " << csf.params.iterations << std::endl;

    std::vector<int> groundIndexes, offGroundIndexes;
    csf.do_filtering(groundIndexes, offGroundIndexes);

    pcl::copyPointCloud(*cloud_in, groundIndexes, ground_out);
    pcl::copyPointCloud(*cloud_in, offGroundIndexes, *non_ground_out);
}
