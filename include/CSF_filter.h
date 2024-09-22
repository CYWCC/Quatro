#ifndef CSF_FILTER_H
#define CSF_FILTER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "CSF.h"

struct CSFParams {
    bool bSloopSmooth;
    double cloth_resolution;
    double rigidness;
    double time_step;
    double class_threshold;
    int iterations;
};

void applyCSFFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                    pcl::PointCloud<pcl::PointXYZ>& ground_out,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& non_ground_out,
                    const CSFParams& params);

#endif // CSF_FILTER_H
