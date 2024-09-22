#include <locale>

// Our modules
#include "fpfh_manager.hpp"
#include "quatro.hpp"
#include "imageProjection.hpp"
#include "patchwork.hpp"
#include "CSF_filter.h"  // add CSF
#include <thread>
#include <future>

namespace fs = std::experimental::filesystem;

boost::shared_ptr<PatchWork<PointType> > patchwork;

pcl::PointCloud<PointType>::ConstPtr getCloud(std::string filename);

std::string lidarType;
std::string groundSegMode;
std::string neighborSelectionMode;

// Just for printing out the results
const char separator    = ' ';
const int nameWidth     = 22;
const int numWidth      = 8;
string dashedLine = std::string(nameWidth + numWidth * 2 + 7, '-');

using namespace std;

void setParams(
        double noise_bound_of_each_measurement, double square_of_the_ratio_btw_noise_and_noise_bound,
        double estimating_scale, int num_max_iter, double control_parameter_for_gnc,
        double rot_cost_thr, const string& reg_type_name, Quatro<PointType, PointType>::Params &params);

void applyCSFFilterAsync(const pcl::PointCloud<PointType>::Ptr& cloud_in,
                         pcl::PointCloud<PointType>& ground_out,
                         pcl::PointCloud<PointType>::Ptr& non_ground_out,
                         const CSFParams& params) {
    applyCSFFilter(cloud_in, ground_out, non_ground_out, params);
}

void processImageProjection(const std::string& lidarType, const std::string& neighborSelectionMode, const std::string& groundSegMode,
                            const pcl::PointCloud<PointType>::Ptr& rawCloud, pcl::PointCloud<PointType>& groundCloud, 
                            pcl::PointCloud<PointType>::Ptr& nonGroundCloud, pcl::PointCloud<PointType>::Ptr& validSegments, 
                            pcl::PointCloud<PointType>& invalidSegments) {
    ImageProjection IP(lidarType, neighborSelectionMode, groundSegMode);
    IP.segmentCloud(rawCloud);
    IP.getGround(groundCloud);
    IP.getValidSegments(*validSegments);
    IP.getOutliers(invalidSegments);
}

void processPatchwork(ros::NodeHandle& nh, const pcl::PointCloud<PointType>::Ptr& rawCloud, pcl::PointCloud<PointType>& groundCloud, 
                      pcl::PointCloud<PointType>::Ptr& nonGroundCloud, pcl::PointCloud<PointType>::Ptr& validSegments, 
                      pcl::PointCloud<PointType>& invalidSegments) {
    patchwork.reset(new PatchWork<PointType>(&nh));
    double t;
    patchwork->estimate_ground(*rawCloud, groundCloud, *nonGroundCloud, t);
    processImageProjection(lidarType, neighborSelectionMode, "Patchwork", nonGroundCloud, groundCloud, nonGroundCloud, validSegments, invalidSegments);
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "run_example");
    ros::NodeHandle nh;

    // data path
    string src_path, tgt_path;
    nh.param<std::string>("/src_path", src_path, "");
    nh.param<std::string>("/tgt_path", tgt_path, "");

    // Feature extraction parameters
    double voxel_size, normal_radius, fpfh_radius;
    nh.param<double>("/voxel_size", voxel_size, 0.3);
    nh.param<double>("/FPFH/normal_radius", normal_radius, 0.5);
    nh.param<double>("/FPFH/fpfh_radius", fpfh_radius, 0.75);

    // Quatro parameters
    bool   estimating_scale;
    int    num_max_iter;
    double noise_bound, noise_bound_coeff, gnc_factor, rot_cost_diff_thr;
    nh.param<bool>("/Quatro/estimating_scale", estimating_scale, false);
    nh.param<double>("/Quatro/noise_bound", noise_bound, 0.25);
    nh.param<double>("/Quatro/noise_bound_coeff", noise_bound_coeff, 0.99);
    nh.param<double>("/Quatro/rotation/gnc_factor", gnc_factor, 1.39);
    nh.param<double>("/Quatro/rotation/rot_cost_diff_thr", rot_cost_diff_thr, 0.0001);
    nh.param<int>("/Quatro/rotation/num_max_iter", num_max_iter, 50);

    // std::cout << "num_max_iter: " << num_max_iter << std::endl;

    nh.param<std::string>("/Lidar_type", lidarType, "Velodyne-64-HDE");
    nh.param<std::string>("/ground_segmentation_mode", groundSegMode, "Patchwork");
    nh.param<std::string>("/neigbor_mode", neighborSelectionMode, "4CrossNeighbor");

    // CSF parameters
    CSFParams csf_params;
    nh.param<bool>("/CSF/bSloopSmooth", csf_params.bSloopSmooth, false);
    nh.param<double>("/CSF/cloth_resolution", csf_params.cloth_resolution, 1.5);
    nh.param<double>("/CSF/rigidness", csf_params.rigidness, 3.0);
    nh.param<double>("/CSF/time_step", csf_params.time_step, 0.9);
    nh.param<double>("/CSF/class_threshold", csf_params.class_threshold, 0.5);
    nh.param<int>("/CSF/iterations", csf_params.iterations, 400);

    // std::cout << "iteration: " << csf_params.iterations << std::endl;


    ros::Publisher SrcPublisher   = nh.advertise<sensor_msgs::PointCloud2>("/source", 100);
    ros::Publisher TgtPublisher   = nh.advertise<sensor_msgs::PointCloud2>("/target", 100);
    ros::Publisher AlignPublisher = nh.advertise<sensor_msgs::PointCloud2>("/align", 100);

    ros::Publisher SrcMatchingPublisher = nh.advertise<sensor_msgs::PointCloud2>("/matched_src", 100);
    ros::Publisher TgtMatchingPublisher = nh.advertise<sensor_msgs::PointCloud2>("/matched_tgt", 100);

    ros::Publisher SrcMCPublisher = nh.advertise<sensor_msgs::PointCloud2>("/max_clique_src", 100);
    ros::Publisher TgtMCPublisher = nh.advertise<sensor_msgs::PointCloud2>("/max_clique_tgt", 100);

    ros::Publisher SrcGroundPublisher    = nh.advertise<sensor_msgs::PointCloud2>("/ground_src", 100);
    ros::Publisher SrcNongroundPublisher = nh.advertise<sensor_msgs::PointCloud2>("/nonground_src", 100);

    ros::Publisher TgtGroundPublisher    = nh.advertise<sensor_msgs::PointCloud2>("/ground_tgt", 100);
    ros::Publisher TgtNongroundPublisher = nh.advertise<sensor_msgs::PointCloud2>("/nonground_tgt", 100);

    ros::Publisher SrcTrueKpsPublisher = nh.advertise<sensor_msgs::PointCloud2>("/true_kps_src", 100);
    ros::Publisher SrcTransformedPublisher = nh.advertise<sensor_msgs::PointCloud2>("/Transformed_src", 100);

    ros::Publisher SrcValidSegPublisher = nh.advertise<sensor_msgs::PointCloud2>("/valid_seg_src", 100);
    ros::Publisher TgtValidSegPublisher = nh.advertise<sensor_msgs::PointCloud2>("/valid_seg_tgt", 100);
    ros::Publisher SrcInvalidSegPublisher = nh.advertise<sensor_msgs::PointCloud2>("/invalid_seg_src", 100);
    ros::Publisher TgtInvalidSegPublisher = nh.advertise<sensor_msgs::PointCloud2>("/invalid_seg_tgt", 100);

    ros::Publisher CorrespondencePublisher = nh.advertise<visualization_msgs::Marker>("/correspondence", 100);
    ros::Publisher CorrMCPublisher = nh.advertise<visualization_msgs::Marker>("/corrMC", 100);

    /***
     * Load toy dataset
     */
    pcl::PointCloud<PointType>::Ptr srcRaw(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tgtRaw(new pcl::PointCloud<PointType>);

    std::string quatro_path = ros::package::getPath("quatro");
    cout << "Quatro path: " << quatro_path << endl;
    *srcRaw = *getCloud(src_path);
    *tgtRaw = *getCloud(tgt_path);

    /***
    * STEP 1. Initialize Quatro
    * If you employ quatro in your own SLAM method,
    * then note that `quatro.reset(params)` is necessary after the optimization for the next global registration
    * Or, it causes munmap error :( (IMPORTANT)
    */
    Quatro<PointType, PointType>         quatro;
    Quatro<PointType, PointType>::Params params;

    setParams(noise_bound, noise_bound_coeff,
              estimating_scale, num_max_iter, gnc_factor, rot_cost_diff_thr, "Quatro", params);
    quatro.reset(params);

    static double tTotal, tSrc, tTgt;

    // Outputs of ground segmentation
    pcl::PointCloud<PointType> srcGround;
    pcl::PointCloud<PointType> tgtGround;
    pcl::PointCloud<PointType>::Ptr ptrSrcNonground(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr ptrTgtNonground(new pcl::PointCloud<PointType>);

    pcl::PointCloud<PointType> srcInvalidSegments;
    pcl::PointCloud<PointType> tgtInvalidSegments;
    pcl::PointCloud<PointType>::Ptr srcValidSegments(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tgtValidSegments(new pcl::PointCloud<PointType>);

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    if (groundSegMode == "LeGO-LOAM") {
        std::cout << "Ground Segmentation Mode: LeGO-LOAM" << std::endl;
        processImageProjection(lidarType, neighborSelectionMode, groundSegMode, srcRaw, srcGround, ptrSrcNonground, srcValidSegments, srcInvalidSegments);
        processImageProjection(lidarType, neighborSelectionMode, groundSegMode, tgtRaw, tgtGround, ptrTgtNonground, tgtValidSegments, tgtInvalidSegments);

    } else if (groundSegMode == "Patchwork") {
        std::cout << "Ground Segmentation Mode: Patchwork" << std::endl;
        processPatchwork(nh, srcRaw, srcGround, ptrSrcNonground, srcValidSegments, srcInvalidSegments);
        processPatchwork(nh, tgtRaw, tgtGround, ptrTgtNonground, tgtValidSegments, tgtInvalidSegments);

    } else if (groundSegMode == "CSF") {
        std::cout << "Ground Segmentation Mode: CSF" << std::endl;
        
        applyCSFFilterAsync(srcRaw, srcGround, ptrSrcNonground, csf_params);
        applyCSFFilterAsync(tgtRaw, tgtGround, ptrTgtNonground, csf_params);

        // // 异步调用CSF滤除操作
        // auto srcFuture = std::async(std::launch::async, applyCSFFilterAsync, srcRaw, std::ref(srcGround), std::ref(ptrSrcNonground), csf_params);
        // auto tgtFuture = std::async(std::launch::async, applyCSFFilterAsync, tgtRaw, std::ref(tgtGround), std::ref(ptrTgtNonground), csf_params);
        // // 等待异步任务完成
        // srcFuture.get();
        // tgtFuture.get();
    }

    std::chrono::system_clock::time_point after_ground = std::chrono::system_clock::now();

    /* --------------------------------------------------------------------- */
    /***
     * cout output of preprocessing
     */
    std::cout.imbue(std::locale(""));
    cout << dashedLine << endl;
    cout << left  << setw(nameWidth) << setfill(separator) << "# of raw cloud" << " | ";
    cout << right << setw(numWidth) << setfill(separator) << srcRaw->size() << " | ";
    cout << right << setw(numWidth) << setfill(separator) << tgtRaw->size() << endl;

    cout << left  << setw(nameWidth) << setfill(separator) << "# of ground" << " | ";
    cout << right << setw(numWidth) << setfill(separator) << srcGround.size() << " | ";
    cout << right << setw(numWidth) << setfill(separator) << tgtGround.size() << endl;

    cout << left  << setw(nameWidth) << setfill(separator) << "# of nonground" << " | ";
    cout << right << setw(numWidth) << setfill(separator) << ptrSrcNonground->points.size() << " | ";
    cout << right << setw(numWidth) << setfill(separator) << ptrTgtNonground->points.size() << endl;

    if (groundSegMode == "Patchwork" || groundSegMode == "LeGO-LOAM") {
        cout << dashedLine << endl;
        cout << left  << setw(nameWidth) << setfill(separator) << "# of valid segments" << " | ";
        cout << right << setw(numWidth) << setfill(separator) << srcValidSegments->points.size() << " | ";
        cout << right << setw(numWidth) << setfill(separator) << tgtValidSegments->points.size() << endl;

        cout << left  << setw(nameWidth) << setfill(separator) << "# of invalid segments" << " | ";
        cout << right << setw(numWidth) << setfill(separator) << srcInvalidSegments.size() << " | ";
        cout << right << setw(numWidth) << setfill(separator) << tgtInvalidSegments.size() << endl;
    }

    /* --------------------------------------------------------------------- */

    /***
     * STEP 4. Extract feature matching pairs
     * We employed FPFH and empirically found that w/o voxelization rather degrades the matching performance!
     * Note that size of the matched pair should be identical to each other, i.e. 3 X N.
     * This part can also be changed with other feature extraction & matching methods.
     */

    // voxelize
    pcl::PointCloud<PointType>::Ptr srcFeat(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tgtFeat(new pcl::PointCloud<PointType>);

    // if(groundSegMode == "Patchwork" || groundSegMode == "LeGO-LOAM") {
    //     voxelize(srcValidSegments, srcFeat, voxel_size);
    //     voxelize(tgtValidSegments, tgtFeat, voxel_size);
    // } else {
    //     voxelize(ptrSrcNonground, srcFeat, voxel_size);
    //     voxelize(ptrTgtNonground, tgtFeat, voxel_size);
    // }


    *srcFeat = *srcRaw;
    *tgtFeat = *tgtRaw;

    std::chrono::system_clock::time_point after_voxelize = std::chrono::system_clock::now();

    FPFHManager fpfhmanager(normal_radius, fpfh_radius);
    fpfhmanager.flushAllFeatures();
    fpfhmanager.setFeaturePair(srcFeat, tgtFeat);

    // Eigen type
//    Eigen::Matrix<double, 3, Eigen::Dynamic> src_matched = fpfhmanager.getSrcMatched();
//    Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_matched = fpfhmanager.getTgtMatched();

    // PCL type
    pcl::PointCloud<PointType>::Ptr srcMatched(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tgtMatched(new pcl::PointCloud<PointType>);
    *srcMatched = fpfhmanager.getSrcKps();
    *tgtMatched = fpfhmanager.getTgtKps();

    /* --------------------------------------------------------------------- */
    /***
     * cout output of Feature extraction & matching
     */
    cout << dashedLine << endl;
    cout << left  << setw(nameWidth) << setfill(separator) << "# after voxelization" << " | ";
    cout << right << setw(numWidth) << setfill(separator) << srcFeat->size() << " | ";
    cout << right << setw(numWidth) << setfill(separator) << tgtFeat->size() << endl;

    cout << left  << setw(nameWidth) << setfill(separator) << "# after matching" << " | ";
    cout << right << setw(numWidth) << setfill(separator) << srcMatched->size() << " | ";
    cout << right << setw(numWidth) << setfill(separator) << tgtMatched->size() << endl;
    cout << dashedLine << endl;
    /* --------------------------------------------------------------------- */

    /***
     * STEP 5. Calculate relative pose between source and target
     * Our method is PCL-friendly!!
     */
    std::chrono::system_clock::time_point before_optim = std::chrono::system_clock::now();
    quatro.setInputSource(srcMatched);
    quatro.setInputTarget(tgtMatched);
    Eigen::Matrix4d output;
    quatro.computeTransformation(output);

    std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
    std::chrono::duration<double> ground_sec = after_ground - start;
    std::chrono::duration<double> voxelize_sec = after_voxelize - after_ground;
    std::chrono::duration<double> phfh_sec = before_optim - after_voxelize;
    std::chrono::duration<double> optim_sec = std::chrono::system_clock::now() - before_optim;
    std::cout << setprecision(4) << "\033[1;32mTotal takes: " << sec.count() << " sec. ";
    std::cout << "(Ground segmentation: " << ground_sec.count() << " sec.) ";
    std::cout << "(Voxelization: " << voxelize_sec.count() << " sec.) ";
    std::cout << "(FPFH: " << phfh_sec.count() << " sec.) ";
    std::cout << "(Setting matching pairs: " << sec.count() - optim_sec.count() << " sec. + Quatro: " << optim_sec.count() << " sec.)\033[0m" << std::endl;

    /**
     * Below codes are for visualization
     */
    // matched feature src
    pcl::PointCloud<PointType> aligned;
    pcl::transformPointCloud(*srcRaw, aligned, output);

    // Good matched visualize
    pcl::PointCloud<PointType> transformed_srcMatched;
    pcl::transformPointCloud(*srcMatched, transformed_srcMatched, output);

    pcl::PointCloud<PointType> transformed_src;
    pcl::PointCloud<PointType> good_matched_pcl;
    pcl::transformPointCloud(*srcFeat, transformed_src, output);

    std::vector<std::pair<int, int>> corr = fpfhmanager.getCorrespondences();
    for (size_t                      i    = 0; i < corr.size(); ++i) {
        auto src_idx = std::get<0>(corr[i]);
        auto dst_idx = std::get<1>(corr[i]);

        double distance = sqrt(
                pow(double(transformed_src[src_idx].x) - double(tgtFeat->points[dst_idx].x), 2) +
                pow(double(transformed_src[src_idx].y) - double(tgtFeat->points[dst_idx].y), 2) +
                pow(double(transformed_src[src_idx].z) - double(tgtFeat->points[dst_idx].z), 2)
        );

        if (distance < voxel_size * 3.0)
        {
            pcl::PointXYZ point_xyz;
            point_xyz.x = double(transformed_src[src_idx].x);
            point_xyz.y = double(transformed_src[src_idx].y);
            point_xyz.z = double(transformed_src[src_idx].z);

            good_matched_pcl.push_back(point_xyz);
        }
    }

    pcl::PointCloud<PointType> srcMaxCliques;
    pcl::PointCloud<PointType> tgtMaxCliques;
    quatro.getMaxCliques(srcMaxCliques, tgtMaxCliques);

    visualization_msgs::Marker corrMarker, corrMCMarker;
    setCorrespondenceMarker(*srcMatched, *tgtMatched, corrMarker);
    setCorrespondenceMarker(srcMaxCliques, srcMaxCliques, corrMCMarker, 0.2, {0.0, 1.0, 0.0}, 5143);

    sensor_msgs::PointCloud2 SrcMsg   = cloud2msg(*srcRaw);
    sensor_msgs::PointCloud2 TgtMsg   = cloud2msg(*tgtRaw);
    sensor_msgs::PointCloud2 AlignMsg = cloud2msg(aligned);

    // Results of preprocessing
    sensor_msgs::PointCloud2 SrcGndMsg = cloud2msg(srcGround);
    sensor_msgs::PointCloud2 TgtGndMsg = cloud2msg(tgtGround);
    sensor_msgs::PointCloud2 SrcNGMsg = cloud2msg(*ptrSrcNonground);
    sensor_msgs::PointCloud2 TgtNGMsg = cloud2msg(*ptrTgtNonground);

    sensor_msgs::PointCloud2 SrcValidSegMsg = cloud2msg(*srcValidSegments);
    sensor_msgs::PointCloud2 TgtValidSegMsg = cloud2msg(*tgtValidSegments);
    sensor_msgs::PointCloud2 SrcInvalidSegMsg = cloud2msg(srcInvalidSegments);
    sensor_msgs::PointCloud2 TgtInvalidSegMsg = cloud2msg(tgtInvalidSegments);

    sensor_msgs::PointCloud2 SrcTrueKPsMsg = cloud2msg(good_matched_pcl);
    sensor_msgs::PointCloud2 SrcTfMsg      = cloud2msg(transformed_srcMatched);

    int cnt = 0;
    cout << "Done" << endl;

    ros::Rate loop_rate(10);
    while (ros::ok()) {
        /***
         * Source, target, and estimate
         */
        SrcPublisher.publish(SrcMsg);
        TgtPublisher.publish(TgtMsg);
        AlignPublisher.publish(AlignMsg);

        /***
         * Visualization of preprocessing and segmentations
         */
        SrcGroundPublisher.publish(SrcGndMsg);
        SrcNongroundPublisher.publish(SrcNGMsg);
        SrcValidSegPublisher.publish(SrcValidSegMsg);
        SrcInvalidSegPublisher.publish(SrcInvalidSegMsg);

        TgtGroundPublisher.publish(TgtGndMsg);
        TgtNongroundPublisher.publish(TgtNGMsg);
        TgtValidSegPublisher.publish(TgtValidSegMsg);
        TgtInvalidSegPublisher.publish(TgtInvalidSegMsg);

        SrcMatchingPublisher.publish(cloud2msg(*srcMatched));
        TgtMatchingPublisher.publish(cloud2msg(*tgtMatched));

        SrcMCPublisher.publish(cloud2msg(srcMaxCliques));
        TgtMCPublisher.publish(cloud2msg(tgtMaxCliques));

        SrcTrueKpsPublisher.publish(SrcTrueKPsMsg);
        SrcTransformedPublisher.publish(SrcTfMsg);

        CorrespondencePublisher.publish(corrMarker);
        CorrMCPublisher.publish(corrMCMarker);

        loop_rate.sleep();
    }
}

void setParams(
        double noise_bound_of_each_measurement, double square_of_the_ratio_btw_noise_and_noise_bound,
        double estimating_scale, int num_max_iter, double control_parameter_for_gnc,
        double rot_cost_thr, const string& reg_type_name, Quatro<PointType, PointType>::Params &params) {
    //Quatro::Params

    params.noise_bound                   = noise_bound_of_each_measurement;
    params.cbar2                         = square_of_the_ratio_btw_noise_and_noise_bound;
    params.estimate_scaling              = estimating_scale;
    params.rotation_max_iterations       = num_max_iter;
    params.rotation_gnc_factor           = control_parameter_for_gnc;
    params.rotation_estimation_algorithm = Quatro<PointType, PointType>::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold       = rot_cost_thr;

    params.reg_name                  = reg_type_name;
    if (reg_type_name == "Quatro") {
        params.inlier_selection_mode = Quatro<PointType, PointType>::INLIER_SELECTION_MODE::PMC_HEU;
    } else { params.inlier_selection_mode = Quatro<PointType, PointType>::INLIER_SELECTION_MODE::PMC_EXACT; }
}

pcl::PointCloud<PointType>::ConstPtr getCloud(std::string filename) {
    FILE *file                    = fopen(filename.c_str(), "rb");
    if (!file) {
        std::cerr << "error: failed to load " << filename << std::endl;
        return nullptr;
    }

    std::vector<float> buffer(1000000);
    size_t             num_points =
                               fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 3;
    fclose(file);

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    cloud->resize(num_points);

    for (int i = 0; i < num_points; i++) {
        auto &pt = cloud->at(i);
        pt.x = buffer[i * 3];
        pt.y = buffer[i * 3 + 1];
        pt.z = buffer[i * 3 + 2];
        // Intensity is not in use
//         pt.intensity = buffer[i * 4 + 3];
    }

    return cloud;
}
