#include <ros/ros.h>
#include <ros/package.h>
#include <actionlib/server/simple_action_server.h>
#include <object_template_alignment_server/PointcloudAlignmentAction.h>
#include <geometry_msgs/PoseStamped.h>

#include <string>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <float.h>
#include <vector>
#include <algorithm>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types_conversion.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <omp.h>
#include <pthread.h>


#include <pcl/visualization/cloud_viewer.h>

#include <pcl/features/normal_3d.h>

#include <pcl/point_types.h>

using namespace Eigen;
using namespace std;

typedef struct Cube {
    VectorXf r0;
    float half_edge_length;
    float lower_bound;
    float upper_bound;
    int depth;
} Cube;

typedef struct QueueElement {
    Cube *cube;
    struct QueueElement *next;
} QueueElement;

typedef struct PriorityQueue {
    QueueElement *head;
} PriorityQueue;

class PointcloudAlignmentAction
{
private:

protected:
    ros::NodeHandle nh_;
    actionlib::SimpleActionServer<object_template_alignment_server::PointcloudAlignmentAction> as_;
    std::string action_name_;
    object_template_alignment_server::PointcloudAlignmentFeedback feedback_;
    object_template_alignment_server::PointcloudAlignmentResult result_;

public:

    typedef pcl::PointCloud<pcl::Normal> SurfaceNormals;
     typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;

    float DISTANCE_THRESHOLD, MIN_OVERLAPPING_PERCENTAGE, TARGET_RADIUS_FACTOR, EVALUATION_THRESHOLD, MIN_PLANE_PORTION, MIN_PLANE_DISTANCE, MIN_SCALING_FACTOR, MAX_SCALING_FACTOR,
          MAX_TIME, ICP_EPS, ICP_EPS2, MAX_NUMERICAL_ERROR, MAX_PERCENTAGE, DAMPING_COEFFICIENT, DELAY_FACTOR, max_radius;
    int NUMBER_SUBCLOUDS, SIZE_SOURCE, SIZE_TARGET, REFINEMENT_ICP_SOURCE_SIZE, REFINEMENT_ICP_TARGET_SIZE, MAX_DEPTH, MAX_ICP_IT, REMOVE_PLANE, MAX_ICP_EVALUATIONS;

    pcl::KdTreeFLANN<pcl::PointXYZ> targetKdTree;
    pcl::KdTreeFLANN<pcl::PointXYZ> sourceKdTree;

    PointcloudAlignmentAction(std::string name) :
        as_(nh_, name, boost::bind(&PointcloudAlignmentAction::executeCB, this, _1), false),
        action_name_(name) {

        initializeParameters();

        as_.start();
    }

    ~PointcloudAlignmentAction(void) {}

        void executeCB(const object_template_alignment_server::PointcloudAlignmentGoalConstPtr &goal) {

        // preprocess pointcloud data
        MatrixXf source_pointcloud = preprocessSourcePointcloud(goal->source_pointcloud);
        MatrixXf target_pointcloud = preprocessTargetPointcloud(goal->target_pointcloud, goal->initial_pose);

        ROS_INFO("input data has been preprocessed");


        // convert initial_pose structure to transformation parameters
        MatrixXf R_icp = MatrixXf(3,3);
        float qx = goal->initial_pose.pose.orientation.x;
        float qy = goal->initial_pose.pose.orientation.y;
        float qz = goal->initial_pose.pose.orientation.z;
        float qw = goal->initial_pose.pose.orientation.w;
        R_icp <<
                 1.0f - 2.0f*qy*qy - 2.0f*qz*qz, 2.0f*qx*qy - 2.0f*qz*qw, 2.0f*qx*qz + 2.0f*qy*qw,
                 2.0f*qx*qy + 2.0f*qz*qw, 1.0f - 2.0f*qx*qx - 2.0f*qz*qz, 2.0f*qy*qz - 2.0f*qx*qw,
                 2.0f*qx*qz - 2.0f*qy*qw, 2.0f*qy*qz + 2.0f*qx*qw, 1.0f - 2.0f*qx*qx - 2.0f*qy*qy;

        VectorXf t_icp(3);
        t_icp << goal->initial_pose.pose.position.x, goal->initial_pose.pose.position.y, goal->initial_pose.pose.position.z;
        float s_icp = 1.;

        MatrixXf R_init = R_icp;
        VectorXf t_init = t_icp;
        float s_init = 1;


        // execute the pointcloud alignment algorithm
        find_pointcloud_alignment(goal->command, source_pointcloud, target_pointcloud, R_icp, t_icp, s_icp);


        // send result to client
        geometry_msgs::Quaternion orientation;
        geometry_msgs::Point position;

        position.x = t_icp(0);
        position.y = t_icp(1);
        position.z = t_icp(2);

        if (rotationIsValid(R_icp) == false || R_icp(0,0) + R_icp(1,1) + R_icp(2,2) == 0) {
            ROS_ERROR("Computed rotation matrix is not valid! Returning initial pose to client.");
            R_icp = R_init;
            t_icp = t_init;
            s_icp = s_init;
        }

        orientation.w = sqrt(1. + R_icp(0,0) + R_icp(1,1) + R_icp(2,2)) / 2.;
        orientation.x = (R_icp(2,1) - R_icp(1,2)) / (4.*orientation.w);
        orientation.y = (R_icp(0,2) - R_icp(2,0)) / (4.*orientation.w);
        orientation.z = (R_icp(1,0) - R_icp(0,1)) / (4.*orientation.w);

        geometry_msgs::PoseStamped result;

        result.pose.orientation = orientation;
        result.pose.position = position;

        result_.transformation_pose = result;

        ROS_INFO("%s: Succeeded", action_name_.c_str());
        as_.setSucceeded(result_);
    }

    // evaluates the command parameter and calls the local or global point cloud alignment algorithm
    float find_pointcloud_alignment(int command, MatrixXf &source_pointcloud, MatrixXf &target_pointcloud, MatrixXf &R_icp, VectorXf &t_icp, float &s_icp) {
        // set evaluation threshold

        if (source_pointcloud.cols() == 0 || target_pointcloud.cols() == 0) {
            ROS_ERROR("source or target pointcloud is zero");
            return FLT_MAX;
        }

        // execute local or global algorithm
        if (command == 0) { // execute local icp
            ROS_INFO("executing local icp");

            MatrixXf *source_subclouds = subsample_source_cloud(source_pointcloud, REFINEMENT_ICP_SOURCE_SIZE);
            MatrixXf target_subcloud = random_filter(target_pointcloud, REFINEMENT_ICP_TARGET_SIZE);
            createTargetKdTree(target_subcloud);
            float err = local_pointcloud_alignment(source_subclouds, target_subcloud , R_icp, t_icp, s_icp);
            float al_per = calc_overlapping_percentage(source_subclouds[0], target_subcloud, R_icp, t_icp, s_icp);
            float no_err = normalizedError(source_subclouds[0], target_subcloud , R_icp, t_icp, s_icp);
            cout<<"err: "<<no_err<<" al_per: "<<al_per<<endl;
            return  err;
        } else  if (command == 1) { // execute global pointcloud alignment
            ROS_INFO("executing global icp");
            return global_pointcloud_alignment(source_pointcloud, target_pointcloud, R_icp, t_icp, s_icp);
        } else { // invalid command
            ROS_ERROR("Received invalid command: %d", command);
            as_.setAborted();
        }
    }

    // the global search procedure
    float global_pointcloud_alignment(MatrixXf &source_pointcloud, MatrixXf &target_pointcloud, MatrixXf &R, VectorXf &t, float &s) {

        // set up parameters, kdTree, ...
        struct timeval start;
        gettimeofday(&start, NULL);

        int queueLength;
        Cube **Q = initPriorityQueue(queueLength);

        MatrixXf R_init = R;
        VectorXf t_init = t;
        float s_init = s;

        int itCt = 0;
        float cur_err = FLT_MAX;
        float cur_percentage = FLT_MIN;

        MatrixXf *source_subclouds = subsample_source_cloud(source_pointcloud, SIZE_SOURCE);
        createSourceKdTree(source_subclouds[0]);

        MatrixXf target_subcloud = random_filter(target_pointcloud, SIZE_TARGET);
        createTargetKdTree(target_subcloud);

        float best_quality = FLT_MAX;


        // process priority queue
        int i = 0;
        #pragma omp parallel for shared(cur_err, R, t, s, i, cur_percentage, itCt, best_quality)
        for (i = 0; i < queueLength; i++) {

            if (i > MAX_ICP_EVALUATIONS || (i > 0 && stopCriterionFulfilled(getPassedTime(start), cur_percentage))) {
                continue;
            }

            MatrixXf R_i = getAARot(Q[i]->r0) * R_init;
            VectorXf t_i = t_init;
            float s_i = s_init;

            local_pointcloud_alignment(source_subclouds, target_subcloud, R_i, t_i, s_i);

            float percentage = calc_overlapping_percentage(source_subclouds[0], target_subcloud, R_i, t_i, s_i);
            float ppe = per_point_error(source_subclouds[0], target_subcloud , R_i, t_i, s_i);

            float quality = eval_quality(percentage, ppe);

            if (quality < best_quality && rotationIsValid(R_i) && s_i > MIN_SCALING_FACTOR && s_i < MAX_SCALING_FACTOR) {
                cur_err = ppe;
                cur_percentage = percentage;

                best_quality = quality;

                R = R_i;
                t = t_i;
                s = s_i;

                sendFeedback(cur_percentage, cur_err);
            }

            itCt++;
        }

        // execute 3 more icp iterations with symmetric alignments for each axis of the current best parameters
        t_init = t;
        s_init = s;

        MatrixXf R_sym[3];
        R_sym[0] = R*getRotationMatrix(M_PI,0,0);
        R_sym[1] = R*getRotationMatrix(0,M_PI,0);
        R_sym[2] = R*getRotationMatrix(0,0,M_PI);

        #pragma omp parallel for shared(cur_err, R, t, s, i, cur_percentage, itCt)
        for (int i = 0; i < 3; i++) {
            MatrixXf R_i = R_sym[i];
            VectorXf t_i = t_init;
            float s_i = s_init;

            local_pointcloud_alignment(source_subclouds, target_subcloud, R_i, t_i, s_i);

            float percentage = calc_overlapping_percentage(source_subclouds[0], target_subcloud, R_i, t_i, s_i);
            float ppe = per_point_error(source_subclouds[0], target_subcloud , R_i, t_i, s_i);

            float quality = eval_quality(percentage, ppe);

            if (quality < best_quality && rotationIsValid(R_i) && s_i > MIN_SCALING_FACTOR && s_i < MAX_SCALING_FACTOR) {
                cur_err = ppe;
                cur_percentage = percentage;

                best_quality = quality;

                R = R_i;
                t = t_i;
                s = s_i;

                sendFeedback(cur_percentage, cur_err);
            }

            itCt++;
        }

        // execute final local icp iteration with more points for more accuracy
        source_subclouds = subsample_source_cloud(source_pointcloud, REFINEMENT_ICP_SOURCE_SIZE);
        target_subcloud = random_filter(target_pointcloud, REFINEMENT_ICP_TARGET_SIZE);
        createTargetKdTree(target_subcloud);

        MatrixXf R_i = R;
        VectorXf t_i = t;
        float s_i = s;
        local_pointcloud_alignment(source_subclouds, target_subcloud, R_i, t_i, s_i);

        if (rotationIsValid(R_i) && s_i > MIN_SCALING_FACTOR && s_i < MAX_SCALING_FACTOR) {
            cur_percentage = calc_overlapping_percentage(source_subclouds[0], target_subcloud, R_i, t_i, s_i);
            cur_err = per_point_error(source_subclouds[0], target_subcloud , R_i, t_i, s_i);
            R = R_i;
            t = t_i;
            s = s_i;

            sendFeedback(cur_percentage, cur_err);
        }

        ROS_INFO("Executed %d icp iterations, per-point error: %f, aligned percentage: %f.", itCt+1, cur_err, cur_percentage);
        return cur_err;
    }

    float eval_quality(float percentage, float ppe) {
        return (1.-percentage)*ppe;
    }

    // the modified ICP algorithm
    float local_pointcloud_alignment(MatrixXf *source_subclouds, MatrixXf const &target_pointcloud, MatrixXf &R, VectorXf &t, float &s) {

        // create variables and stuff..
        int source_size = source_subclouds[0].cols(); // all subclouds have the same size

        float err_old;
        int itCt = 0;
        int source_pos = 0; // denotes the currently used subsample of the source cloud

        MatrixXf correspondences(3, source_size);
        VectorXf distances(source_size);
        MatrixXf source_proj(3, source_size);
        MatrixXf source_cloud;
        MatrixXf source_trimmed, correspondences_trimmed;
        MatrixXf R_old(3,3);
        VectorXf t_old(3);
        float s_old;

        // start ICP iteration
        while(itCt < MAX_ICP_IT) {

            source_cloud = source_subclouds[source_pos % NUMBER_SUBCLOUDS]; // update subsample of the source cloud
            itCt++;

            R_old = R;
            t_old = t;
            s_old = s;

            apply_transformation(source_cloud, source_proj, R, t, s);

            // E-Step: assign all projected source points their correspondence
            if (find_correspondences(source_proj, target_pointcloud, correspondences, distances) == false) {
                return FLT_MAX;
            }

            // discard all source-correspondence-pairs which distance is too high
            if (trim_pointcloud(source_cloud, correspondences, distances, source_trimmed, correspondences_trimmed) == false) {
                return FLT_MAX;
            }

            // M-Step: minimize distance between source and correspondence points
            if (find_transformation(source_trimmed, correspondences_trimmed, R, t, s) == false) {
                return FLT_MAX;
            }

            // if the parameters did not change, use different subsample of the source pointcloud
            // if all subsamples have been tried without any effect, quit the loop
            if ((R-R_old).norm() + (t-t_old).norm() + abs(s-s_old) < ICP_EPS) {
                if (source_pos == 0) {
                    err_old = calc_error(source_cloud, target_pointcloud, R, t, s);
                } else if (source_pos % NUMBER_SUBCLOUDS == 0) {
                    if ((R-R_old).norm() + (t-t_old).norm() + abs(s-s_old) < ICP_EPS2) {
                        break;
                    } else {
                        err_old = calc_error(source_cloud, target_pointcloud, R, t, s);;
                    }
                }
                source_pos++;
            }
        }

        cout<<"ICP iterations: "<<itCt<<endl;
        return calc_error(source_subclouds[0], target_pointcloud, R, t, s);
    }

    // calculates the percentage of points which distance to their nearest neighbor is smaller than the evaluation threshold
    float calc_overlapping_percentage(MatrixXf const &source_cloud, MatrixXf const &target_cloud, MatrixXf const &R, MatrixXf const &t, float s) {
        return (((float) pointsLowerThanThreshold(source_cloud, target_cloud, R, t, s)) / ((float) source_cloud.cols()));
    }

    // calculates the number of points which distance to their nearest neighbor is smaller than the evaluation threshold
    int pointsLowerThanThreshold(MatrixXf const &source_cloud, MatrixXf const &target_pointcloud, MatrixXf const &R, VectorXf const& t, float s) {
        MatrixXf source_proj(source_cloud.rows(), source_cloud.cols());
        apply_transformation(source_cloud, source_proj, R, t, s);
        MatrixXf correspondences(source_cloud.rows(), source_cloud.cols());
        VectorXf distances(source_cloud.cols());
        int number = 0;

        if (find_correspondences(source_proj, target_pointcloud , correspondences, distances) == false) {
            return INT_MAX;
        }

        for (int i = 0; i < source_cloud.cols(); i++) {
            if (distances(i) < EVALUATION_THRESHOLD) {
                number++;
            }
        }

        return number;
    }

    // discards all source-correspondence pairs which distance is higher than the distance threshold
    bool trim_pointcloud(MatrixXf const &pointcloud, MatrixXf const &correspondences, VectorXf const &distances, MatrixXf &pointcloud_trimmed, MatrixXf &correspondences_trimmed) {

        int min_valid_points = (int) (MIN_OVERLAPPING_PERCENTAGE*((float) pointcloud.cols()));
        int number_inliers = 0;

        for (int i = 0; i < distances.rows(); i++) {
            if (distances(i) < DISTANCE_THRESHOLD) {
                number_inliers++;
            }
        }

        if (number_inliers < min_valid_points) {
            number_inliers = min_valid_points;
        }

        /*if (number_inliers == 0) {
            return false;
        }*/

        pointcloud_trimmed = MatrixXf(3,number_inliers);
        correspondences_trimmed = MatrixXf(3, number_inliers);

        VectorXf distances_sorted = distances;

        sort(distances_sorted);

        float threshold = distances_sorted(number_inliers-1);

        int pos = 0;
        for (int i = 0; i < correspondences.cols(); i++) {
            if (distances(i) <= threshold && pos < number_inliers) {
                pointcloud_trimmed(0,pos) = pointcloud(0, i);
                pointcloud_trimmed(1,pos) = pointcloud(1, i);
                pointcloud_trimmed(2,pos) = pointcloud(2, i);

                correspondences_trimmed(0,pos) = correspondences(0, i);
                correspondences_trimmed(1,pos) = correspondences(1, i);
                correspondences_trimmed(2,pos) = correspondences(2, i);

                pos++;
            }
        }

        return true;
    }

    // assigns all points from the source cloud their nearest neighbor from the target cloud
    bool find_correspondences(MatrixXf const &source_pointcloud, MatrixXf const &target_pointcloud , MatrixXf &correspondences, VectorXf &distances) {
        pcl::PointXYZ searchPoint;

        for (int i = 0; i < source_pointcloud.cols(); i++) {
            searchPoint.x = source_pointcloud(0,i);
            searchPoint.y = source_pointcloud(1,i);
            searchPoint.z = source_pointcloud(2,i);

            if ((isnormal(searchPoint.x) == 0 && searchPoint.x != 0) ||
                (isnormal(searchPoint.y) == 0 && searchPoint.y != 0) ||
                (isnormal(searchPoint.z) == 0 && searchPoint.z != 0)) {
                return false;
            }

            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);
            if (targetKdTree.nearestKSearch (searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                correspondences(0,i) = target_pointcloud(0,pointIdxNKNSearch[0]);
                correspondences(1,i) = target_pointcloud(1,pointIdxNKNSearch[0]);
                correspondences(2,i) = target_pointcloud(2,pointIdxNKNSearch[0]);

                distances(i) = sqrt(pointNKNSquaredDistance[0]);
            }
        }

        return true;
    }

    // minimizes the sum of squared distances between the source-correspondence pairs by optimizing the transformation parameters
    bool find_transformation(MatrixXf const &pointcloud, MatrixXf const &correspondences, MatrixXf &R, VectorXf &t, float &s) {
        VectorXf mean1 = pointcloud.array().rowwise().mean();
        VectorXf mean2 = correspondences.array().rowwise().mean();

        MatrixXf pointcloud_norm = pointcloud.array().colwise() - mean1.array();
        MatrixXf correspondences_norm = correspondences.array().colwise() - mean2.array();

        MatrixXf W(3,3);
        W(0,0) = (pointcloud_norm.block(0,0,1,pointcloud.cols()) * correspondences_norm.block(0,0,1,correspondences.cols()).transpose())(0);
        W(0,1) = (pointcloud_norm.block(0,0,1,pointcloud.cols()) * correspondences_norm.block(1,0,1,correspondences.cols()).transpose())(0);
        W(0,2) = (pointcloud_norm.block(0,0,1,pointcloud.cols()) * correspondences_norm.block(2,0,1,correspondences.cols()).transpose())(0);

        W(1,0) = (pointcloud_norm.block(1,0,1,pointcloud.cols()) * correspondences_norm.block(0,0,1,correspondences.cols()).transpose())(0);
        W(1,1) = (pointcloud_norm.block(1,0,1,pointcloud.cols()) * correspondences_norm.block(1,0,1,correspondences.cols()).transpose())(0);
        W(1,2) = (pointcloud_norm.block(1,0,1,pointcloud.cols()) * correspondences_norm.block(2,0,1,correspondences.cols()).transpose())(0);

        W(2,0) = (pointcloud_norm.block(2,0,1,pointcloud.cols()) * correspondences_norm.block(0,0,1,correspondences.cols()).transpose())(0);
        W(2,1) = (pointcloud_norm.block(2,0,1,pointcloud.cols()) * correspondences_norm.block(1,0,1,correspondences.cols()).transpose())(0);
        W(2,2) = (pointcloud_norm.block(2,0,1,pointcloud.cols()) * correspondences_norm.block(2,0,1,correspondences.cols()).transpose())(0);

        JacobiSVD<MatrixXf> svd(W, ComputeThinU | ComputeThinV);

        MatrixXf U = -svd.matrixU();
        MatrixXf V = -svd.matrixV();

        R = U*V.transpose();
        R = R.inverse();

        if (R.determinant() < 0) {
            MatrixXf V = svd.matrixV();
            V(0,2) = -V(0,2);
            V(1,2) = -V(1,2);
            V(2,2) = -V(2,2);
            R = U*V.transpose();
            R = R.inverse();
        }

        MatrixXf a = R*pointcloud_norm;
        MatrixXf b = correspondences_norm;

        MatrixXf tmp1 = a.cwiseProduct(b);
        MatrixXf tmp2 = a.cwiseProduct(a);

        s = (((float) tmp1.rows())*((float) tmp1.cols())*tmp1.norm()) / (((float) tmp2.rows())*((float) tmp2.cols())*tmp2.norm());

        s = 1.0f;

        t = mean2 - s*R*mean1;

        return parametersValid(R,t,s);
    }

    // applies the current transformation T(R,t,s) to the given pointcloud
    void apply_transformation(MatrixXf const &pointcloud, MatrixXf &pointcloud_proj, MatrixXf const &R, VectorXf const &t, float s) {
        pointcloud_proj = s*R*pointcloud;
        pointcloud_proj = pointcloud_proj.array().colwise() + t.array();
    }

    // calculates the sum-of-least-squares of all inliers and divides it by the number of inliers
    float per_point_error(MatrixXf const &source_pointcloud, MatrixXf const &target_pointcloud, MatrixXf const &R, VectorXf const &t, float s) {
        MatrixXf source_proj(3, source_pointcloud.cols());
        MatrixXf correspondences(3, source_pointcloud.cols());
        VectorXf distances(source_pointcloud.cols());

        apply_transformation(source_pointcloud, source_proj, R, t, s);
        if (find_correspondences(source_proj, target_pointcloud, correspondences, distances) == false) {
            return FLT_MAX;
        }

        MatrixXf diff = source_proj - correspondences;

        float err = 0;
        int n_p = 0;

        for (int i = 0; i < distances.rows(); i++) {
            if (distances(i) < EVALUATION_THRESHOLD) {
                err += diff(0,i)*diff(0,i) + diff(1,i)*diff(1,i) + diff(2,i)*diff(2,i);
                n_p++;
            }
        }

        if (n_p == 0) {
            return FLT_MAX;
        }

        return sqrt(err) / ((float) n_p);
    }


    // calculates the sum of least squares error between the source points and their nearest neighbors in the target cloud
    float calc_error(MatrixXf const &source_pointcloud, MatrixXf const &target_pointcloud, MatrixXf const &R, VectorXf const &t, float s) {

        MatrixXf source_proj(3, source_pointcloud.cols());
        MatrixXf correspondences(3, source_pointcloud.cols());
        VectorXf distances(source_pointcloud.cols());
        MatrixXf source_proj_trimmed, correspondences_trimmed;

        apply_transformation(source_pointcloud, source_proj, R, t, s);
        if (find_correspondences(source_proj, target_pointcloud, correspondences, distances) == false) {
            return FLT_MAX;
        }
        trim_pointcloud(source_proj, correspondences, distances, source_proj_trimmed, correspondences_trimmed);


        MatrixXf diff = source_proj_trimmed - correspondences_trimmed;

        float err = 0;
        for (int i = 0; i < diff.cols(); i++) {

            err += diff(0,i)*diff(0,i) + diff(1,i)*diff(1,i) + diff(2,i)*diff(2,i);
        }

        return sqrt(err);
    }

    // sorts the given vector in increasing order
    void sort(VectorXf &v) {
      std::sort(v.data(), v.data()+v.size());
    }

    // converts the Eigen matrix to a vector
    VectorXf matrixToVector(MatrixXf m) {
        m.transposeInPlace();
        VectorXf v(Map<VectorXf>(m.data(), m.cols()*m.rows()));
        return v;
    }

    // splits the given cube of the octree into eight subcubes
    Cube **splitCube(Cube *cube) {
        Cube **subcubes = new Cube*[8];
        VectorXf offset(3);
        float hel = cube->half_edge_length/2.;
        float signs[2] = {-1.,+1.};

        int position = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    offset << hel*signs[i],hel*signs[j],hel*signs[k];
                    subcubes[position++] = createCube(cube->r0 + offset, hel, cube->depth + 1);
                }
            }
        }

        return subcubes;
    }

    // returns the number of cubes in the priority queue with the given depth
    int depthNumber(PriorityQueue *Q, int depth) {
        int ct = 0;
        QueueElement *tmp = Q->head;
        while (tmp != NULL) {
            if (tmp->cube->depth == depth) {
                ct++;
            }
            tmp = tmp->next;
        }
        return ct;
    }

    // permutes the numbers between start and end
    int *getPermutation(int start, int end) {
        int length = end-start+1;

        vector<int> indices;
        for (int i = 0; i < length; i++) {
            indices.push_back(start+i);
        }
        random_shuffle(indices.begin(), indices.end());

        int *permutation = new int[length];
        for (int i = 0; i < length; i++) {
            permutation[i] = indices[i];
        }

        return permutation;
    }

    // initalizes a new priority queue with all cubes up to MAX_DEPTH and returns its length
    Cube **initPriorityQueue(int &queueLength) {
        PriorityQueue *Q = createQueue();

        VectorXf r0_init(3);
        r0_init<<0,0,0;
        Cube *C_init = createCube(r0_init, M_PI, 0);

        fillPriorityQueue(Q, C_init, 0, MAX_DEPTH);
        int nCubes = length(Q);
        Cube **priorityQueue = new Cube *[nCubes];
        int depth = 0;
        int startPos = 0;
        while (depth <= MAX_DEPTH) {
            int ct = depthNumber(Q, depth++);
            int *offset = getPermutation(0, ct-1);

            for (int i = 0; i < ct; i++) {
                Cube *tmp = extractFirstElement(Q);
                priorityQueue[startPos+offset[i]] = tmp;
            }
            startPos += ct;
        }

        queueLength = nCubes;

        return priorityQueue;
    }

    // fills given priority queue with all cubes of the current depth
    void fillPriorityQueue(PriorityQueue *Q, Cube *cube, int curDepth, int MAX_DEPTH) {
        float vecLength = sqrt(cube->r0[0]*cube->r0[0] + cube->r0[1]*cube->r0[1] + cube->r0[2]*cube->r0[2]);

        if (vecLength < M_PI) {
            insert(Q, cube);
        }

        if (curDepth < MAX_DEPTH) {
            Cube **subcubes = splitCube(cube);

            for (int i = 0; i < 8; i++) {
                fillPriorityQueue(Q, subcubes[i], curDepth + 1, MAX_DEPTH);
            }
        }
    }

    // allocate queue memory
    PriorityQueue *createQueue() {
        PriorityQueue *queue = new PriorityQueue;
        queue->head = NULL;
        return queue;
    }

    // insert a new cube into the priority quene at
    void insert(PriorityQueue *queue, Cube *cube) {
        if (queue == NULL) {
            perror("queue == NULL");
            return;
        }
        QueueElement *newElement = new QueueElement;
        newElement->cube = cube;

        if (queue->head == NULL) {
            queue->head = newElement;
            queue->head->next = NULL;
            return;

        } else {
            if (betterThan(cube, queue->head->cube) == true) {
                newElement->next = queue->head;
                queue->head = newElement;
                return;
            }
            QueueElement *tmp = queue->head;
            while (tmp->next != 0) {
                if (betterThan(cube, tmp->next->cube) == true) {
                    newElement->next = tmp->next;
                    tmp->next = newElement;
                    return;;
                }

                tmp = tmp->next;
            }
            tmp->next = newElement;
            newElement->next = NULL;
        }
    }

    // deletes the priority queue
    void deleteQueue(PriorityQueue *queue) {
        if (queue == NULL)
            return;
        if (queue->head == NULL)
            free(queue);
        QueueElement *temp = queue->head, *next = NULL;
        while (temp != NULL) {
            next = temp->next;
            delete(temp);
            temp = next;
        }
        delete(queue);
    }

    // returns the length of the priority queue
    int length(PriorityQueue *queue) {
        if (queue == NULL || queue->head == NULL)
            return 0;

        int counter = 0;
        QueueElement *temp = queue->head;
        while (temp != NULL) {
            counter++;
            temp = temp->next;
        }
        return counter;
    }

    // extracts the first element from the prioirty queue and removes it from the queue
    Cube *extractFirstElement(PriorityQueue *queue) {
        if (queue == NULL || queue->head == NULL)
            return NULL;

        QueueElement *element = queue->head;
        Cube *cube = element->cube;
        queue->head = queue->head->next;
        delete(element);

        return cube;
    }

    // returns the passed time since start
    float getPassedTime(struct timeval start) {
        struct timeval end;
        gettimeofday(&end, NULL);

        return (float) (((1.0/1000)*((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))/1000.);
    }

    // returns the Angle-Axis rotation matrix for the given vector
    MatrixXf getAARot(VectorXf &r) {
        MatrixXf R = MatrixXf::Identity(3,3);

        if (r.norm() == 0) {
            return R;
        }

        MatrixXf r_x(3,3);
        r_x << 0, -r(2), r(1),
            r(2), 0, -r(0),
            -r(1), r(0), 0;


        R += (r_x*sin(r.norm()))/(r.norm());
        R += (r_x*r_x*(1-cos(r.norm())))/(r.norm()*r.norm());

        return R;
    }

    // creates a new cube with given half_edge length, r0 and depth
    Cube* createCube(VectorXf r0, float half_edge_length, int depth) {
        Cube *C = new Cube;
        C->r0 = r0;
        C->half_edge_length = half_edge_length;
        C->depth = depth;

        return C;
    }

    // compares two cubes and states which of the two cubes has to be searched first (lower depth is better)
    bool betterThan(Cube *cube1, Cube *cube2) {
        if (cube1->depth < cube2->depth) {
            return true;
        } else if (cube1->depth == cube2->depth && cube1->lower_bound < cube2->lower_bound) {
            return true;
        } else {
            return false;
        }
    }

    // updates the kdTree with the given point cloud
    void createTargetKdTree(MatrixXf &pointcloud) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr target_pcl_pointcloud (new pcl::PointCloud<pcl::PointXYZ>);;

        // Fill in the cloud data
        target_pcl_pointcloud->width    = pointcloud.cols();
        target_pcl_pointcloud->height   = 1;
        target_pcl_pointcloud->is_dense = false;
        target_pcl_pointcloud->points.resize(target_pcl_pointcloud->width * target_pcl_pointcloud->height);

        for (int i = 0; i < pointcloud.cols(); i++) {
            target_pcl_pointcloud->points[i].x = pointcloud(0, i);
            target_pcl_pointcloud->points[i].y = pointcloud(1, i);
            target_pcl_pointcloud->points[i].z = pointcloud(2, i);
        }

        targetKdTree.setInputCloud(target_pcl_pointcloud);
    }

    void createSourceKdTree(MatrixXf &pointcloud) {

        pcl::PointCloud<pcl::PointXYZ>::Ptr source_pcl_pointcloud (new pcl::PointCloud<pcl::PointXYZ>);;

        // Fill in the cloud data
        source_pcl_pointcloud->width    = pointcloud.cols();
        source_pcl_pointcloud->height   = 1;
        source_pcl_pointcloud->is_dense = false;
        source_pcl_pointcloud->points.resize(source_pcl_pointcloud->width * source_pcl_pointcloud->height);

        for (int i = 0; i < pointcloud.cols(); i++) {
            source_pcl_pointcloud->points[i].x = pointcloud(0, i);
            source_pcl_pointcloud->points[i].y = pointcloud(1, i);
            source_pcl_pointcloud->points[i].z = pointcloud(2, i);
        }

        sourceKdTree.setInputCloud(source_pcl_pointcloud);
    }

    // returns a subcloud of the given point cloud with number_points in it by a random sampling procedure
    MatrixXf random_filter(MatrixXf &pointcloud, int number_points) {
        if (pointcloud.cols() <= number_points) {
            return pointcloud;
        }

        vector<int> indices;
        for (int i = 0; i < pointcloud.cols(); i++) {
            indices.push_back(i);
        }
        random_shuffle(indices.begin(), indices.end());

        MatrixXf filtered_pointcloud(pointcloud.rows(), number_points);

        for (int i = 0; i < number_points; i++) {
            filtered_pointcloud(0,i) = pointcloud(0, indices[i]);
            filtered_pointcloud(1,i) = pointcloud(1, indices[i]);
            filtered_pointcloud(2,i) = pointcloud(2, indices[i]);
        }

        return filtered_pointcloud;
    }

    // returns a Euler rotation matrix for the given parameters
    MatrixXf getRotationMatrix(float xRot, float yRot, float zRot) {
        MatrixXf R(3,3);

        R << cos(zRot)*cos(yRot), -sin(zRot)*cos(xRot)+cos(zRot)*sin(yRot)*sin(xRot), sin(zRot)*sin(xRot)+cos(zRot)*sin(yRot)*cos(xRot),
             sin(zRot)*cos(yRot),  cos(zRot)*cos(xRot)+sin(zRot)*sin(yRot)*sin(xRot),-cos(zRot)*sin(xRot)+sin(zRot)*sin(yRot)*cos(xRot),
             -sin(yRot),           cos(yRot)*sin(xRot),                               cos(yRot)*cos(xRot);

        return R;
    }

    // preprocesses the source pointcloud: convertes it to a Eigen matrix and calculates the maximum radius between points in the source cloud
    MatrixXf preprocessSourcePointcloud(sensor_msgs::PointCloud2 source_msg) {
        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pointcloud_source (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(source_msg , *pointcloud_source);

        MatrixXf source_pointcloud = MatrixXf(3,pointcloud_source->size());

        max_radius = FLT_MIN;

        for (int i = 0; i < pointcloud_source->size(); i++) {
            source_pointcloud(0,i) = pointcloud_source->at(i).x;
            source_pointcloud(1,i) = pointcloud_source->at(i).y;
            source_pointcloud(2,i) = pointcloud_source->at(i).z;

            float radius = sqrt(pointcloud_source->at(i).x*pointcloud_source->at(i).x +
                                pointcloud_source->at(i).y*pointcloud_source->at(i).y +
                                pointcloud_source->at(i).z*pointcloud_source->at(i).z);
            if (radius > max_radius) {
                max_radius = radius;
            }
        }

        return source_pointcloud;
    }

    // preprocesses the target pointcloud: convertes it to a Eigen matrix, discards all points that are too far away from the inital position (depending of the max_radius)
    // and removes the plane if chosen
    MatrixXf preprocessTargetPointcloud(sensor_msgs::PointCloud2 target_msg, geometry_msgs::PoseStamped initial_pose) {
        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pointcloud_target (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(target_msg, *pointcloud_target);

        MatrixXf target_pointcloud(3, pointcloud_target->size());

        for (int i = 0; i < pointcloud_target->size(); i++) {
            target_pointcloud(0,i) = pointcloud_target->at(i).x;
            target_pointcloud(1,i) = pointcloud_target->at(i).y;
            target_pointcloud(2,i) = pointcloud_target->at(i).z;
        }


        if (REMOVE_PLANE == 1) {
            ROS_INFO("removing plane");
            target_pointcloud = removePlane(target_pointcloud);
        }

        int target_size = 0;
        for (int i = 0; i < target_pointcloud.cols(); i++) {
            float distToCenter = sqrt(pow(target_pointcloud(0,i) - initial_pose.pose.position.x,2) +
                              pow(target_pointcloud(1,i) - initial_pose.pose.position.y,2) +
                              pow(target_pointcloud(2,i) - initial_pose.pose.position.z,2));


            if (distToCenter < TARGET_RADIUS_FACTOR*max_radius) {
                target_size++;
            }
        }

        if (target_size == target_pointcloud.cols()) {
            return target_pointcloud;
        }

        MatrixXf target_pointcloud_new = MatrixXf(3,target_size);

        int pos = 0;
        for (int i = 0; i < target_pointcloud.cols(); i++) {
            float dist = sqrt(pow(target_pointcloud(0,i) - initial_pose.pose.position.x,2) +
                              pow(target_pointcloud(1,i) - initial_pose.pose.position.y,2) +
                              pow(target_pointcloud(2,i) - initial_pose.pose.position.z,2));

            if (dist < TARGET_RADIUS_FACTOR*max_radius && pos < target_size) {
                target_pointcloud_new(0,pos) = target_pointcloud(0,i);
                target_pointcloud_new(1,pos) = target_pointcloud(1,i);
                target_pointcloud_new(2,pos) = target_pointcloud(2,i);

                pos++;
            }
        }

        return target_pointcloud_new;
    }

    // returns the chosen number of subsamples of the source cloud
    MatrixXf *subsample_source_cloud(MatrixXf source_pointcloud, float size_source) {
        MatrixXf *source_subclouds = new MatrixXf[NUMBER_SUBCLOUDS];

        for (int i = 0; i < NUMBER_SUBCLOUDS; i++) {
            source_subclouds[i] = random_filter(source_pointcloud, size_source);
        }
        return source_subclouds;
    }

    // removes the biggest plane from the given pointcloud
    MatrixXf removePlane(MatrixXf &pointcloud) {
          pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);

          pcl_pointcloud->width  = pointcloud.cols();
          pcl_pointcloud->height = 1;
          pcl_pointcloud->points.resize(pcl_pointcloud->width * pcl_pointcloud->height);


          for (int i = 0; i < pointcloud.cols(); i++) {
              pcl_pointcloud->points[i].x = pointcloud(0,i);
              pcl_pointcloud->points[i].y = pointcloud(1,i);
              pcl_pointcloud->points[i].z = pointcloud(2,i);
          }

          pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
          pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
          pcl::SACSegmentation<pcl::PointXYZ> seg;

          seg.setOptimizeCoefficients (true);

          seg.setModelType (pcl::SACMODEL_PLANE);
          seg.setMethodType (pcl::SAC_RANSAC);
          seg.setDistanceThreshold (MIN_PLANE_DISTANCE);

          seg.setInputCloud (pcl_pointcloud);
          seg.segment (*inliers, *coefficients);

          if (inliers->indices.size () == 0) {
            ROS_ERROR ("Could not estimate a planar model for the given dataset.");
            return pointcloud;
          }

          if (inliers->indices.size() < MIN_PLANE_PORTION*((float) pointcloud.cols())) {
              return pointcloud;
          }          

          MatrixXf new_pointcloud(3,pointcloud.cols()-inliers->indices.size());

          int pos = 0;
          for (int i = 0; i < pointcloud.cols(); i++) {
              float dist = abs(coefficients->values[0]*pointcloud(0,i) +
                               coefficients->values[1]*pointcloud(1,i) +
                               coefficients->values[2]*pointcloud(2,i) +
                               coefficients->values[3]);
              dist /= sqrt(coefficients->values[0]*coefficients->values[0] +
                           coefficients->values[1]*coefficients->values[1] +
                           coefficients->values[2]*coefficients->values[2]);

              if (dist > MIN_PLANE_DISTANCE && pos < new_pointcloud.cols()) {
                  new_pointcloud(0,pos) = pointcloud(0,i);
                  new_pointcloud(1,pos) = pointcloud(1,i);
                  new_pointcloud(2,pos) = pointcloud(2,i);

                  pos++;
              }
          }

          return new_pointcloud;
    }

    // sends feedback to the client
    void sendFeedback(float percentage, float err) {
        feedback_.aligned_percentage = percentage;
        feedback_.normalized_error = err;
        as_.publishFeedback(feedback_);
    }

    // calculates the normalized per-point error
    float normalizedError(MatrixXf source_pointcloud, MatrixXf target_pointcloud, MatrixXf R, VectorXf t, float s) {
        int n_points = pointsLowerThanThreshold(source_pointcloud, target_pointcloud,R,t,s);
        float err = calc_error(source_pointcloud, target_pointcloud, R, t, s);
        return err /= ((float) n_points);
    }

    // states if the given rotation matrix is a valid one
    bool rotationIsValid(MatrixXf R) {
        if (abs(R.determinant()-1) > MAX_NUMERICAL_ERROR || (R*R.transpose() - MatrixXf::Identity(3,3)).norm() > MAX_NUMERICAL_ERROR) {
            return false;
        }
        return true;
    }

    // validates all transformation parameters
    bool parametersValid(MatrixXf R, VectorXf t, float s) {
        for (int i = 0; i < R.rows(); i++) {
            for (int j = 0; j < R.cols(); j++) {
                if (isnormal(R(i,j)) == 0 && R(i,j) != 0) {
                    return false;
                }
            }
        }

        for (int i = 0; i < t.cols(); i++) {
            if (isnormal(t(i)) == 0 && t(i) != 0) {
                return false;
            }
        }

        if (isnormal(s) == 0 && s != 0) {
            return false;
        }

        return true;
    }

    void initializeParameters() {
        DISTANCE_THRESHOLD = getFloatParameter("distance_threshold");
        MIN_OVERLAPPING_PERCENTAGE = getFloatParameter("min_overlapping_percentage");
        TARGET_RADIUS_FACTOR = getFloatParameter("target_radius_factor");
        NUMBER_SUBCLOUDS = getIntegerParameter("number_subclouds");
        SIZE_SOURCE = getIntegerParameter("size_source");
        SIZE_TARGET = getIntegerParameter("size_target");
        REFINEMENT_ICP_SOURCE_SIZE = getIntegerParameter("refinement_icp_source_size");
        REFINEMENT_ICP_TARGET_SIZE = getIntegerParameter("refinement_icp_target_size");
        EVALUATION_THRESHOLD = getFloatParameter("evaluation_threshold");
        MIN_PLANE_PORTION = getFloatParameter("min_plane_portion");
        MIN_PLANE_DISTANCE = getFloatParameter("min_plane_distance");
        MIN_SCALING_FACTOR = getFloatParameter("min_scaling_factor");
        MAX_SCALING_FACTOR = getFloatParameter("max_scaling_factor");
        MAX_DEPTH = getIntegerParameter("max_depth");
        ICP_EPS = getFloatParameter("icp_eps");
        MAX_ICP_IT = getIntegerParameter("max_icp_it");
        ICP_EPS2 = getFloatParameter("icp_eps2");
        MAX_NUMERICAL_ERROR = getFloatParameter("max_numerical_error");
        MAX_PERCENTAGE = getFloatParameter("max_percentage");
        DAMPING_COEFFICIENT = getFloatParameter("damping_coefficient");
        DELAY_FACTOR = getFloatParameter("delay_factor");
        REMOVE_PLANE = getIntegerParameter("remove_plane");
        MAX_ICP_EVALUATIONS = getIntegerParameter("max_icp_evaluations");
    }

    float getFloatParameter(string parameter_name) {
        string key;
        if (nh_.searchParam(parameter_name, key) == true) {
          double val;
          ros::param::get(key, val);

          ROS_INFO("%s: %f", parameter_name.c_str(), val);

          return (float) val;
        } else {
            ROS_ERROR("parameter %s not found", parameter_name.c_str());
            return 0;
        }
    }

    int getIntegerParameter(string parameter_name) {

        string key;
        if (nh_.searchParam(parameter_name, key) == true) {
          int val;
          nh_.getParam(key, val);

          ROS_INFO("%s: %d", parameter_name.c_str(), val);

          return val;
        } else {
            ROS_ERROR("parameter %s not found", parameter_name.c_str());

            return 0;
        }
    }

    // states if the time-dependent stop-criterion has been fulfilled
    bool stopCriterionFulfilled(float passedTime, float overlapping_percentage) {
        float min_percentage = (float) exp(-DAMPING_COEFFICIENT*(passedTime - DELAY_FACTOR))*100.;

        if (min_percentage < overlapping_percentage) {
            return true;
        } else {
            return false;
        }
    }

    // ******************************************************************************************************
    // functions only necessary for debugging
    // ******************************************************************************************************

    void savePointcloud(MatrixXf pointcloud, string filename) {
        ofstream file;

        file.open(filename.c_str());
        if (!file.is_open()) {
            cout<<"Fehler beim oeffnen von "<<filename<<"!"<<endl;
        }

        for (int i = 0; i < pointcloud.cols(); i++) {
            file << pointcloud(0,i)<<" "<<pointcloud(1,i)<<" "<<pointcloud(2,i) << endl;
        }

        file.close();
    }

    void printDistances(MatrixXf source_cloud, MatrixXf target_cloud, MatrixXf R, VectorXf t, float s) {
        MatrixXf source_proj(source_cloud.rows(), source_cloud.cols());
        apply_transformation(source_cloud, source_proj, R, t , s);
        MatrixXf correspondences(source_cloud.rows(), source_cloud.cols());
        VectorXf distances(source_cloud.cols());

        if (find_correspondences(source_proj, target_cloud, correspondences, distances) == false) {
            return;
        }


        ofstream file;

        file.open("/home/sebastian/Desktop/distances.txt");
        if (!file.is_open()) {
            cout<<"Fehler beim oeffnen von distances.txt!"<<endl;
        }

        for (int i = 0; i < distances.rows(); i++) {
            file << distances(i) << endl;
        }

        file.close();

    }

    void visualizePointcloud(MatrixXf pointcloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_pcl (new pcl::PointCloud<pcl::PointXYZ>);

        // Generate pointcloud data
        pointcloud_pcl->width = pointcloud.cols();
        pointcloud_pcl->height = 1;
        pointcloud_pcl->points.resize (pointcloud_pcl->width * pointcloud_pcl->height);

        for (size_t i = 0; i < pointcloud_pcl->points.size (); ++i) {
            pointcloud_pcl->points[i].x = pointcloud(0,i);
            pointcloud_pcl->points[i].y = pointcloud(1,i);
            pointcloud_pcl->points[i].z = pointcloud(2,i);
        }

        pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
        viewer.showCloud (pointcloud_pcl);
        while (!viewer.wasStopped ()) {}
    }

    void visualizePointclouds(MatrixXf pointcloud1, MatrixXf pointcloud2) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud1_pcl (new pcl::PointCloud<pcl::PointXYZ>);

        for (int i = 0; i < pointcloud2.cols(); i++) {
            pointcloud2(0,i) = pointcloud2(0,i)+2;
            pointcloud2(1,i) = pointcloud2(1,i)+2;
            pointcloud2(2,i) = pointcloud2(2,i)+0.5;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_pcl (new pcl::PointCloud<pcl::PointXYZ>);

        // Generate pointcloud data
        pointcloud_pcl->width = pointcloud1.cols()+pointcloud2.cols();
        pointcloud_pcl->height = 1;
        pointcloud_pcl->points.resize (pointcloud_pcl->width * pointcloud_pcl->height);

        for (size_t i = 0; i < pointcloud1_pcl->points.size (); ++i) {
            pointcloud_pcl->points[i].x = pointcloud1(0,i);
            pointcloud_pcl->points[i].y = pointcloud1(1,i);
            pointcloud_pcl->points[i].z = pointcloud1(2,i);
        }

        for (size_t i = 0; i < pointcloud2.cols(); ++i) {
            pointcloud_pcl->points[i+pointcloud1.cols()].x = pointcloud2(0,i);
            pointcloud_pcl->points[i+pointcloud1.cols()].y = pointcloud2(1,i);
            pointcloud_pcl->points[i+pointcloud1.cols()].z = pointcloud2(2,i);
        }

        pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
        viewer.showCloud (pointcloud_pcl);
        while (!viewer.wasStopped ()) {}
    }

    void saveData(MatrixXf pc1, MatrixXf pc2, MatrixXf cp) {
        savePointcloud(pc1, "/home/sebastian/Desktop/pc1.txt");
        savePointcloud(pc2, "/home/sebastian/Desktop/pc2.txt");
        savePointcloud(cp, "/home/sebastian/Desktop/cp.txt");
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_alignment");

    PointcloudAlignmentAction pointcloud_alignment(ros::this_node::getName());
    ros::spin();

    return 0;
}
