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


#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

#include <sensor_msgs/PointCloud2.h>
#include <vigir_object_template_msgs/TemplateServerList.h>
#include <vigir_ocs_msgs/OCSObjectSelection.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types_conversion.h>

#include <pthread.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <float.h>

#include <vigir_object_template_msgs/SetAlignObjectTemplate.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <omp.h>


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


static const float DISTANCE_THRESHOLD = 0.02;
static const float MIN_OVERLAPPING_PERCENTAGE = 0.15;
static const int TARGET_RADIUS_FACTOR = 1.3;
static const int NUMBER_SUBCLOUDS = 5;
static const int SIZE_SOURCE = 250;
static const int SIZE_TARGET = 500;
static const int REFINEMENT_ICP_SOURCE_SIZE = 500;
static const int REFINEMENT_ICP_TARGET_SIZE = 2000;
static const float EVALUATION_THRESHOLD = 0.015;
static const int MIN_PLANE_PORTION = 0.2;
static const float MIN_PLANE_DISTANCE = 0.01;
static const float MIN_SCALING_FACTOR = 0.8;
static const float MAX_SCALING_FACTOR = 1.2;
static float MAX_TIME = 1.2;
int MAX_DEPTH = 2;

static pcl::KdTreeFLANN<pcl::PointXYZ> targetKdTree;

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
    PointcloudAlignmentAction(std::string name) :
    as_(nh_, name, boost::bind(&PointcloudAlignmentAction::executeCB, this, _1), false),
    action_name_(name) {
        as_.start();
    }

    ~PointcloudAlignmentAction(void) {}

    void executeCB(const object_template_alignment_server::PointcloudAlignmentGoalConstPtr &goal) {

        // preprocess pointcloud data
        float max_radius;
        MatrixXf source_pointcloud = preprocessSourcePointcloud(goal->source_pointcloud, max_radius);
        MatrixXf target_pointcloud = preprocessTargetPointcloud(goal->target_pointcloud, max_radius, goal->initial_pose);
        if (false) { // TODO
            savePointcloud(target_pointcloud, "/home/sebastian/Desktop/plane1.txt");
            target_pointcloud = removePlane(target_pointcloud);
            savePointcloud(target_pointcloud, "/home/sebastian/Desktop/plane2.txt");
        }



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


        // execute the pointcloud alignment algorithm
        find_pointcloud_alignment(goal->command, source_pointcloud, target_pointcloud, R_icp, t_icp, s_icp);


        // send goal
        geometry_msgs::Quaternion orientation;
        geometry_msgs::Point position;

        position.x = t_icp(0);
        position.y = t_icp(1);
        position.z = t_icp(2);

        if (R_icp(0,0) + R_icp(1,1) + R_icp(2,2) == 0) {
            ROS_ERROR("Received invalid Rotation matrix!");
            as_.setAborted();
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

    float find_pointcloud_alignment(int command, MatrixXf source_pointcloud, MatrixXf target_pointcloud, MatrixXf &R_icp, VectorXf &t_icp, float &s_icp) {

        if (source_pointcloud.cols() == 0 || target_pointcloud.cols() == 0) {
            return FLT_MAX;
        }

        // execute algorithm accoring to command
        if (command == 0) { // execute local icp
            cout<<"executing local icp"<<endl;

            MatrixXf *source_subclouds = subsample_source_cloud(source_pointcloud, REFINEMENT_ICP_SOURCE_SIZE);
            MatrixXf target_subcloud = random_filter(target_pointcloud, REFINEMENT_ICP_TARGET_SIZE);
            createKdTree(target_subcloud);

            return  local_pointcloud_alignment(source_subclouds, target_subcloud , R_icp, t_icp, s_icp);
        } else  if (command == 1) { // execute global pointcloud alignment
            cout<<"executing global icp"<<endl;
            return global_pointcloud_alignment(source_pointcloud, target_pointcloud, R_icp, t_icp, s_icp);

        } else { // invalid command
            ROS_ERROR("Received invalid command: %d", command);
            as_.setAborted();
        }
    }

    float global_pointcloud_alignment(MatrixXf source_pointcloud, MatrixXf target_pointcloud, MatrixXf &R, VectorXf &t, float &s) {

        struct timeval start;
        gettimeofday(&start, NULL);

        int queueLength;
        Cube **Q = createPriorityQueue(queueLength);

        MatrixXf R_init = R;
        VectorXf t_init = t;
        float s_init = s;

        int itCt = 0;
        float cur_err = FLT_MAX;
        float max_percentage = FLT_MIN;

        MatrixXf *source_subclouds = subsample_source_cloud(source_pointcloud, SIZE_SOURCE);
        MatrixXf target_subcloud = random_filter(target_pointcloud, SIZE_TARGET);
        createKdTree(target_subcloud);

        #pragma omp parallel for shared(cur_err, R, t, s)
        for (int i = 0; i < queueLength; i++) {

            if (i != 0 && getPassedTime(start) > MAX_TIME) {
                continue;
            }

            MatrixXf R_i = getAARot(Q[i]->r0) * R_init;
            VectorXf t_i = t_init;
            float s_i = s_init;

            local_pointcloud_alignment(source_subclouds, target_subcloud, R_i, t_i, s_i);

            float percentage = (((float) pointsLowerThanThreshold(source_pointcloud, target_subcloud, R_i, t_i, s_i)) / ((float) source_pointcloud.cols()))*100.;

            if (percentage > max_percentage && s_i > MIN_SCALING_FACTOR && s_i < MAX_SCALING_FACTOR) {
                cur_err = weightedError(source_pointcloud, target_subcloud, R_i, t_i, s_i);
                max_percentage = percentage;

                R = R_i;
                t = t_i;
                s = s_i;

                sendFeedback(max_percentage, cur_err);
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

        #pragma omp parallel for shared(cur_err, R, t, s)
        for (int i = 0; i < 3; i++) {
            MatrixXf R_i = R_sym[i];
            VectorXf t_i = t_init;
            float s_i = s_init;

            local_pointcloud_alignment(source_subclouds, target_subcloud, R_i, t_i, s_i);

            float percentage = (((float) pointsLowerThanThreshold(source_pointcloud, target_subcloud, R_i, t_i, s_i)) / ((float) source_pointcloud.cols()))*100.;

            if (percentage > max_percentage && s_i > MIN_SCALING_FACTOR && s_i < MAX_SCALING_FACTOR) {
                cur_err = weightedError(source_pointcloud, target_pointcloud , R_i, t_i, s_i);
                max_percentage = percentage;

                R = R_i;
                t = t_i;
                s = s_i;

                sendFeedback(max_percentage, cur_err);
            }

            itCt++;
        }

        // execute final local icp iteration with more points for more accuracy
        source_subclouds = subsample_source_cloud(source_pointcloud, REFINEMENT_ICP_SOURCE_SIZE);
        target_subcloud = random_filter(target_pointcloud, REFINEMENT_ICP_TARGET_SIZE); // TODO: andere variable verwenden
        createKdTree(target_subcloud);

        MatrixXf R_i = R;
        VectorXf t_i = t;
        float s_i = s;
        local_pointcloud_alignment(source_subclouds, target_subcloud, R_i, t_i, s_i);

        float percentage = (((float) pointsLowerThanThreshold(source_pointcloud, target_subcloud, R_i, t_i, s_i)) / ((float) source_pointcloud.cols()))*100.;

        if (percentage > max_percentage) {
            cur_err = weightedError(source_pointcloud, target_pointcloud , R_i, t_i, s_i);
            max_percentage = percentage;

            R = R_i;
            t = t_i;
            s = s_i;

            sendFeedback(max_percentage, cur_err);
        }

        cout<<"Executed "<<itCt+1<<" icp iterations, error: "<<cur_err<<endl;
        return cur_err;
    }

    float local_pointcloud_alignment(MatrixXf *source_subclouds, MatrixXf target_pointcloud, MatrixXf &R, VectorXf &t, float &s) {

        float eps = 1e-7;
        int maxIt = 300;
        float err_eps = 0.1;

        int source_size = source_subclouds[0].cols();

        float err_old;
        int itCt = 0;

        MatrixXf correspondences(3, source_size);
        VectorXf distances(source_size);
        MatrixXf source_proj(3, source_size);
        MatrixXf source_trimmed, correspondences_trimmed;
        MatrixXf R_old(3,3);
        VectorXf t_old(3);
        float s_old;

        int source_pos = 0;

        MatrixXf source_cloud = source_subclouds[source_pos];


        while(itCt < maxIt) {

            source_cloud = source_subclouds[source_pos % NUMBER_SUBCLOUDS];
            itCt++;

            R_old = R;
            t_old = t;
            s_old = s;

            apply_transformation(source_cloud, source_proj, R, t, s);

            find_correspondences(source_proj, target_pointcloud, correspondences, distances);

            if (trim_pointcloud(source_cloud, correspondences, distances, source_trimmed, correspondences_trimmed) == false) {
                return DBL_MAX;
            }

            find_transformation(source_trimmed, correspondences_trimmed, R, t, s);

            // if the parameters did not change, use different subsample of the source pointcloud
            // if all subsamples have been tried without any effect, quit the loop
            if ((R-R_old).norm() + (t-t_old).norm() + abs(s-s_old) < eps) {
                if (source_pos == 0) {
                    err_old = calc_error(source_cloud, target_pointcloud, R, t, s);
                } else if (source_pos % 5 == 0) {
                    float err = calc_error(source_cloud, target_pointcloud, R, t, s);
                    if (abs(err-err_old) < err_eps) {
                        break;
                    } else {
                        err_old = err;
                    }
                }
                source_pos++;

            }
        }

        return calc_error(source_subclouds[0], target_pointcloud, R, t, s);
    }

    int pointsLowerThanThreshold(MatrixXf source_cloud, MatrixXf target_pointcloud, MatrixXf R, VectorXf t, float s) {
        MatrixXf source_proj(source_cloud.rows(), source_cloud.cols());
        apply_transformation(source_cloud, source_proj, R, t, s);
        MatrixXf correspondences(source_cloud.rows(), source_cloud.cols());
        VectorXf distances(source_cloud.cols());
        int number = 0;

        find_correspondences(source_proj, target_pointcloud , correspondences, distances);

        for (int i = 0; i < source_cloud.cols(); i++) {
            if (distances(i) < EVALUATION_THRESHOLD) {
                number++;
            }
        }

        return number;
    }

    bool trim_pointcloud(MatrixXf pointcloud, MatrixXf correspondences, VectorXf distances, MatrixXf &pointcloud_trimmed, MatrixXf &correspondences_trimmed) {

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

        if (number_inliers == 0) {
            return false;
        }

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

    void find_correspondences(MatrixXf source_pointcloud, MatrixXf target_pointcloud , MatrixXf &correspondences, VectorXf &distances) {
        pcl::PointXYZ searchPoint;

        for (int i = 0; i < source_pointcloud.cols(); i++) {
            searchPoint.x = source_pointcloud(0,i);
            searchPoint.y = source_pointcloud(1,i);
            searchPoint.z = source_pointcloud(2,i);

            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);
            if (targetKdTree.nearestKSearch (searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                correspondences(0,i) = target_pointcloud(0,pointIdxNKNSearch[0]);
                correspondences(1,i) = target_pointcloud(1,pointIdxNKNSearch[0]);
                correspondences(2,i) = target_pointcloud(2,pointIdxNKNSearch[0]);

                distances(i) = sqrt(pointNKNSquaredDistance[0]);
            }
        }
    }

    void find_transformation(MatrixXf pointcloud, MatrixXf correspondences, MatrixXf &R, VectorXf &t, float &s) {
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

        s = 1.;

        t = mean2 - s*R*mean1;
    }

    void apply_transformation(MatrixXf pointcloud, MatrixXf &pointcloud_proj, MatrixXf R, VectorXf t, float s) {
        pointcloud_proj = s*R*pointcloud;
        pointcloud_proj = pointcloud_proj.array().colwise() + t.array();
    }

float calc_error(MatrixXf source_pointcloud, MatrixXf target_pointcloud, MatrixXf R, VectorXf t, float s) {

        MatrixXf source_proj(3, source_pointcloud.cols());
        MatrixXf correspondences(3, source_pointcloud.cols());
        VectorXf distances(source_pointcloud.cols());
        MatrixXf source_proj_trimmed, correspondences_trimmed;

        apply_transformation(source_pointcloud, source_proj, R, t, s);
        find_correspondences(source_proj, target_pointcloud, correspondences, distances);
        trim_pointcloud(source_proj, correspondences, distances, source_proj_trimmed, correspondences_trimmed);

        MatrixXf diff = source_proj_trimmed - correspondences_trimmed;

        float err = 0;
        for (int i = 0; i < diff.cols(); i++) {

            err += sqrt(diff(0,i)*diff(0,i) + diff(1,i)*diff(1,i) + diff(2,i)*diff(2,i));
        }

        return err;
    }

    void sort(VectorXf &v) {
      std::sort(v.data(), v.data()+v.size());
    }

    VectorXf matrixToVector(MatrixXf m) {
        m.transposeInPlace();
        VectorXf v(Map<VectorXf>(m.data(), m.cols()*m.rows()));
        return v;
    }

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

    Cube **createPriorityQueue(int &queueLength) {
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

    PriorityQueue *createQueue() {
        PriorityQueue *queue = new PriorityQueue;
        queue->head = NULL;
        return queue;
    }

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

    Cube *extractFirstElement(PriorityQueue *queue) {
        if (queue == NULL || queue->head == NULL)
            return NULL;

        QueueElement *element = queue->head;
        Cube *cube = element->cube;
        queue->head = queue->head->next;
        delete(element);

        return cube;
    }

    float getPassedTime(struct timeval start) {
        struct timeval end;
        gettimeofday(&end, NULL);

        return (float) (((1.0/1000)*((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))/1000.);
    }

    MatrixXf getAARot(VectorXf r) {
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

    Cube* createCube(VectorXf r0, float half_edge_length, int depth) {
        Cube *C = new Cube;
        C->r0 = r0;
        C->half_edge_length = half_edge_length;
        C->depth = depth;

        return C;
    }

    bool betterThan(Cube *cube1, Cube *cube2) {
        if (cube1->depth < cube2->depth) {
            return true;
        } else if (cube1->depth == cube2->depth && cube1->lower_bound < cube2->lower_bound) {
            return true;
        } else {
            return false;
        }
    }

    void createKdTree(MatrixXf pointcloud) {

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

    MatrixXf random_filter(MatrixXf pointcloud, int number_points) {
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

    MatrixXf getRotationMatrix(float xRot, float yRot, float zRot) {
        MatrixXf R(3,3);

        R << cos(zRot)*cos(yRot), -sin(zRot)*cos(xRot)+cos(zRot)*sin(yRot)*sin(xRot), sin(zRot)*sin(xRot)+cos(zRot)*sin(yRot)*cos(xRot),
             sin(zRot)*cos(yRot),  cos(zRot)*cos(xRot)+sin(zRot)*sin(yRot)*sin(xRot),-cos(zRot)*sin(xRot)+sin(zRot)*sin(yRot)*cos(xRot),
             -sin(yRot),           cos(yRot)*sin(xRot),                               cos(yRot)*cos(xRot);

        return R;
    }

    MatrixXf preprocessSourcePointcloud(sensor_msgs::PointCloud2 source_msg, float &max_radius) {
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

    MatrixXf preprocessTargetPointcloud(sensor_msgs::PointCloud2 target_msg, float max_radius, geometry_msgs::PoseStamped initial_pose) {
        boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pointcloud_target (new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(target_msg, *pointcloud_target);

        int target_size = 0;
        for (int i = 0; i < pointcloud_target->size(); i++) {
            float distToCenter = sqrt(pow(pointcloud_target->at(i).x - initial_pose.pose.position.x,2) +
                              pow(pointcloud_target->at(i).y - initial_pose.pose.position.y,2) +
                              pow(pointcloud_target->at(i).z - initial_pose.pose.position.z,2));


            if (distToCenter < TARGET_RADIUS_FACTOR*max_radius) {
                target_size++;
            }

        }


        MatrixXf target_pointcloud = MatrixXf(3,target_size);

        int pos = 0;
        for (int i = 0; i < pointcloud_target->size(); i++) {
            float dist = sqrt(pow(pointcloud_target->at(i).x - initial_pose.pose.position.x,2) +
                              pow(pointcloud_target->at(i).y - initial_pose.pose.position.y,2) +
                              pow(pointcloud_target->at(i).z - initial_pose.pose.position.z,2));


            if (dist < TARGET_RADIUS_FACTOR*max_radius && pos < target_size) {
                target_pointcloud(0,pos) = pointcloud_target->at(i).x;
                target_pointcloud(1,pos) = pointcloud_target->at(i).y;
                target_pointcloud(2,pos) = pointcloud_target->at(i).z;

                pos++;
            }
        }

        return target_pointcloud;
    }

    MatrixXf *subsample_source_cloud(MatrixXf source_pointcloud, float size_source) {
        MatrixXf *source_subclouds = new MatrixXf[NUMBER_SUBCLOUDS];

        for (int i = 0; i < NUMBER_SUBCLOUDS; i++) {
            source_subclouds[i] = random_filter(source_pointcloud, size_source);
        }

        return source_subclouds;
    }

    MatrixXf removePlane(MatrixXf pointcloud) {
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

          if (inliers->indices.size () == 0)
          {
            ROS_ERROR ("Could not estimate a planar model for the given dataset.");
            return pointcloud;
          }

          if (inliers->indices.size() < MIN_PLANE_PORTION*((float) pointcloud.cols())) {
              return pointcloud;
          }
          cout<<"removing plane"<<endl;

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

    void sendFeedback(float percentage, float err) {
        feedback_.aligned_percentage = percentage;
        feedback_.normalized_error = err;
        as_.publishFeedback(feedback_);
    }

    float weightedError(MatrixXf source_pointcloud, MatrixXf target_pointcloud, MatrixXf R, VectorXf t, float s) {
        int n_points = pointsLowerThanThreshold(source_pointcloud, target_pointcloud,R,t,s);
        float err = calc_error(source_pointcloud, target_pointcloud, R, t, s);
        return err /= ((float) n_points);
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

        find_correspondences(source_proj, target_cloud, correspondences, distances);

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

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_alignment");

    PointcloudAlignmentAction pointcloud_alignment(ros::this_node::getName());

    ros::spin();

    return 0;
}
