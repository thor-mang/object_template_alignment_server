#include <ros/ros.h>
#include <ros/package.h>
#include <actionlib/server/simple_action_server.h>
#include <object_template_alignment_server/GoIcpAction.h>
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


//#include <pcl/visualization/cloud_viewer.h>


using namespace Eigen;
using namespace std;

typedef struct Cube {
    VectorXf r0, t0;
    float half_edge_length;
    float lower_bound;
    float upper_bound;
} Cube;

typedef struct QueueElement {
    Cube *cube;
    struct QueueElement *next;
} QueueElement;

typedef struct PriorityQueue {
    QueueElement *head;
} PriorityQueue;

static float final_percentage = 0, final_error = FLT_MAX;

class GoIcpAction
{
private:

protected:
    ros::NodeHandle nh_;
    actionlib::SimpleActionServer<object_template_alignment_server::GoIcpAction> as_;
    std::string action_name_;
    object_template_alignment_server::GoIcpFeedback feedback_;
    object_template_alignment_server::GoIcpResult result_;

public:

    float DISTANCE_THRESHOLD, MIN_OVERLAPPING_PERCENTAGE, TARGET_RADIUS_FACTOR, EVALUATION_THRESHOLD, MIN_PLANE_PORTION, MIN_PLANE_DISTANCE, MIN_SCALING_FACTOR, MAX_SCALING_FACTOR,
          MAX_TIME, ICP_EPS, ICP_EPS2, MAX_NUMERICAL_ERROR, MAX_PERCENTAGE, DAMPING_COEFFICIENT, DELAY_FACTOR;
    int NUMBER_SUBCLOUDS, SIZE_SOURCE, SIZE_TARGET, REFINEMENT_ICP_SOURCE_SIZE, REFINEMENT_ICP_TARGET_SIZE, MAX_DEPTH, MAX_ICP_IT;

    pcl::KdTreeFLANN<pcl::PointXYZ> targetKdTree;

    GoIcpAction(std::string name) :
    as_(nh_, name, boost::bind(&GoIcpAction::executeCB, this, _1), false),
    action_name_(name) {

        initializeParameters();

        as_.start();
    }

    ~GoIcpAction(void) {}

        void executeCB(const object_template_alignment_server::GoIcpGoalConstPtr &goal) {
        cout<<"execute Callback"<<endl;

        // preprocess pointcloud data
        float max_radius;
        MatrixXf source_pointcloud = preprocessSourcePointcloud(goal->source_pointcloud, max_radius);
        MatrixXf target_pointcloud = preprocessTargetPointcloud(goal->target_pointcloud, max_radius, goal->initial_pose);

        //visualizePointcloud(target_pointcloud);

        cout<<"input data has been preprocessed"<<endl;


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

        MatrixXf R_init = R_icp;
        VectorXf t_init = t_icp;

        cout<<"calling find pointcloud alignment"<<endl;

        // execute the pointcloud alignment algorithm
        //find_pointcloud_alignment(goal->command, source_pointcloud, target_pointcloud, R_icp, t_icp);

        source_pointcloud = random_filter(source_pointcloud, SIZE_SOURCE);
        target_pointcloud = random_filter(target_pointcloud, SIZE_TARGET);
        createKdTree(target_pointcloud);

        float E_star = go_icp(source_pointcloud, target_pointcloud, goal->eps, R_icp, t_icp, goal->overlapping_percentage);

        // send goal
        geometry_msgs::Quaternion orientation;
        geometry_msgs::Point position;

        position.x = t_icp(0);
        position.y = t_icp(1);
        position.z = t_icp(2);

        if (rotationIsValid(R_icp) == false || R_icp(0,0) + R_icp(1,1) + R_icp(2,2) == 0) {
            ROS_ERROR("Received invalid Rotation matrix! Returning initial pose.");
            R_icp = R_init;
            t_icp = t_init;
        }

        orientation.w = sqrt(1. + R_icp(0,0) + R_icp(1,1) + R_icp(2,2)) / 2.;
        orientation.x = (R_icp(2,1) - R_icp(1,2)) / (4.*orientation.w);
        orientation.y = (R_icp(0,2) - R_icp(2,0)) / (4.*orientation.w);
        orientation.z = (R_icp(1,0) - R_icp(0,1)) / (4.*orientation.w);

        geometry_msgs::PoseStamped result;

        result.pose.orientation = orientation;
        result.pose.position = position;

        //result_.aligned_percentage = final_percentage;
        //result_.normalized_error = final_error;

        result_.transformation_pose = result;

        ROS_INFO("%s: Succeeded", action_name_.c_str());
        as_.setSucceeded(result_);
    }

    float go_icp(MatrixXf &pc1, MatrixXf &pc2, float eps, MatrixXf &R_star, VectorXf &t_star, float overlapping_portion) {
        cout<<"go_icp"<<endl;
        Cube *initial_rotation_cube = createRotationCube(VectorXf(3), M_PI);
        Cube *initial_translation_cube = createTranslationCube(VectorXf(3), 3);

        return rotationBnB(pc1, pc2, eps, initial_rotation_cube, initial_translation_cube, R_star, t_star, overlapping_portion);
    }

    float rotationBnB(MatrixXf &pc1, MatrixXf &pc2, float eps, Cube *initial_rotation_cube, Cube *initial_translation_cube, MatrixXf &R_star, VectorXf &t_star, float overlapping_portion) {
        cout<<"rotationBnB"<<endl;
        PriorityQueue *Q = createQueue();
        insert(Q, initial_rotation_cube);
        float E_star = FLT_MAX;

        while (length(Q) > 0) {
            Cube *C_cur = extractFirstElement(Q);
            if (E_star - C_cur->lower_bound < eps) {
                break;
            }
            Cube **subcubes = splitRotationCube(C_cur);
            for (int i = 0; i < 8; i++) {
                VectorXf t_star_i, gamma_r(pc1.cols());

                subcubes[i]->upper_bound = translationBnB(pc1, pc2, eps, initial_translation_cube, subcubes[i]->r0, gamma_r, E_star, t_star_i);
                if (subcubes[i]->upper_bound < E_star) {
                    MatrixXf R_icp = getAARot(subcubes[i]->r0);
                    VectorXf t_icp = t_star_i;
                    E_star = icp(pc1, pc2, R_icp, t_icp, overlapping_portion);
                    R_star = R_icp;
                    t_star = t_icp;
                }
                gamma_r = calc_gamma_r(pc1, subcubes[i]->half_edge_length);
                subcubes[i]->lower_bound = translationBnB(pc1, pc2, eps, initial_translation_cube, subcubes[i]->r0, gamma_r, E_star, t_star_i);
                if (subcubes[i]->lower_bound >= E_star) {
                    continue;
                }
                insert(Q, subcubes[i]);
            }
        }

        cout<<"rotation BnB end"<<endl;

        return E_star;
    }

    float translationBnB(MatrixXf &pc1, MatrixXf &pc2, float eps, Cube *initial_cube, VectorXf r, VectorXf gamma_r, float &E_star, VectorXf &t_star) {
        cout<<"translation BnB"<<endl;
        PriorityQueue *Q = createQueue();
        insert(Q, initial_cube);
        float E_t_star = E_star;

        cout<<"eps: "<<eps<<endl;
        while (length(Q) > 0) {
            cout<<"iteration"<<endl;
            Cube *C_cur = extractFirstElement(Q);
            if (E_t_star - C_cur->lower_bound < eps) {
                break;
            }
            Cube **subcubes = splitTranslationCube(C_cur);
            for (int i = 0; i < 8; i++) {
                cout<<"i: "<<i<<endl;
                subcubes[i]->upper_bound = calc_upper_bound_translation(pc1, pc2, getAARot(r), subcubes[i]->t0, subcubes[i]->half_edge_length, gamma_r);
                if (subcubes[i]->upper_bound < E_t_star) {
                    E_t_star = subcubes[i]->upper_bound;
                    t_star = subcubes[i]->t0;
                }
                subcubes[i]->lower_bound = calc_lower_bound_translation(pc1, pc2, getAARot(r), subcubes[i]->t0, subcubes[i]->half_edge_length, gamma_r);
                if (subcubes[i]->lower_bound >= E_t_star) {
                    continue;
                }
                insert(Q, subcubes[i]);
            }
        }

        cout<<"translation BnB end"<<endl;

        return E_t_star;
    }

    float calc_upper_bound_translation(MatrixXf &pc1, MatrixXf &pc2, MatrixXf const &R, VectorXf const &t, float sigma_t, VectorXf gamma_r) {
        float lower_bound = 0;

        float gamma_t = sqrt(3)*sigma_t;

        MatrixXf pc1_proj(pc1.rows(), pc1.cols()), cp(pc1.rows(), pc1.cols());
        VectorXf distances(pc1.cols());

        apply_transformation(pc1, pc1_proj, R, t);
        find_correspondences(pc1_proj, pc2, cp, distances);

        MatrixXf diff = pc1_proj-cp;
        float e_i;
        for (int i = 0; i < diff.cols(); i++) {
            e_i = sqrt(diff(0,i)*diff(0,i) + diff(1,i)*diff(1,i) + diff(2,i)*diff(2,i));
            e_i -= gamma_r(i);
            if (e_i > 0) {
                lower_bound += e_i*e_i;
            }
        }

        return lower_bound;
    }

    float calc_lower_bound_translation(MatrixXf &pc1, MatrixXf &pc2, MatrixXf const &R, VectorXf const  &t, float sigma_t, VectorXf gamma_r) {
        float lower_bound = 0;

        float gamma_t = sqrt(3)*sigma_t;

        MatrixXf pc1_proj(pc1.rows(), pc1.cols()), cp(pc1.rows(), pc1.cols());
        VectorXf distances(pc1.cols());

        apply_transformation(pc1, pc1_proj, R, t);
        find_correspondences(pc1_proj, pc2, cp, distances);

        MatrixXf diff = pc1_proj-cp;
        float e_i;
        for (int i = 0; i < diff.cols(); i++) {
            e_i = sqrt(diff(0,i)*diff(0,i) + diff(1,i)*diff(1,i) + diff(2,i)*diff(2,i));
            e_i -= (gamma_r(i)+gamma_t);
            if (e_i > 0) {
                lower_bound += e_i*e_i;
            }
        }

        return lower_bound;
    }

    VectorXf calc_gamma_r(MatrixXf &pc1, float sigma_r) {// TODO: nur einmal berechnen
        pc1 = pc1.cwiseProduct(pc1);
        pc1 = pc1.row(0) + pc1.row(1) + pc1.row(2);
        pc1 = pc1.array().sqrt();

        VectorXf pc1_norm = matrixToVector(pc1);

        return 2.*sin(min(((sqrt(3.)*sigma_r)/2.),M_PI/2.)) * pc1_norm;
    }




    float icp(MatrixXf &pc1, MatrixXf &pc2, MatrixXf &R_icp, VectorXf &t_icp, float overlapping_portion) {

        int itCt = 0;

        MatrixXf cp(pc1.rows(), pc1.cols());
        VectorXf distances(pc1.cols());
        MatrixXf pc1_proj(pc1.rows(), pc1.cols());

        int number_inliers = floor(overlapping_portion*((float) pc1.cols()));
        MatrixXf pc1_trimmed(3, number_inliers);
        MatrixXf pc1_proj_trimmed(3, number_inliers);
        MatrixXf cp_trimmed(3, number_inliers);
        MatrixXf R_old(3,3);
        VectorXf t_old(3);

        apply_transformation(pc1, pc1_proj, R_icp, t_icp);

        while ((((R_icp-R_old).norm() + (t_icp-t_old).norm() > ICP_EPS) || (itCt == 0)) && (itCt < MAX_ICP_IT)) {
            itCt++;

            R_old = R_icp;
            t_old = t_icp;

            find_correspondences(pc1_proj, pc2, cp, distances);

            trim_pc(pc1, cp, distances, pc1_trimmed, cp_trimmed, number_inliers);

            find_transformation(pc1_trimmed, cp_trimmed, R_icp, t_icp);

            apply_transformation(pc1, pc1_proj, R_icp, t_icp);
        }

        find_correspondences(pc1_proj, pc2, cp, distances);
        trim_pc(pc1_proj, cp, distances, pc1_proj_trimmed, cp_trimmed, number_inliers);

        return calc_error(pc1_proj_trimmed, cp_trimmed);
    }

    void trim_pc(MatrixXf &pc, MatrixXf cp, VectorXf distances, MatrixXf &pc_trimmed, MatrixXf &cp_trimmed, int number_inliers) {
        VectorXf distances_tmp = distances;

        sort(distances_tmp);

        float threshold = distances_tmp(number_inliers-1);

        int pos = 0;
        for (int i = 0; i < cp.cols(); i++) {
            if (distances(i) <= threshold && pos < number_inliers) {
                pc_trimmed(0,pos) = pc(0, i);
                pc_trimmed(1,pos) = pc(1, i);
                pc_trimmed(2,pos) = pc(2, i);

                cp_trimmed(0,pos) = cp(0, i);
                cp_trimmed(1,pos) = cp(1, i);
                cp_trimmed(2,pos) = cp(2, i);

                pos++;
            }
        }
    }

    void find_correspondences(MatrixXf const &source_pointcloud, MatrixXf const &target_pointcloud , MatrixXf &correspondences, VectorXf &distances) {
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

    void find_transformation(MatrixXf const &pointcloud, MatrixXf const &correspondences, MatrixXf &R, VectorXf &t) {
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

        t = mean2 - R*mean1;
    }

    void apply_transformation(MatrixXf const &pointcloud, MatrixXf &pointcloud_proj, MatrixXf const &R, VectorXf const &t) {
        pointcloud_proj = R*pointcloud;
        pointcloud_proj = pointcloud_proj.array().colwise() + t.array();
    }

    void sort(VectorXf &v) {
      std::sort(v.data(), v.data()+v.size());
    }

    VectorXf matrixToVector(MatrixXf m) {
        m.transposeInPlace();
        VectorXf v(Map<VectorXf>(m.data(), m.cols()*m.rows()));
        return v;
    }

    Cube **splitRotationCube(Cube *cube) {
        Cube **subcubes = new Cube*[8];
        VectorXf offset(3);
        float hel = cube->half_edge_length/2.;
        float signs[2] = {-1.,+1.};

        int position = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    offset << hel*signs[i],hel*signs[j],hel*signs[k];
                    subcubes[position++] = createRotationCube(cube->r0 + offset, hel);
                }
            }
        }

        return subcubes;
    }

    Cube **splitTranslationCube(Cube *cube) {
        Cube **subcubes = new Cube*[8];
        VectorXf offset(3);
        float hel = cube->half_edge_length/2.;
        float signs[2] = {-1.,+1.};

        int position = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    offset << hel*signs[i],hel*signs[j],hel*signs[k];
                    subcubes[position++] = createTranslationCube(cube->t0 + offset, hel);
                }
            }
        }

        return subcubes;
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
            if (queue->head->cube->lower_bound > cube->lower_bound) {
                newElement->next = queue->head;
                queue->head = newElement;
                return;
            }
            QueueElement *tmp = queue->head;
            while (tmp->next != 0) {
                if (tmp->next->cube->lower_bound > cube->lower_bound) {
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

    Cube* createRotationCube(VectorXf r0, float half_edge_length) {
        Cube *C = new Cube;
        C->r0 = r0;
        C->half_edge_length = half_edge_length;

        return C;
    }

    Cube* createTranslationCube(VectorXf t0, float half_edge_length) {
        Cube *C = new Cube;
        C->t0 = t0;
        C->half_edge_length = half_edge_length;

        return C;
    }

    void createKdTree(MatrixXf &pointcloud) {

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

        MatrixXf target_pointcloud(3, pointcloud_target->size());

        for (int i = 0; i < pointcloud_target->size(); i++) {
            target_pointcloud(0,i) = pointcloud_target->at(i).x;
            target_pointcloud(1,i) = pointcloud_target->at(i).y;
            target_pointcloud(2,i) = pointcloud_target->at(i).z;
        }

        if (true) {
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

    float calc_error(MatrixXf pc1, MatrixXf pc2) {
        MatrixXf diff = pc1 - pc2;
        diff = diff.row(0).cwiseProduct(diff.row(0)) + diff.row(1).cwiseProduct(diff.row(1)) + diff.row(2).cwiseProduct(diff.row(2));
        diff = diff.array().sqrt();

        return diff.sum();
    }

    MatrixXf *subsample_source_cloud(MatrixXf source_pointcloud, float size_source) {
        MatrixXf *source_subclouds = new MatrixXf[NUMBER_SUBCLOUDS];

        for (int i = 0; i < NUMBER_SUBCLOUDS; i++) {
            source_subclouds[i] = random_filter(source_pointcloud, size_source);
        }

        return source_subclouds;
    }

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

    bool rotationIsValid(MatrixXf R) {
        if (abs(R.determinant()-1) > MAX_NUMERICAL_ERROR || (R*R.transpose() - MatrixXf::Identity(3,3)).norm() > MAX_NUMERICAL_ERROR) {
            return false;
        }
        return true;
    }

    void initializeParameters() {
        DISTANCE_THRESHOLD = 0.02;//getFloatParameter("distance_threshold");
        //MIN_OVERLAPPING_PERCENTAGE = getFloatParameter("min_overlapping_percentage");
        TARGET_RADIUS_FACTOR = 1.3;//getFloatParameter("target_radius_factor");
        //NUMBER_SUBCLOUDS = getIntegerParameter("number_subclouds");
        SIZE_SOURCE = 250;//getIntegerParameter("size_source");
        SIZE_TARGET = 500;//getIntegerParameter("size_target");
        //REFINEMENT_ICP_SOURCE_SIZE = getIntegerParameter("refinement_icp_source_size");
        //REFINEMENT_ICP_TARGET_SIZE = getIntegerParameter("refinement_icp_target_size");
        //EVALUATION_THRESHOLD = getFloatParameter("evaluation_threshold");
        MIN_PLANE_PORTION = 0.2;//getFloatParameter("min_plane_portion");
        MIN_PLANE_DISTANCE = 0.01;//getFloatParameter("min_plane_distance");
        //MIN_SCALING_FACTOR = getFloatParameter("min_scaling_factor");
        //MAX_SCALING_FACTOR = getFloatParameter("max_scaling_factor");
        //MAX_DEPTH = getIntegerParameter("max_depth");
        ICP_EPS = 1e-5;//getFloatParameter("icp_eps");
        MAX_ICP_IT = 300;// getIntegerParameter("max_icp_it");
        //ICP_EPS2 = getFloatParameter("icp_eps2");
        //MAX_NUMERICAL_ERROR = getFloatParameter("max_numerical_error");
        //MAX_PERCENTAGE = getFloatParameter("max_percentage");
        //DAMPING_COEFFICIENT = getFloatParameter("damping_coefficient");
        //DELAY_FACTOR = getFloatParameter("delay_factor");
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

          ROS_INFO("%s:%d", parameter_name.c_str(), val);

          return val;
        } else {
            ROS_ERROR("parameter %s not found", parameter_name.c_str());

            return 0;
        }
    }



    bool stopCriterionFulfilled(float passedTime, float overlapping_percentage) {
        float min_percentage = exp(-DAMPING_COEFFICIENT*(passedTime - DELAY_FACTOR))*100;

        if (min_percentage < overlapping_percentage) {
            return true;
        } else {
            return false;
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pointcloud_alignment");

    GoIcpAction pointcloud_alignment(ros::this_node::getName());

    ros::spin();

    return 0;
}
