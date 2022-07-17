/******************************************************************************
 * Copyright 2020 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/*
 * Copyright 2018-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// headers in STL
#include <iostream>

// headers in local files
#include "modules/perception/lidar/lib/detector/point_pillars_detection/common.h"
#include "modules/perception/lidar/lib/detector/point_pillars_detection/preprocess_points_cuda.h"

namespace apollo {
namespace perception {
namespace lidar {

__global__ void make_pillar_histo_kernel(
    const float* dev_points, float* dev_pillar_point_feature_in_coors,
    int* pillar_count_histo, const int num_points,
    const int max_points_per_pillar, const int grid_x_size,
    const int grid_y_size, const int grid_z_size, const float min_x_range,
    const float min_y_range, const float min_z_range, const float pillar_x_size,
    const float pillar_y_size, const float pillar_z_size,
    const int num_point_feature)
/*
dev_points : 点云信息,从主机拷贝到设备上的
num_points ： 点数
max_points_per_pillar : 20
pillar_x_size : 0.32
pillar_y_size : 0.32
pillar_z_size : 6
min_x_range_: -74.88f
min_y_range_: -74.88f
min_z_range_: -2.0
grid_x_size_: 468
grid_y_size_: 468
grid_z_size_ : 1
num_point_feature : 5

dev_pillar_point_feature_in_coors
：记录过滤后的点云信息，在构造函数中分配内存大小，大小为grid_y_size_ *
grid_x_size_ * max_num_points_per_pillar_ *num_point_feature_ * sizeof(float)
pillar_count_histo
：在网格范围内，每个pillar的点个数，在构造函数中分配内存大小，大小为grid_y_size_
* grid_x_size_ * sizeof(int)
*/
{
  // 线程索引
  int th_i = threadIdx.x + blockIdx.x * blockDim.x;
  // 线程索引大于等于点云数，直接return
  if (th_i >= num_points) {
    return;
  }
  // 每个点所属pillar的y，x,z网格坐标
  int y_coor = floor((dev_points[th_i * num_point_feature + 1] - min_y_range) /
                     pillar_y_size);
  int x_coor = floor((dev_points[th_i * num_point_feature + 0] - min_x_range) /
                     pillar_x_size);
  int z_coor = floor((dev_points[th_i * num_point_feature + 2] - min_z_range) /
                     pillar_z_size);

  // 过滤掉网格以外
  if (x_coor >= 0 && x_coor < grid_x_size && y_coor >= 0 &&
      y_coor < grid_y_size && z_coor >= 0 && z_coor < grid_z_size) {
    // 统计历史每个pillar的点个数
    int count =
        atomicAdd(&pillar_count_histo[y_coor * grid_x_size + x_coor], 1);
    // 保证pillar的点个数小于max_points_per_pillar
    if (count < max_points_per_pillar) {
      int ind =
          y_coor * grid_x_size * max_points_per_pillar * num_point_feature + x_coor * max_points_per_pillar * num_point_feature +
          count * num_point_feature;
      for (int i = 0; i < num_point_feature; ++i) {
        // 过滤后，每个pillar中的点特征存储在dev_pillar_point_feature_in_coors
        dev_pillar_point_feature_in_coors[ind + i] = dev_points[th_i * num_point_feature + i];
      }
    }
  }
}

// 块：grid_x_size_  线程数：grid_y_size_
__global__ void make_pillar_index_kernel(

    int* dev_pillar_count_histo, int* dev_counter, int* dev_pillar_count,
    int* dev_x_coors, int* dev_y_coors, float* dev_num_points_per_pillar,
    int* dev_sparse_pillar_map, const int max_pillars,
    const int max_points_per_pillar, const int grid_x_size,
    const int num_inds_for_scan)
/*
dev_pillar_count_histo：在网格范围内，每个pillar的点个数，在构造函数中分配内存大小，大小为grid_y_size_*
grid_x_size_ * sizeof(int) max_pillars : 32000 num_inds_for_scan ： 1024


dev_counter :
dev_pillar_count : pillar的个数
dev_x_coors ：  pillar的网格坐标x
dev_y_coors :   pillar的网格坐标y
dev_num_points_per_pillar :每个pillar的点云数,保证每个pillar的点云数不大于max_points_per_pillar grid_x_size： 468 dev_sparse_pillar_map : pillar稀疏特征图  dev_sparse_pillar_map[y *
num_inds_for_scan + x] = 1;
*/
{
  int x = blockIdx.x;
  int y = threadIdx.x;
  int num_points_at_this_pillar = dev_pillar_count_histo[y * grid_x_size + x];
  if (num_points_at_this_pillar == 0) {
    return;
  }

  int count = atomicAdd(dev_counter, 1);
  if (count < max_pillars) {
    atomicAdd(dev_pillar_count, 1);
    // 保证每个pillar的点云数不大于max_points_per_pillar
    if (num_points_at_this_pillar >= max_points_per_pillar) {
      dev_num_points_per_pillar[count] = max_points_per_pillar;
    } else {
      dev_num_points_per_pillar[count] = num_points_at_this_pillar;
    }
    // pillar的网格坐标x,y
    dev_x_coors[count] = x;
    dev_y_coors[count] = y;
    dev_sparse_pillar_map[y * num_inds_for_scan + x] = 1;
  }
}

// 块：pillar个数 线程数：max_num_points_per_pillar_ 20
__global__ void make_pillar_feature_kernel(
    float* dev_pillar_point_feature_in_coors, float* dev_pillar_point_feature,
    float* dev_pillar_coors, int* dev_x_coors, int* dev_y_coors,
    float* dev_num_points_per_pillar, const int max_num_points_per_pillar,
    const int num_point_feature, const int grid_x_size)   

/*
dev_pillar_point_feature_in_coors_：记录过滤后的点云信息，在构造函数中分配内存大小，大小为grid_y_size_*grid_x_size_ * max_num_points_per_pillar_ *num_point_feature_ * sizeof(float)，存在为1
dev_pillar_point_feature ：每个pillar点特征 大小为20000×20×5
dev_pillar_coors ： 每个pillar的坐标，大小为 kMaxNumPillars * 4 * sizeof(float))
dev_x_coors ： 每个pillar的坐标x,kMaxNumPillars * sizeof(int)
dev_y_coors ： 每个pillar的坐标y,kMaxNumPillars * sizeof(int)
dev_num_points_per_pillar ：每个pillar点云数，大小为kMaxNumPillars *sizeof(float)),保证每个pillar的点云数不大于kMaxNumPointsPerPillar 
max_num_points_per_pillar : kMaxNumPointsPerPillar 20 
num_point_feature : kNumPointFeature 5 grid_x_size :468
*/
{
  // pillar的索引
  int ith_pillar = blockIdx.x;
  int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
  int ith_point = threadIdx.x;
  if (ith_point >= num_points_at_this_pillar) {
    return;
  }
  // 网格坐标xy
  int x_ind = dev_x_coors[ith_pillar];
  int y_ind = dev_y_coors[ith_pillar];
  // 类似于三维(m,20，50)
  int pillar_ind = ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature;
  int coors_ind = y_ind * grid_x_size * max_num_points_per_pillar * num_point_feature +
                  x_ind * max_num_points_per_pillar * num_point_feature +
                  ith_point * num_point_feature;
  for (int i = 0; i < num_point_feature; ++i) {
    // 代表了每个生成的跑pillar数据，三维表示为[M，20,5]，按一维存储
    dev_pillar_point_feature[pillar_ind + i] = dev_pillar_point_feature_in_coors[coors_ind + i];
  }

  float coor_x = static_cast<float>(x_ind);
  float coor_y = static_cast<float>(y_ind);
  dev_pillar_coors[ith_pillar * 4 + 0] = 0;  // batch idx
  dev_pillar_coors[ith_pillar * 4 + 1] = 0;  // z
  dev_pillar_coors[ith_pillar * 4 + 2] = coor_y;
  dev_pillar_coors[ith_pillar * 4 + 3] = coor_x;
}

PreprocessPointsCuda::PreprocessPointsCuda(
    /*
  num_threads_: 64
  max_num_pillars_: 32000
  max_num_points_per_pillar_: 20
  num_point_feature_: 5
  num_inds_for_scan_: 1024
  grid_x_size_: 468
  grid_y_size_: 468
  grid_z_size_ : 1
  pillar_x_size_: 0.32
  pillar_y_size_: 0.32
  pillar_z_size_: 6
  min_x_range_: -74.88f
  min_y_range_: -74.88f
  min_z_range_: -2.0
  */
    const int num_threads, const int max_num_pillars,
    const int max_points_per_pillar, const int num_point_feature,
    const int num_inds_for_scan, const int grid_x_size, const int grid_y_size,
    const int grid_z_size, const float pillar_x_size, const float pillar_y_size,
    const float pillar_z_size, const float min_x_range, const float min_y_range,
    const float min_z_range)
    : num_threads_(num_threads),
      max_num_pillars_(max_num_pillars),
      max_num_points_per_pillar_(max_points_per_pillar),
      num_point_feature_(num_point_feature),
      num_inds_for_scan_(num_inds_for_scan),
      grid_x_size_(grid_x_size),
      grid_y_size_(grid_y_size),
      grid_z_size_(grid_z_size),
      pillar_x_size_(pillar_x_size),
      pillar_y_size_(pillar_y_size),
      pillar_z_size_(pillar_z_size),
      min_x_range_(min_x_range),
      min_y_range_(min_y_range),
      min_z_range_(min_z_range) {
  GPU_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&dev_pillar_point_feature_in_coors_),
                 grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ *
                     num_point_feature_ * sizeof(float)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_pillar_count_histo_),
                       grid_y_size_ * grid_x_size_ * sizeof(int)));
  GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_counter_), sizeof(int)));
  GPU_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&dev_pillar_count_), sizeof(int)));
}

PreprocessPointsCuda::~PreprocessPointsCuda() {
  GPU_CHECK(cudaFree(dev_pillar_point_feature_in_coors_));

  GPU_CHECK(cudaFree(dev_pillar_count_histo_));

  GPU_CHECK(cudaFree(dev_counter_));
  GPU_CHECK(cudaFree(dev_pillar_count_));
}

void PreprocessPointsCuda::DoPreprocessPointsCuda(
    const float* dev_points, const int in_num_points, int* dev_x_coors,
    int* dev_y_coors, float* dev_num_points_per_pillar,
    float* dev_pillar_point_feature, float* dev_pillar_coors,
    int* dev_sparse_pillar_map, int* host_pillar_count)
/*
dev_points : 设备点
in_num_points ： 点数
dev_x_coors ： 每个pillar网格坐标x
dev_y_coors    每个pillar网格坐标x
dev_num_points_per_pillar ：每个pillar点云数
dev_pillar_point_feature ：每个pillar点特征 大小三维表示为(M,20,5)
dev_pillar_coors ：每个pillar的坐标(batch_index,z,y,x)
dev_sparse_pillar_map ： 稀疏pillar特征图,大小1024*1024 dev_sparse_pillar_map[y
* num_inds_for_scan + x] = 1 host_pillar_count ： pillar个数
*/
{
  GPU_CHECK(cudaMemset(dev_pillar_count_histo_, 0,
                       grid_y_size_ * grid_x_size_ * sizeof(int)));
  GPU_CHECK(cudaMemset(dev_counter_, 0, sizeof(int)));
  GPU_CHECK(cudaMemset(dev_pillar_count_, 0, sizeof(int)));
  // 线程块数：点云数/64
  int num_block = DIVUP(in_num_points, num_threads_);

  // dev_pillar_point_feature_in_coors_：记录过滤后的点云信息，在构造函数中分配内存大小，大小为grid_y_size_
  // *grid_x_size_ * max_num_points_per_pillar_ *num_point_feature_ * sizeof(float)
  make_pillar_histo_kernel<<<num_block, num_threads_>>>(
      dev_points, dev_pillar_point_feature_in_coors_, dev_pillar_count_histo_,
      in_num_points, max_num_points_per_pillar_, grid_x_size_, grid_y_size_,
      grid_z_size_, min_x_range_, min_y_range_, min_z_range_, pillar_x_size_,
      pillar_y_size_, pillar_z_size_, num_point_feature_);

  // 计算每个pillar点个数：dev_num_points_per_pillar，不满足0填充的  和 稀疏特征图 dev_sparse_pillar_map
  // 每个pillar每个pilla的网格坐标x,y : dev_x_coors,dev_y_coors
  make_pillar_index_kernel<<<grid_x_size_, grid_y_size_>>>(
      dev_pillar_count_histo_, dev_counter_, dev_pillar_count_, dev_x_coors,
      dev_y_coors, dev_num_points_per_pillar, dev_sparse_pillar_map,
      max_num_pillars_, max_num_points_per_pillar_, grid_x_size_,
      num_inds_for_scan_);
  // 将pillar个数数据 从设备拷贝到主机
  GPU_CHECK(cudaMemcpy(host_pillar_count, dev_pillar_count_, sizeof(int),
                       cudaMemcpyDeviceToHost));
  // 块：pillar个数 线程数：max_num_points_per_pillar_
  // 计算每个pillar点特征：dev_pillar_point_feature 三维表示为[M，20,5] 和每个pillar的网格坐标dev_pillar_coors [M,4]   (batch idx,z,y,x)
  make_pillar_feature_kernel<<<host_pillar_count[0],max_num_points_per_pillar_>>>(
      dev_pillar_point_feature_in_coors_, dev_pillar_point_feature,
      dev_pillar_coors, dev_x_coors, dev_y_coors, dev_num_points_per_pillar,
      max_num_points_per_pillar_, num_point_feature_, grid_x_size_);
}

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
