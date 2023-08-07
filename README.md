# Calculating Rotation of oriented bounding box Around Each Axis 

I'm currently trying to compute the rotation angles around each axis in a 3D environment using Open3D. My initial approach was to use ICP (Iterative Closest Point) to find the transformation matrix for rotating a provided bounding box so it aligns with a reference bounding box. However, I believe this method might be an overkill for my goal.

My main problem is that the default transformation matrix given by Open3D's bounding box leads to inconsistent results. I'm hypothesizing that it's feasible to achieve the desired outcome using standard geometric calculations, by leveraging the normal of the plane.

I've also written some pytests to validate new methods. I'm looking for any alternative methods or guidance on how I can make this more straightforward and consistent.


## OrientedBox Class:

Represents an OBB around a point cloud.
- The class begins with initializing the point cloud and then calculates the plane of the point cloud using RANSAC. This plane separates the inliers (points close to the plane) from the outliers.
- The class has methods to wrap the inliers in a bounding box and to compute two lines: one along the longest side and one along the shortest side of the bounding box.
- The class also has properties to get the center and the dimensions (length and width) of the bounding box.


## TranformOrientedBox Class

Its purpose is to apply transformations to an OrientedBox (OBB) to align it in a particular manner.
- The class starts by computing basic properties such as width, length, and center of the provided OrientedBox.
- sample_points_on_bbox_edges samples points on the edges of a bounding box.
- preprocess_point_cloud down-samples a point cloud, estimates its normals, and computes FPFH (Fast Point Feature Histogram) features for the points.
- refine_registration refines point cloud registration using the Iterative Closest Point (ICP) method.
- generate_target_box creates a target OBB with a specific orientation.
- execute_global_registration performs global point cloud registration using RANSAC.
- In essence, the OrientedBox class provides a representation and basic operations for OBBs, while TranformOrientedBox allows you to apply transformations to these boxes to achieve certain alignments, potentially useful in scenarios such as robotic picking tasks where alignment matters.

## Install depedencies
```
	poetry install
```
## Run the tests
```
	poetry run pytest run.py
```
