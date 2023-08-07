import numpy as np
import open3d as o3d
from loguru import logger
import copy
from scipy.spatial.transform import Rotation
import pytest

def argmedian(x):
    """Find the index of the median value in an array.

    Parameters:
    - x (numpy.ndarray): Input 1D numpy array.

    Returns:
    - int: Index of the median value.
    """
    return np.argpartition(x, len(x) // 2)[len(x) // 2]


class OrientedBox:
    """Class to represent an oriented bounding box (OBB) around a point cloud.

    Attributes:
    - pc (o3d.geometry.PointCloud): Input point cloud.
    - cf (o3d.geometry.TriangleMesh): Coordinate frame.
    """

    def __init__(self, pc: o3d.geometry.PointCloud):
        """Initialize the OrientedBox.

        Parameters:
        - pc (o3d.geometry.PointCloud): Input point cloud.
        """
        self.pc = pc
        self._calculate_plane()
        self._wrap_pc_by_bounding_box()
        self._initialize_skeleton_lines()

    def _calculate_plane(
        self,
    ):
        """
        Calculate the plane of the point cloud using RANSAC.

        This method estimates the normals of the point cloud, orients them towards the camera, and then performs RANSAC
        to find the plane model that best fits the inlier points.

        Note: This method is called internally during object initialization.
        """
        self.pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=50, max_nn=50)
        )

        self.pc.orient_normals_towards_camera_location()
        self.pc.normalize_normals()
        self.plane_model, inliers = self.pc.segment_plane(
            distance_threshold=5, ransac_n=30, num_iterations=500
        )

        self.inlier_cloud = self.pc.select_by_index(inliers)
        self.outlier_cloud = self.pc.select_by_index(inliers, invert=True)
        self.top_mesh = o3d.geometry.TetraMesh.create_from_point_cloud(
            self.inlier_cloud
        )[0]

    def _create_line_along_long_side(self):
        """
        Create a line along the longest side of the bounding box.

        This method finds the axis with the largest extent (longest side) of the bounding box and forms a line
        along that axis.

        Note: This method is called internally during object initialization.
        """
        # Get the rotation matrix of the bounding box
        rotation_matrix = self.bounding_box.R

        # Find the axis with the largest extent (longest axis)
        extent = self.bounding_box.extent
        largest_extent_index = np.argmax(self.bounding_box.extent)
        center_point = self.bounding_box.get_center()
        # Find the two points along the longest axis to form the middle line
        point1 = (
            center_point
            - 0.5
            * extent[largest_extent_index]
            * rotation_matrix[:, largest_extent_index]
        )
        point2 = (
            center_point
            + 0.5
            * extent[largest_extent_index]
            * rotation_matrix[:, largest_extent_index]
        )

        # Convert points to NumPy arrays for convenience (optional)
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)

        self.long_line = o3d.geometry.LineSet()
        self.long_line.points = o3d.utility.Vector3dVector([point1, point2])
        lines = [[0, 1]]
        self.long_line.lines = o3d.utility.Vector2iVector(lines)
        line_colors = [[1, 0, 0] for i in range(len(lines))]
        self.long_line.colors = o3d.utility.Vector3dVector(line_colors)

    def _create_line_along_short_side(self):
        """
        Create a line along the shortest side of the bounding box.

        This method finds the axis with the smallest extent (shortest side) of the bounding box and forms a line
        along that axis.

        Note: This method is called internally during object initialization.
        """
        # Get the rotation matrix of the bounding box
        rotation_matrix = self.bounding_box.R

        # Find the axis with the smallest extent (shortest side)
        extent = self.bounding_box.extent
        shortest_extent_index = argmedian(extent)
        center_point = self.bounding_box.get_center()

        # Find the two points along the shortest axis to form the middle line
        point1 = (
            center_point
            - 0.5
            * extent[shortest_extent_index]
            * rotation_matrix[:, shortest_extent_index]
        )
        point2 = (
            center_point
            + 0.5
            * extent[shortest_extent_index]
            * rotation_matrix[:, shortest_extent_index]
        )

        # Convert points to NumPy arrays for convenience (optional)
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)

        # Create the short line as a LineSet
        self.short_line = o3d.geometry.LineSet()
        self.short_line.points = o3d.utility.Vector3dVector([point1, point2])
        lines = [[0, 1]]
        self.short_line.lines = o3d.utility.Vector2iVector(lines)
        line_colors = [
            [0, 1, 0] for i in range(len(lines))
        ]  # Green color for the short line
        self.short_line.colors = o3d.utility.Vector3dVector(line_colors)

    def _width(
        self,
    ):
        """
        Calculate the width of the bounding box.

        Returns:
        - float: The width of the bounding box.
        """
        points = np.asarray(self.short_line.points)
        # Calculate the Euclidean distances between consecutive points
        distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
        total_length = np.sum(distances)
        return total_length

    def _length(self):
        """
        Calculate the length of the bounding box.

        Returns:
        - float: The length of the bounding box.
        """
        points = np.asarray(self.long_line.points)
        # Calculate the Euclidean distances between consecutive points
        distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
        total_length = np.sum(distances)
        return total_length

    @property
    def center(self):
        """
        Get the center point of the bounding box.

        Returns:
        - np.ndarray: The (x, y, z) coordinates of the center point.
        """
        return self.bounding_box.get_center()

    @property
    def dimensions(self):
        """
        Get the dimensions (length and width) of the bounding box.

        Returns:
        - Tuple[float, float]: A tuple containing the length and width of the bounding box.
        """
        return self._length(), self._width()

    def _wrap_pc_by_bounding_box(self):
        """
        Wrap the point cloud with an oriented bounding box.

        This method fits an oriented bounding box to the inlier points of the point cloud.

        Note: This method is called internally during object initialization.
        """
        self.bounding_box = self.inlier_cloud.get_minimal_oriented_bounding_box(
            robust=True
        )

    def _initialize_skeleton_lines(self):
        """
        Initialize the skeleton lines along the long and short sides of the bounding box.

        This method creates line segments representing the long and short sides of the bounding box.

        Note: This method is called internally during object initialization.
        """
        self._create_line_along_long_side()
        self._create_line_along_short_side()


class TranformOrientedBox:
    # Transformation that levels an OrientedBox and returns the rotation matrix
    def __init__(self, oriented_box: OrientedBox):
        dimensions = oriented_box.dimensions
        self.width = dimensions[1]
        self.length = dimensions[0]
        self.center = oriented_box.center
        self.source = oriented_box.bounding_box

    def _sample_points_on_bbox_edges(
        self, bbox: o3d.geometry.OrientedBoundingBox, num_points: int = 100
    ) -> np.ndarray:
        """
        Samples points on the edges of the given bounding box.

        Parameters:
        - bbox: The input Open3D oriented bounding box.
        - num_points: Number of points to sample on each edge.

        Returns:
        - A numpy array of shape (12 * num_points, 3), where each row represents the (x, y, z) coordinates of a sampled point.
        """

        corners = np.asarray(bbox.get_box_points())

        # Define the 12 edges using the corner indices
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        sampled_points = []

        for start, end in edges:
            # Linearly interpolate between the two corner points
            for alpha in np.linspace(0, 1, num_points):
                point = (1 - alpha) * corners[start] + alpha * corners[end]
                sampled_points.append(point)

        points_on_edges = np.vstack(sampled_points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_on_edges)
        return pcd

    def _preprocess_point_cloud(self, pcd, voxel_size):
        """
        Preprocesses a point cloud using voxel down-sampling, normal estimation, and FPFH feature computation.

        Parameters:
        - pcd (o3d.geometry.PointCloud): The input point cloud to preprocess.
        - voxel_size (float): Voxel size for down-sampling.

        Returns:
        - Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]: A tuple containing the down-sampled point cloud and its associated FPFH feature.
        """
        logger.info(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        logger.info(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20)
        )
        pcd_down.orient_normals_towards_camera_location()

        radius_feature = voxel_size * 5
        logger.info(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=25),
        )
        return pcd_down, pcd_fpfh

    def _refine_registration(
        self,
        source,
        target,
        source_fpfh,
        target_fpfh,
        voxel_size,
        distance_threshold,
        transformation,
    ):
        """
        Refines the point cloud registration using ICP (Iterative Closest Point) algorithm.

        Parameters:
        - source (o3d.geometry.PointCloud): Source point cloud.
        - target (o3d.geometry.PointCloud): Target point cloud.
        - source_fpfh (o3d.pipelines.registration.Feature): FPFH feature of the source point cloud.
        - target_fpfh (o3d.pipelines.registration.Feature): FPFH feature of the target point cloud.
        - voxel_size (float): Voxel size for down-sampling.
        - distance_threshold (float): Threshold for correspondence rejection during ICP.
        - transformation (np.ndarray): Initial transformation matrix.

        Returns:
        - o3d.pipelines.registration.RegistrationResult: The result of the ICP registration.
        """
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            distance_threshold,
            transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        return result

    def _generate_target_box(
        self,
    ):
        """
        Generates a target oriented bounding box based on the dimensions and center.

        Returns:
        - o3d.geometry.OrientedBoundingBox: The generated target oriented bounding box.
        """
        generated_box = o3d.geometry.TriangleMesh.create_box(
            width=self.width, height=self.length, depth=1
        )
        ## always 90 degrees rotated of the center to match the pickup location of the robot
        R = generated_box.get_rotation_matrix_from_xyz((0, 0, np.radians(90)))
        generated_box.translate([-self.width / 2, -self.length / 2, 0])
        generated_box.translate(self.center)
        generated_box.rotate(R)
        return generated_box.get_oriented_bounding_box()

    def _execute_global_registration(
        self, source_down, target_down, source_fpfh, target_fpfh, voxel_size
    ):
        """
        Executes global point cloud registration using RANSAC-based feature matching.

        Parameters:
        - source_down (o3d.geometry.PointCloud): Down-sampled source point cloud.
        - target_down (o3d.geometry.PointCloud): Down-sampled target point cloud.
        - source_fpfh (o3d.pipelines.registration.Feature): FPFH feature of the down-sampled source point cloud.
        - target_fpfh (o3d.pipelines.registration.Feature): FPFH feature of the down-sampled target point cloud.
        - voxel_size (float): Voxel size for down-sampling.

        Returns:
        - o3d.pipelines.registration.RegistrationResult: The result of the RANSAC-based registration.
        """
        distance_threshold = voxel_size * 1.5
        logger.info(":: RANSAC registration on downsampled point clouds.")

        logger.info("   Since the downsampling voxel size is %.3f," % voxel_size)
        logger.info("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                source_fpfh,
                target_fpfh,
                True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999),
            )
        )
        return result

    def __call__(self):
        """
        Performs the transformation of the source OrientedBox to match it with the target object.

        Returns:
        - np.ndarray: The transformation matrix that aligns the source OrientedBox with the target object.
        """
        self.target = self._generate_target_box()
        self.target = self._sample_points_on_bbox_edges(self.target)
        self.source = self._sample_points_on_bbox_edges(self.source)

        voxel_size = 10

        source_down, source_fpfh = self._preprocess_point_cloud(self.source, voxel_size)
        target_down, target_fpfh = self._preprocess_point_cloud(self.target, voxel_size)

        result = self._execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )
        result = self._refine_registration(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            voxel_size,
            100,
            np.identity(4),
        )
        result = self._refine_registration(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            voxel_size,
            25,
            result.transformation,
        )

        return result.transformation


### DEMO purposes
def _generate_a_sample_top_plane_of_box(input_rot):
    number_of_points_to_sample = 5000

    def generate_sample_box():
        generated_box = o3d.geometry.TriangleMesh.create_box(
            width=200, height=500, depth=5
        )

        R = generated_box.get_rotation_matrix_from_xyz((0, 0, np.radians(90)))
        generated_box.rotate(R)
        generated_box.translate([0, 0,  400])

        roi = generated_box.sample_points_uniformly(
            number_of_points=number_of_points_to_sample
        )
        return generated_box, roi

    generated_box, roi1 = generate_sample_box()
    R = generated_box.get_rotation_matrix_from_quaternion(input_rot)
    
    generated_box.rotate(R, center=(generated_box.get_center()))

    generated_box_sampled_points = generated_box.sample_points_uniformly(
        number_of_points=5000
    )

    generated_box_sampled_points.estimate_normals()
    return generated_box_sampled_points



def simulate_rotation_and_validate_rotation_matrix(input_rot):
    box = _generate_a_sample_top_plane_of_box(input_rot)
    box = OrientedBox(box)
    transform_f = TranformOrientedBox(box)
    transformation_matrix = transform_f()
    rotation_matrix = transformation_matrix[:3, :3]

    degrees = Rotation.from_matrix(np.copy(rotation_matrix)).as_quat()
    #Visualize if needed

    # source = transform_f.source
    # target = transform_f.target

    # source.paint_uniform_color([1, 0, 0])
    # target.paint_uniform_color([0, 0, 1])
    # cf = o3d.geometry.TriangleMesh.create_coordinate_frame(150)
    # output = copy.deepcopy(box.inlier_cloud)
    # output.transform(transformation_matrix)
    # o3d.visualization.draw_geometries([source, target, cf, output])
    return degrees

def quaternion_to_euler(q):
    """Convert a quaternion to Euler angles in degrees."""
    r = Rotation.from_quat(q)
    return r.as_euler('xyz', degrees=True)

@pytest.mark.parametrize("rot_x, rot_y, rot_z", [
    (10,0,0),
    (30,0,0),
    (-15,0,0),
    (55, 0, 0),
    (55, 0, 20),
    (12, 0, 15),
    (0, 15, 40),
    (0, 25,-4),
    (0, 8, 12),
    (-15, 16, 69),
    (0, 8, 12),
    (0, 8, 80),
    (0, 8, -60),
    (0,12 , -12),
    (-15, 0, -30),
    (10, 20, 30),
    (75, 5, 65),
    (-15, 30, -20),
    (23, 40, -23),
    (5, 25, 55),
    (0, 0, 0)       # No rotation
])
def test_simulate_rotation_and_validate_rotation_matrix(rot_x, rot_y, rot_z):
    # Test the function with different rotation angles
    
    input_quaternion = Rotation.from_euler('zyx', [rot_x, rot_y, rot_z], degrees=True).as_quat() # for some reason rotation sequence must be applied in this order, validated with open3d rotation in visualizer
    calculated_quaternion = simulate_rotation_and_validate_rotation_matrix(input_quaternion)    
    calculated_angles = quaternion_to_euler(calculated_quaternion)
    calculated_angles[1] *= -1
    assert np.allclose(calculated_angles, [rot_x, rot_y, rot_z], atol=5), \
        f"Rotation angles mismatch for input ({rot_x}, {rot_y}, {rot_z}). " \
        f"Expected: {[rot_x, rot_y, rot_z]}, Got: {calculated_angles}"
