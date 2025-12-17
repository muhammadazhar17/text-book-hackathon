---
sidebar_position: 4
---

# Lesson 3: Nav2 Path Planning and Navigation

## Introduction

Navigation is a critical capability for humanoid robots operating in human environments. Navigation2 (Nav2) is the navigation stack for ROS 2 that provides path planning, obstacle avoidance, and localization capabilities essential for autonomous humanoid robotics. This lesson explores how to implement and optimize Nav2 for humanoid robotics applications, including the unique challenges and requirements of humanoid navigation in complex, human-centric environments.

Humanoid robots present distinct navigation challenges compared to traditional mobile robots due to their anthropomorphic form, need to navigate human-scale environments, and requirements for social navigation around humans.

## Learning Objectives

By the end of this lesson, you will be able to:
1. Understand the architecture and components of Navigation2 (Nav2)
2. Configure Nav2 for humanoid robot navigation requirements
3. Implement custom behaviors and planners for humanoid-specific navigation
4. Optimize navigation for human-robot interaction scenarios
5. Integrate perception systems with navigation for robust path planning
6. Deploy and validate Nav2 systems on humanoid robot platforms

## Navigation2 (Nav2) Architecture

### Overview of Nav2 Components

Navigation2 consists of several interconnected components that work together to provide complete navigation capabilities:

#### Core Components
- **Navigation Server**: Main orchestrator of navigation tasks
- **Motion Control Server**: Handles path following and control
- **Recovery Server**: Executes recovery behaviors when stuck
- **Lifecycle Manager**: Manages lifecycle of navigation components

#### Planning Components
- **Global Planner**: Computes optimal path from start to goal
- **Local Planner**: Creates safe trajectories in real-time
- **Costmap 2D**: Represents obstacles and navigation costs

#### Localization Components
- **AMCL (Adaptive Monte Carlo Localization)**: Probabilistic pose estimation
- **SLAM**: Simultaneous Localization and Mapping (when map not available)

#### Behavior Components
- **Recovery Behaviors**: Actions to take when navigation fails
- **Action Servers**: Interface for navigation commands
- **Transform Management**: Coordinate system management

### Nav2 Execution Architecture

Nav2 operates using the following workflow:
1. **Goal Reception**: Receive navigation goal from action client
2. **Global Planning**: Compute global path to goal
3. **Local Planning**: Generate safe trajectories to follow path
4. **Motion Control**: Execute control commands to drive robot
5. **Monitoring**: Continuously monitor for obstacles and re-plan as needed
6. **Recovery**: Execute recovery behaviors if navigation fails

## Humanoid-Specific Navigation Challenges

### Physical Characteristics

Humanoid robots have distinct physical characteristics that affect navigation:

#### High Center of Mass
- Requires careful path planning to maintain balance
- Need to avoid sharp turns at high speeds
- Consider dynamic stability when planning motion

#### Multi-Contact Locomotion
- Walking patterns different from wheeled robots
- Need to consider step placement for legged navigation
- Balance-aware path planning for stable locomotion

#### Anthropomorphic Dimensions
- Navigation must account for human-scale obstacles
- Doorway and corridor navigation requirements
- Interaction with human-scale furniture and structures

### Environmental Requirements

Humanoid robots are designed to operate in human environments:

#### Human-Scale Navigation
- Standard doorway heights: ~2.1m
- Standard corridor widths: ~1.2m minimum
- Furniture navigation (tables, chairs, couches)
- Staircase navigation capability

#### Social Navigation
- Navigation around humans with appropriate social distance
- Right of way protocols in crowded environments
- Respect for personal space during navigation

## Nav2 Configuration for Humanoid Robots

### Costmap Configuration

Costmaps in Nav2 represent the environment with cost values for navigation planning:

```yaml
# humanoid_costmap_params.yaml

# Global Costmap Configuration
global_costmap:
  global_costmap:
    ros__parameters:
      # Map settings
      update_frequency: 5.0
      publish_frequency: 2.0
      width: 40
      height: 40
      resolution: 0.05  # 5cm resolution for detailed humanoid navigation
      origin_x: -20.0
      origin_y: -20.0
      
      # Plugin settings
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      
      # Static layer (occupancy grid from map)
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true
      
      # Obstacle layer (sensors)
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.5  # Humanoid height consideration
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_range: 6.0
          obstacle_range: 4.0  # Humanoid needs more space
          
      # Inflation layer (safety margins)
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0  # Higher for humanoid safety
        inflation_radius: 1.0    # Larger safety margin for humanoid

# Local Costmap Configuration
local_costmap:
  local_costmap:
    ros__parameters:
      # Map settings
      update_frequency: 10.0
      publish_frequency: 5.0
      width: 10  # Smaller local window for humanoid agility
      height: 10
      resolution: 0.05
      origin_x: -5.0
      origin_y: -5.0
      
      # Plugin settings
      plugins: ["voxel_layer", "inflation_layer"]
      
      # Voxel layer (3D obstacle representation)
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: true
        publish_voxel_map: true
        origin_z: 0.0
        size_z: 10
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.5
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.5
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_range: 6.0
          obstacle_range: 4.0
          inf_is_valid: true
          
      # Inflation layer (safety margins)
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 5.0  # Higher for local obstacle avoidance
        inflation_radius: 0.8    # Balance between safety and agility
```

### Global Planner Configuration

Global path planning for humanoid robots requires special considerations:

```yaml
# global_planner_params.yaml

planners:
  GridBased:
    ros__parameters:
      # Global planner plugin
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5       # Allow some tolerance for humanoid path following
      use_astar: false     # Navfn typically performs better for humanoid scenarios
      allow_unknown: true  # Allow navigation through unknown areas if necessary
      
# Alternative: Use SmacPlanner for smoother paths better suited for humanoid locomotion
smoother:
  SmacPlannerHybrid:
    ros__parameters:
      tolerance: 0.25
      downsampling_factor: 1
      error_distance: 0.5
      cost_penalty: 1.5
      angle_quantization_bins: 72  # More bins for humanoid orientation planning  
      motion_model_for_search: "REEDS_SHEPP"  # Better for humanoid turning
```

### Local Planner Configuration

Local planning for humanoid robots differs significantly from wheeled robots:

```yaml
# local_planner_params.yaml

controller_server:
  ros__parameters:
    controller_frequency: 20.0
    min_frequency: 5.0
    max_frequency: 50.0
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]
    
    # Progress checker for humanoid navigation
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5  # Larger for humanoid stride
      movement_time_allowance: 10.0
    
    # Goal checker for humanoid navigation
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.3   # Humanoid positioning tolerance
      yaw_goal_tolerance: 0.3  # Humanoid heading tolerance
      stateful: true
    
    # Follow path controller with humanoid-specific parameters
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller/RegulatedPurePursuitController"
      desired_linear_vel: 0.5      # Conservative speed for humanoid stability
      max_linear_accel: 0.5        # Gentle acceleration for balance
      max_linear_decel: 1.0        # Faster deceleration for safety
      lookahead_dist: 0.6          # Longer lookahead for smoother movement
      min_lookahead_dist: 0.3
      max_lookahead_dist: 1.0
      lookahead_time: 1.5          # Time-based lookahead for stability
      rotate_to_heading_angular_vel: 1.0
      max_angular_accel: 2.0
      simulate_ahead_time: 1.0
      speed_regulator_gain: 150.0
      speed_limit_percentage: 0.8
      use_rotate_to_heading: true
      rotate_to_heading_min_angle: 0.2
```

## Custom Behavior Trees for Humanoid Navigation

Nav2 uses behavior trees for navigation orchestration. For humanoid robots, custom behaviors may be needed:

```xml
<!-- humanoid_navigator_bt.xml -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithReplanning">
        <RateController hz="1.0">
          <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
        </RateController>
        <RecoveryNode number_of_retries="1" name="FollowPathRecovery">
          <FollowPath path="{path}" controller_id="FollowPath"/>
          <ReactiveFailure name="GoalReachedFailure"/>
        </RecoveryNode>
      </PipelineSequence>
      <ReactiveSequence name="ClearGlobalCostmapRecovery">
        <ClearEntireCostmap name="GlobalClear" service_name="global_costmap/clear_entirely_global_costmap"/>
      </ReactiveSequence>
    </RecoveryNode>
  </BehaviorTree>

  <!-- Custom humanoid behavior: Social Navigation -->
  <BehaviorTree ID="SocialNavigate">
    <SequenceStar name="SocialNavigateSequence">
      <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
      <KeepRobotNotMovingForTime time="1.0" />  <!-- Pause for social awareness -->
      <HumanAwareFollowPath path="{path}" controller_id="FollowPath" />
    </SequenceStar>
  </BehaviorTree>
</root>
```

### Custom Humanoid Action Server

For specialized humanoid navigation behaviors:

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

import threading
import time
import math

class HumanoidNavigationActionServer(Node):
    def __init__(self):
        super().__init__('humanoid_navigation_server')
        
        # Initialize action server
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        # Navigation parameters specific to humanoid
        self.linear_velocity = 0.5      # m/s - conservative for balance
        self.angular_velocity = 0.6     # rad/s - careful turning
        self.min_distance_to_goal = 0.3 # meters - humanoid approach distance
        self.turning_radius = 0.4       # meters - minimum turning radius
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(
            String, 
            '/navigation_status', 
            10
        )
        
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        
        # TF buffer for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Navigation state
        self.current_goal = None
        self.navigation_active = False
        self.current_pose = None
        
        self.get_logger().info('Humanoid Navigation Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject navigation goal"""
        self.get_logger().info(f'Received navigation goal: {goal_request.pose}')
        
        # Check if goal is valid for humanoid navigation
        if self.is_valid_humanoid_goal(goal_request.pose):
            return GoalResponse.ACCEPT
        else:
            self.get_logger().warn(f'Rejected navigation goal: {goal_request.pose}')
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """Accept or reject goal cancellation"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute navigation goal with humanoid-specific behavior"""
        self.get_logger().info('Executing navigation goal')
        
        result = NavigateToPose.Result()
        feedback = NavigateToPose.Feedback()
        
        try:
            # Set current goal
            self.current_goal = goal_handle.request.pose
            self.navigation_active = True
            
            # Plan and execute navigation
            success = self.navigate_to_pose(goal_handle, feedback)
            
            if success:
                result.result.result = 1  # SUCCESS
                goal_handle.succeed()
                self.get_logger().info('Navigation succeeded')
            else:
                result.result.result = 4  # FAILURE
                goal_handle.abort()
                self.get_logger().warn('Navigation failed')
                
        except Exception as e:
            self.get_logger().error(f'Navigation error: {str(e)}')
            result.result.result = 4  # FAILURE
            goal_handle.abort()
        
        finally:
            self.navigation_active = False
            self.current_goal = None
            
        return result

    def navigate_to_pose(self, goal_handle, feedback):
        """Main navigation behavior for humanoid robot"""
        goal_pose = goal_handle.request.pose
        
        # Transform goal to robot's frame if needed
        transformed_goal = self.transform_to_robot_frame(goal_pose)
        if not transformed_goal:
            return False
        
        # Check initial feasibility
        if not self.check_initial_feasibility(transformed_goal):
            return False
        
        # Main navigation loop
        while rclpy.ok() and self.navigation_active:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return False
            
            # Get current robot pose
            current_pose = self.get_current_pose()
            if current_pose is None:
                self.get_logger().error('Cannot determine current pose')
                return False
            
            # Compute navigation command
            cmd = self.compute_navigation_command(current_pose, transformed_goal)
            
            if cmd is None:
                return False  # Failed to compute command
            
            # Execute command and check for obstacles
            success = self.execute_navigation_command(cmd)
            
            if not success:
                self.get_logger().error('Failed to execute navigation command')
                return False
            
            # Check if goal reached
            if self.is_goal_reached(current_pose, transformed_goal):
                self.get_logger().info('Goal reached successfully')
                return True
            
            # Update feedback
            self.update_feedback(feedback, current_pose, transformed_goal)
            goal_handle.publish_feedback(feedback)
            
            # Small delay for stability
            time.sleep(0.1)
        
        return False

    def compute_navigation_command(self, current_pose, goal_pose):
        """Compute navigation command for humanoid robot"""
        # Calculate distance to goal
        dx = goal_pose.pose.position.x - current_pose.pose.position.x
        dy = goal_pose.pose.position.y - current_pose.pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate required heading
        required_yaw = math.atan2(dy, dx)
        current_yaw = self.get_yaw_from_quaternion(current_pose.pose.orientation)
        
        # Calculate heading error
        yaw_error = self.normalize_angle(required_yaw - current_yaw)
        
        # Generate command based on distance and yaw error
        cmd = {
            'linear_vel': 0.0,
            'angular_vel': 0.0,
            'distance': distance
        }
        
        # Approach behavior: slow down as getting closer
        if distance < 1.0:
            cmd['linear_vel'] = max(0.2, distance * 0.5)  # Slow approach
        else:
            cmd['linear_vel'] = min(self.linear_velocity, distance * 0.8)  # Adjust based on distance
        
        # Turning behavior: gentle turns for balance
        cmd['angular_vel'] = max(-self.angular_velocity, min(self.angular_velocity, yaw_error * 1.5))
        
        return cmd

    def execute_navigation_command(self, cmd):
        """Execute navigation command and monitor for safety"""
        # In real implementation, this would publish to robot controllers
        # For this example, we'll just simulate the command execution
        linear_vel = cmd['linear_vel']
        angular_vel = cmd['angular_vel']
        
        # Check for obstacles before executing
        if self.is_path_blocked():
            self.get_logger().warn('Path is blocked, stopping navigation')
            return False
        
        # Publish command to robot (simplified)
        command_msg = f"LIN_VEL:{linear_vel},ANG_VEL:{angular_vel}"
        self.cmd_vel_pub.publish(String(data=command_msg))
        
        return True

    def is_goal_reached(self, current_pose, goal_pose):
        """Check if humanoid robot has reached the goal"""
        dx = goal_pose.pose.position.x - current_pose.pose.position.x
        dy = goal_pose.pose.position.y - current_pose.pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check position
        position_reached = distance <= self.min_distance_to_goal
        
        # Check orientation (if important)
        required_yaw = math.atan2(dy, dx)
        current_yaw = self.get_yaw_from_quaternion(current_pose.pose.orientation)
        yaw_error = abs(self.normalize_angle(required_yaw - current_yaw))
        orientation_reached = yaw_error <= 0.3  # 0.3 radians =~ 17 degrees
        
        return position_reached  # For humanoid, position is often more important than exact orientation

    def is_path_blocked(self):
        """Check if navigation path is blocked by obstacles"""
        # This would interface with LIDAR/sensor data
        # Simplified implementation
        return False  # Placeholder

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_yaw_from_quaternion(self, quaternion):
        """Extract yaw from quaternion"""
        import tf_transformations
        euler = tf_transformations.euler_from_quaternion([
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w
        ])
        return euler[2]  # yaw is the third element

    def get_current_pose(self):
        """Get current robot pose - would normally interface with localization"""
        # In real implementation, this would get pose from AMCL or other localization
        # For this example, return a dummy pose
        return PoseStamped()  # Placeholder

    def transform_to_robot_frame(self, pose):
        """Transform pose to robot's coordinate frame if needed"""
        # Implementation would use TF transforms
        return pose  # Placeholder

    def check_initial_feasibility(self, goal_pose):
        """Check if navigation goal is initially feasible"""
        # Check if goal is within reasonable bounds
        if abs(goal_pose.pose.position.x) > 100 or abs(goal_pose.pose.position.y) > 100:
            return False
        
        return True

    def is_valid_humanoid_goal(self, pose):
        """Check if goal pose is valid for humanoid navigation"""
        # Additional validation for humanoid-specific requirements
        return True

    def update_feedback(self, feedback, current_pose, goal_pose):
        """Update navigation feedback"""
        # Calculate distance remaining
        dx = goal_pose.pose.position.x - current_pose.pose.position.x
        dy = goal_pose.pose.position.y - current_pose.pose.position.y
        distance_remaining = math.sqrt(dx*dx + dy*dy)
        
        feedback.distance_remaining = distance_remaining

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Process LIDAR data for real-time obstacle detection
        pass

def main(args=None):
    rclpy.init(args=args)
    
    navigation_server = HumanoidNavigationActionServer()
    
    # Use multi-threaded executor to handle callbacks
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(navigation_server)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        navigation_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Social Navigation for Humanoid Robots

Humanoid robots operating in human environments must consider social navigation norms:

### Social Costmap Layer

```yaml
# social_costmap_params.yaml

social_costmap:
  social_costmap:
    ros__parameters:
      update_frequency: 10.0
      publish_frequency: 5.0
      width: 10
      height: 10
      resolution: 0.05
      origin_x: -5.0
      origin_y: -5.0
      
      plugins: ["static_layer", "obstacle_layer", "social_layer", "inflation_layer"]
      
      social_layer:
        plugin: "nav2_social_layer/SocialLayer"  # Custom layer for social navigation
        enabled: true
        humans_topic: "/humans/tracked"  # Topic with detected humans
        personal_space_radius: 0.8      # Maintain 0.8m from humans
        social_zone_radius: 1.2         # Yield to humans in 1.2m zone
        comfort_zone_weight: 10.0       # High cost for violating comfort zone
```

### Social Navigation Behavior

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration

class SocialNavigationNode(Node):
    def __init__(self):
        super().__init__('social_navigation_node')
        
        # Publishers for social navigation
        self.social_marker_pub = self.create_publisher(
            MarkerArray, 
            '/social_navigation/markers', 
            10
        )
        
        self.social_status_pub = self.create_publisher(
            String,
            '/social_navigation/status',
            10
        )
        
        # Subscriber for human tracking
        self.humans_sub = self.create_subscription(
            MarkerArray,  # Simplified - would be a custom HumanTracked message
            '/humans/tracked',
            self.humans_callback,
            10
        )
        
        # Navigation parameters for social behavior
        self.personal_space_radius = 0.8  # meters
        self.social_zone_radius = 1.2     # meters
        self.yield_speed_factor = 0.5     # Slow down near humans
        
        # Track human positions for social navigation
        self.human_positions = {}
        self.human_count = 0
        
        self.get_logger().info('Social Navigation Node initialized')

    def humans_callback(self, msg):
        """Process human tracking information"""
        for marker in msg.markers:
            human_id = marker.id
            position = marker.pose.position
            
            # Update human position and timestamp
            self.human_positions[human_id] = {
                'position': position,
                'timestamp': self.get_clock().now()
            }
        
        self.human_count = len(self.human_positions)
        
        # Filter old human positions (remove those not seen recently)
        current_time = self.get_clock().now()
        recent_threshold = Duration(sec=5)  # Remove humans not seen in 5 seconds
        
        self.human_positions = {
            hid: data for hid, data in self.human_positions.items()
            if (current_time - data['timestamp']).nanoseconds / 1e9 < 5.0
        }
        
        self.visualize_social_zones()

    def calculate_social_adjustment(self, robot_pose, target_pose):
        """Calculate navigation adjustment based on nearby humans"""
        if not self.human_positions:
            return target_pose  # No humans, no adjustment needed
        
        adjusted_pose = PoseStamped()
        adjusted_pose.header = target_pose.header
        adjusted_pose.pose = target_pose.pose
        
        # Calculate repulsive forces from nearby humans
        repulsion_force = Point(x=0.0, y=0.0, z=0.0)
        
        robot_pos = robot_pose.pose.position
        
        for human_id, human_data in self.human_positions.items():
            human_pos = human_data['position']
            
            # Calculate distance to human
            dx = robot_pos.x - human_pos.x
            dy = robot_pos.y - human_pos.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < self.social_zone_radius:
                # Calculate repulsion force (stronger when closer)
                force_magnitude = self.calculate_repulsion_force(distance)
                
                # Direction away from human
                force_direction_x = dx / distance if distance > 0 else 0
                force_direction_y = dy / distance if distance > 0 else 0
                
                # Apply stronger force when in personal space
                if distance < self.personal_space_radius:
                    force_magnitude *= 2.0
                
                repulsion_force.x += force_magnitude * force_direction_x
                repulsion_force.y += force_magnitude * force_direction_y
        
        # Adjust target position based on repulsion forces
        safety_margin = 0.5  # meters to maintain from humans
        adjusted_pose.pose.position.x += repulsion_force.x * safety_margin
        adjusted_pose.pose.position.y += repulsion_force.y * safety_margin
        
        # Log social navigation status
        status_msg = f"Near {len(self.human_positions)} humans. Adjustment: ({repulsion_force.x:.2f}, {repulsion_force.y:.2f})"
        self.social_status_pub.publish(String(data=status_msg))
        
        return adjusted_pose

    def calculate_repulsion_force(self, distance):
        """Calculate repulsion force based on distance to human"""
        # Force decreases with distance (inverse relationship)
        max_force = 1.0  # Maximum repulsion when very close
        min_distance = 0.3  # Distance at which maximum force applies
        
        if distance <= min_distance:
            return max_force
        elif distance >= self.social_zone_radius:
            return 0.0
        else:
            # Linear decrease in force
            return max_force * (self.social_zone_radius - distance) / (self.social_zone_radius - min_distance)

    def visualize_social_zones(self):
        """Visualize social zones around humans"""
        markers = MarkerArray()
        
        current_time = self.get_clock().now()
        
        for human_id, human_data in self.human_positions.items():
            # Visualize personal space
            personal_marker = Marker()
            personal_marker.header.frame_id = "map"
            personal_marker.header.stamp = current_time.to_msg()
            personal_marker.ns = "personal_space"
            personal_marker.id = human_id * 2
            personal_marker.type = Marker.CYLINDER
            personal_marker.action = Marker.ADD
            
            personal_marker.pose.position = human_data['position']
            personal_marker.pose.orientation.w = 1.0
            personal_marker.scale.x = self.personal_space_radius * 2  # diameter
            personal_marker.scale.y = self.personal_space_radius * 2  # diameter
            personal_marker.scale.z = 0.1  # height
            personal_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3)  # transparent red
            
            # Visualize social zone
            social_marker = Marker()
            social_marker.header.frame_id = "map"
            social_marker.header.stamp = current_time.to_msg()
            social_marker.ns = "social_zone"
            social_marker.id = human_id * 2 + 1
            social_marker.type = Marker.CYLINDER
            social_marker.action = Marker.ADD
            
            social_marker.pose.position = human_data['position']
            social_marker.pose.orientation.w = 1.0
            social_marker.scale.x = self.social_zone_radius * 2  # diameter
            social_marker.scale.y = self.social_zone_radius * 2  # diameter
            social_marker.scale.z = 0.05  # lower height for distinction
            social_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.2)  # transparent yellow
            
            markers.markers.extend([personal_marker, social_marker])
        
        self.social_marker_pub.publish(markers)
```

## Performance Optimization for Humanoid Navigation

### Real-Time Path Planning Optimization

Humanoid robots require careful optimization for real-time navigation:

```python
import heapq
import numpy as np
from collections import defaultdict

class OptimizedHumanoidPathPlanner:
    def __init__(self, map_resolution=0.05):
        self.resolution = map_resolution
        self.costmap = None
        self.global_path_cache = {}  # Cache for previously computed paths
        self.max_cache_size = 100
        
        # Optimized data structures for A* algorithm
        self.open_set = []
        self.closed_set = set()
        self.g_costs = {}
        self.parents = {}
        
        # Navigation optimization parameters
        self.max_iterations = 10000  # Limit iterations for real-time performance
        self.path_smoothing_factor = 0.1  # For post-processing path smoothing

    def plan_path(self, start_pos, goal_pos, costmap):
        """Optimized path planning for humanoid navigation"""
        start_cell = self.pos_to_cell(start_pos)
        goal_cell = self.pos_to_cell(goal_pos)
        self.costmap = costmap
        
        # Check cache for previously computed path
        cache_key = (start_cell, goal_cell)
        if cache_key in self.global_path_cache:
            self.get_logger().info("Using cached path")
            return self.global_path_cache[cache_key]
        
        # Run optimized A* pathfinding
        path_cells = self.optimized_astar(start_cell, goal_cell)
        
        if path_cells:
            # Convert cells back to positions
            path_pos = [self.cell_to_pos(cell) for cell in path_cells]
            
            # Smooth the path for humanoid locomotion
            smoothed_path = self.smooth_path(path_pos)
            
            # Cache the result
            self.cache_path(cache_key, smoothed_path)
            
            return smoothed_path
        else:
            return None

    def optimized_astar(self, start, goal):
        """Optimized A* algorithm for faster pathfinding"""
        # Initialize
        self.open_set = [(0, start)]  # (f_cost, position)
        self.closed_set = set()
        self.g_costs = {start: 0}
        self.parents = {start: None}
        
        iterations = 0
        
        while self.open_set and iterations < self.max_iterations:
            # Get node with lowest f-cost
            current_f, current = heapq.heappop(self.open_set)
            
            # Check if we reached the goal
            if current == goal:
                return self.reconstruct_path(current)
            
            # Skip if already processed
            if current in self.closed_set:
                continue
                
            self.closed_set.add(current)
            
            # Check neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in self.closed_set:
                    continue
                
                # Calculate tentative g-cost
                tentative_g = self.g_costs[current] + self.get_cost(current, neighbor)
                
                # If this path is better, update
                if neighbor not in self.g_costs or tentative_g < self.g_costs[neighbor]:
                    self.g_costs[neighbor] = tentative_g
                    h_cost = self.heuristic(neighbor, goal)
                    f_cost = tentative_g + h_cost
                    self.parents[neighbor] = current
                    
                    heapq.heappush(self.open_set, (f_cost, neighbor))
            
            iterations += 1
        
        # No path found
        return None

    def get_neighbors(self, pos):
        """Get valid neighbors for humanoid navigation (8-connected)"""
        neighbors = []
        x, y = pos
        
        # 8-connected neighborhood (allows diagonal movement)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip current cell
                    
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if (0 <= nx < self.costmap.shape[0] and 
                    0 <= ny < self.costmap.shape[1]):
                    
                    # Check if cell is traversable (cost is not infinite)
                    if self.costmap[nx, ny] < 254:  # Assuming 255 is obstacle
                        neighbors.append((nx, ny))
        
        return neighbors

    def get_cost(self, pos1, pos2):
        """Get movement cost between two adjacent cells"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Average cost of the two cells
        avg_cost = (self.costmap[x1, y1] + self.costmap[x2, y2]) / 2.0
        
        # Add diagonal movement penalty
        if abs(x1 - x2) == 1 and abs(y1 - y2) == 1:
            avg_cost *= 1.414  # sqrt(2) for diagonal distance
        
        return avg_cost

    def heuristic(self, pos1, pos2):
        """Manhattan distance heuristic (optimistic for fast computation)"""
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2)

    def reconstruct_path(self, goal):
        """Reconstruct path from goal to start"""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = self.parents.get(current)
        
        return path[::-1]  # Reverse to get start->goal path

    def pos_to_cell(self, pos):
        """Convert position to grid cell coordinates"""
        x = int(pos.x / self.resolution)
        y = int(pos.y / self.resolution)
        return (x, y)

    def cell_to_pos(self, cell):
        """Convert grid cell to position coordinates"""
        x, y = cell
        pos = Point()
        pos.x = x * self.resolution + self.resolution / 2
        pos.y = y * self.resolution + self.resolution / 2
        pos.z = 0.0
        return pos

    def smooth_path(self, path):
        """Smooth path for humanoid locomotion"""
        if len(path) < 3:
            return path
        
        smoothed_path = [path[0]]
        
        # Apply path smoothing algorithm
        i = 0
        while i < len(path) - 2:
            # Try to connect current point to points further ahead
            j = len(path) - 1
            
            while j > i + 1:
                if self.is_line_clear(path[i], path[j]):
                    # Add the further point and skip intermediate points
                    smoothed_path.append(path[j])
                    i = j
                    break
                j -= 1
            
            if j == i + 1:
                # No shortcut found, add next point
                smoothed_path.append(path[i + 1])
                i += 1
        
        if smoothed_path[-1] != path[-1]:
            smoothed_path.append(path[-1])
        
        return smoothed_path

    def is_line_clear(self, pos1, pos2):
        """Check if line between two points is clear of obstacles"""
        # Bresenham's line algorithm to check for obstacles
        x1, y1 = self.pos_to_cell(pos1)
        x2, y2 = self.pos_to_cell(pos2)
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            # Check if current cell is traversable
            if self.costmap[x, y] >= 254:  # Obstacle found
                return False
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True

    def cache_path(self, key, path):
        """Cache computed path with size limiting"""
        if len(self.global_path_cache) >= self.max_cache_size:
            # Remove oldest entry (in a real implementation, track insertion order)
            if self.global_path_cache:
                oldest_key = next(iter(self.global_path_cache))
                del self.global_path_cache[oldest_key]
        
        self.global_path_cache[key] = path

    def get_logger(self):
        """Simple logger for optimization feedback"""
        import sys
        return type('Logger', (), {'info': lambda _, msg: print(msg), 
                                  'warn': lambda _, msg: print(f'WARN: {msg}', file=sys.stderr),
                                  'error': lambda _, msg: print(f'ERROR: {msg}', file=sys.stderr)})()
```

## Integration with Isaac ROS Perception

### Perception-Enhanced Navigation

Combining Nav2 with Isaac ROS perception for enhanced navigation:

```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String

class PerceptionEnhancedNavigationNode(Node):
    def __init__(self):
        super().__init__('perception_enhanced_navigation')
        
        # Navigation server interface
        self.navigation_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup()
        )
        
        # Isaac ROS perception integration
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,  # Depth image
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        # Publishers for enhanced navigation
        self.enhanced_costmap_pub = self.create_publisher(
            OccupancyGrid,
            '/enhanced_costmap',
            10
        )
        
        self.dynamic_obs_pub = self.create_publisher(
            MarkerArray,
            '/dynamic_obstacles',
            10
        )
        
        # Navigation state
        self.static_costmap = None
        self.dynamic_costmap = None
        self.last_detections = []
        self.detection_timestamp = None
        
        # Perception parameters for navigation
        self.detection_trust_threshold = 0.7  # Minimum confidence for obstacles
        self.dynamic_obstacle_lifetime = 5.0  # Seconds to keep dynamic obstacles
        self.perception_influence_radius = 2.0  # Radius for perception influence
        
        self.get_logger().info('Perception Enhanced Navigation Node initialized')

    def detection_callback(self, msg):
        """Process object detections for navigation"""
        current_time = self.get_clock().now()
        
        # Filter detections based on confidence
        valid_detections = [
            det for det in msg.detections
            if any(result.score > self.detection_trust_threshold for result in det.results)
        ]
        
        # Update dynamic obstacles based on detections
        self.update_dynamic_obstacles(valid_detections, current_time)
        
        # Update enhanced costmap
        if self.static_costmap is not None:
            self.update_enhanced_costmap(current_time)
        
        self.last_detections = valid_detections
        self.detection_timestamp = current_time

    def update_dynamic_obstacles(self, detections, current_time):
        """Update dynamic obstacle representation"""
        dynamic_obstacles = []
        
        for detection in detections:
            # Convert detection to obstacle representation
            if hasattr(detection, 'results') and detection.results:
                result = detection.results[0]  # Take first result
                
                # Transform detection to map coordinates
                # This would require camera calibration and robot pose
                obstacle_pose = self.transform_detection_to_map(result)
                
                if obstacle_pose is not None:
                    obstacle = {
                        'position': obstacle_pose.pose.position,
                        'timestamp': current_time,
                        'confidence': result.score,
                        'class': result.hypothesis.names
                    }
                    dynamic_obstacles.append(obstacle)
        
        # Publish dynamic obstacles for visualization
        self.publish_dynamic_obstacles(dynamic_obstacles)

    def transform_detection_to_map(self, detection_result):
        """Transform object detection to map coordinates"""
        # This would require:
        # 1. Camera calibration parameters
        # 2. Robot pose in map
        # 3. Depth information
        # 4. Projection from image to 3D space
        
        # Simplified implementation - in reality, this would be more complex
        return PoseStamped()  # Placeholder

    def update_enhanced_costmap(self, current_time):
        """Update costmap with perception-enhanced data"""
        if self.static_costmap is None:
            return
        
        # Copy static costmap as base
        enhanced_costmap = OccupancyGrid()
        enhanced_costmap.header = self.static_costmap.header
        enhanced_costmap.info = self.static_costmap.info
        enhanced_costmap.data = list(self.static_costmap.data)
        
        # Integrate dynamic obstacles
        for obstacle in self.get_recent_dynamic_obstacles(current_time):
            self.add_dynamic_obstacle_to_costmap(
                enhanced_costmap, 
                obstacle['position'], 
                obstacle['confidence']
            )
        
        # Publish enhanced costmap
        self.enhanced_costmap_pub.publish(enhanced_costmap)

    def get_recent_dynamic_obstacles(self, current_time):
        """Get dynamic obstacles that are still valid"""
        recent_threshold = rclpy.duration.Duration(seconds=self.dynamic_obstacle_lifetime)
        
        return [
            obs for obs in self.dynamic_obstacles
            if (current_time - obs['timestamp']).nanoseconds / 1e9 < self.dynamic_obstacle_lifetime
        ]

    def add_dynamic_obstacle_to_costmap(self, costmap, position, confidence):
        """Add dynamic obstacle to costmap with confidence-based cost"""
        # Convert position to costmap cell
        try:
            cell_x = int((position.x - costmap.info.origin.position.x) / costmap.info.resolution)
            cell_y = int((position.y - costmap.info.origin.position.y) / costmap.info.resolution)
            
            # Check bounds
            if (0 <= cell_x < costmap.info.width and 
                0 <= cell_y < costmap.info.height):
                
                # Convert confidence to cost (higher confidence = higher cost)
                cost_contribution = min(254, int(confidence * 200))  # Max cost of 254
                
                # Get current cost
                current_cell_idx = cell_y * costmap.info.width + cell_x
                current_cost = costmap.data[current_cell_idx]
                
                # Update cost (take maximum to preserve static obstacles)
                new_cost = max(current_cost, cost_contribution)
                costmap.data[current_cell_idx] = new_cost
                
        except Exception as e:
            self.get_logger().warn(f'Error adding dynamic obstacle: {e}')

    def depth_callback(self, msg):
        """Process depth information for 3D obstacle detection"""
        # Integrate depth data with object detections for 3D obstacle mapping
        # This would involve creating 3D occupancy grids or voxel maps
        pass

    def publish_dynamic_obstacles(self, obstacles):
        """Publish dynamic obstacles for visualization"""
        markers = MarkerArray()
        
        current_time = self.get_clock().now()
        
        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = current_time.to_msg()
            marker.ns = "dynamic_obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            marker.pose.position = obstacle['position']
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3  # 30cm cube for obstacle
            marker.scale.y = 0.3
            marker.scale.z = 1.0  # Height for visibility
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            markers.markers.append(marker)
        
        self.dynamic_obs_pub.publish(markers)
```

## Validation and Testing

### Navigation Performance Metrics

To validate humanoid navigation, we need specific metrics:

```python
class NavigationValidator:
    def __init__(self, robot_type="humanoid"):
        self.robot_type = robot_type
        self.navigation_stats = {
            'total_paths': 0,
            'successful_paths': 0,
            'average_path_length': 0.0,
            'average_execution_time': 0.0,
            'average_speed': 0.0,
            'obstacle_awareness': 0.0,
            'social_compliance': 0.0
        }
        
        # Humanoid-specific metrics
        self.balance_metrics = {
            'average_lean_angle': 0.0,
            'balance_recovery_events': 0,
            'stability_score': 0.0
        }
    
    def evaluate_navigation_performance(self, path, execution_data):
        """Evaluate navigation performance for humanoid robot"""
        stats = {}
        
        # Path efficiency
        stats['path_efficiency'] = self.calculate_path_efficiency(path)
        
        # Execution metrics
        stats['execution_success'] = execution_data.get('success', False)
        stats['execution_time'] = execution_data.get('time', 0)
        stats['average_speed'] = execution_data.get('avg_speed', 0)
        
        # Safety metrics
        stats['min_obstacle_distance'] = execution_data.get('min_obstacle_dist', float('inf'))
        stats['obstacle_awareness'] = self.evaluate_obstacle_awareness(execution_data)
        
        # Social navigation metrics
        stats['social_compliance'] = self.evaluate_social_compliance(execution_data)
        
        # Humanoid-specific metrics
        stats['balance_stability'] = self.evaluate_balance_stability(execution_data)
        stats['step_efficiency'] = self.evaluate_step_efficiency(execution_data)
        
        return stats
    
    def calculate_path_efficiency(self, path):
        """Calculate path efficiency (length vs direct distance)"""
        if len(path) < 2:
            return 1.0  # Perfect efficiency for short paths
        
        # Calculate actual path length
        path_length = 0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_length += math.sqrt(dx*dx + dy*dy)
        
        # Calculate direct distance
        direct_distance = math.sqrt(
            (path[-1][0] - path[0][0])**2 + 
            (path[-1][1] - path[0][1])**2
        )
        
        if direct_distance == 0:
            return 1.0
        
        # Path efficiency (lower ratio means more efficient)
        return direct_distance / path_length
    
    def evaluate_obstacle_awareness(self, execution_data):
        """Evaluate how well robot avoids obstacles"""
        if not execution_data.get('obstacle_data'):
            return 0.0
        
        close_approaches = execution_data['obstacle_data'].get('close_approaches', [])
        total_obstacles = execution_data['obstacle_data'].get('total_detected', 1)
        
        # Proportion of obstacles that robot approached too closely
        dangerous_approaches = len([dist for dist in close_approaches if dist < 0.5])
        
        # Awareness score (1.0 = perfect awareness)
        awareness_score = 1.0 - (dangerous_approaches / total_obstacles)
        return max(0.0, awareness_score)
    
    def evaluate_social_compliance(self, execution_data):
        """Evaluate compliance with social navigation norms"""
        if not execution_data.get('social_data'):
            return 0.0
        
        social_violations = execution_data['social_data'].get('violations', 0)
        total_encounters = execution_data['social_data'].get('encounters', 1)
        
        # Compliance score (1.0 = no violations)
        compliance_score = 1.0 - (social_violations / total_encounters)
        return max(0.0, compliance_score)
    
    def evaluate_balance_stability(self, execution_data):
        """Evaluate humanoid balance during navigation"""
        if not execution_data.get('balance_data'):
            return 0.0
        
        lean_angles = execution_data['balance_data'].get('lean_angles', [])
        if not lean_angles:
            return 1.0  # Assume stability if no data
        
        # Calculate average lean angle
        avg_lean = sum(lean_angles) / len(lean_angles)
        
        # Balance score (lower lean = higher score)
        max_acceptable_lean = 0.3  # 0.3 radians ~= 17 degrees
        stability_score = max(0.0, 1.0 - (avg_lean / max_acceptable_lean))
        
        return stability_score
    
    def evaluate_step_efficiency(self, execution_data):
        """Evaluate efficiency of humanoid stepping patterns"""
        if not execution_data.get('locomotion_data'):
            return 0.0
        
        steps_taken = execution_data['locomotion_data'].get('steps', 0)
        
        # This would compare steps taken vs optimal stepping
        # For now, return a simplified metric
        return min(1.0, 100.0 / max(1, steps_taken))  # More steps = lower efficiency
```

## Summary

Navigation for humanoid robots requires specialized consideration due to their physical characteristics, human-centric operating environments, and social interaction requirements. This lesson covered:

1. **Nav2 Architecture**: Understanding the components and configuration needed for humanoid navigation
2. **Humanoid-Specific Challenges**: Addressing balance, anthropomorphic dimensions, and social navigation requirements
3. **Custom Behavior Trees**: Implementing specialized navigation behaviors for humanoid robots
4. **Social Navigation**: Incorporating human-aware navigation patterns
5. **Performance Optimization**: Optimizing path planning for real-time humanoid navigation
6. **Perception Integration**: Combining Nav2 with Isaac ROS perception for enhanced navigation
7. **Validation Techniques**: Evaluating navigation performance with humanoid-specific metrics

The integration of Nav2 with Isaac ROS perception and the implementation of social navigation behaviors enables humanoid robots to navigate safely and effectively in human environments while maintaining balance and respecting social norms.

## Resources and Further Reading

- Navigation2 Documentation: https://navigation.ros.org/
- "Human-Robot Interaction: Challenges and Solutions" (Research Publication)
- "Legged Robot Navigation in Complex Terrain" (Academic Reference)

## APA Citations for This Lesson

ROS Navigation2 Development Team. (2023). *Navigation2 Documentation*. Retrieved from https://navigation.ros.org/

Sisbot, R. D., et al. (2020). Socially Aware Navigation in Crowded Human Environments. *Journal of Human-Robot Interaction*, 9(2), 1-25.

Author, A. A. (2025). Lesson 3: Nav2 Path Planning and Navigation. In *Physical AI & Humanoid Robotics Course*. Physical AI Educational Initiative.