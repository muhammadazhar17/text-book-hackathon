---
sidebar_position: 5
---

# Lesson 4: Practical Exercises Integrating ROS 2 Concepts

## Introduction

Welcome to the final lesson of Module 1, where we'll put all the concepts together through practical exercises. In this lesson, we'll integrate the ROS 2 concepts we've learned—nodes, topics, services, and URDF models—into comprehensive projects that demonstrate real-world humanoid robotics applications.

This lesson is designed to be hands-on, with several practical exercises that build upon each other to create increasingly complex systems. We'll start with a simple example and gradually work up to a complete humanoid robot system that combines all the concepts from Module 1.

## Learning Objectives

By the end of this lesson, you will be able to:
1. Integrate ROS 2 nodes for communication, control, and AI decision making
2. Combine URDF models with ROS 2 for robot representation and control
3. Implement Python-based AI agents that interact with robot simulation
4. Create a complete system demonstrating all Module 1 concepts
5. Validate and test integrated robotic systems

## 1. Exercise 1: Basic Communication System

### Objective
Create a simple communication system with multiple ROS 2 nodes that simulates sensor data and processes it.

### Implementation Steps

1. **Sensor Simulation Node**: Creates mock sensor data
2. **Processing Node**: Processes sensor data
3. **Visualization Node**: Displays processed data

Here's the implementation:

**sensor_simulator.py**:
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32
import math
import random

class SensorSimulator(Node):
    def __init__(self):
        super().__init__('sensor_simulator')
        
        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu_data', 10)
        self.joint_status_pub = self.create_publisher(Float32, 'joint_temperature', 10)
        
        # Timer to publish data at 50 Hz
        self.timer = self.create_timer(0.02, self.publish_sensors)
        
        # Initialize state
        self.time = 0.0
        self.get_logger().info('Sensor simulator initialized')

    def publish_sensors(self):
        # Create and publish joint states
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = ['hip_joint', 'knee_joint', 'ankle_joint']
        joint_msg.position = [math.sin(self.time)*0.1, math.cos(self.time)*0.2, math.sin(self.time*0.5)*0.15]
        joint_msg.velocity = [math.cos(self.time)*0.1, -math.sin(self.time)*0.2, 0.5*math.cos(self.time*0.5)*0.15]
        joint_msg.effort = [0.0, 0.0, 0.0]
        
        self.joint_pub.publish(joint_msg)
        
        # Create and publish IMU data
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.orientation.x = math.sin(self.time * 0.1) * 0.1
        imu_msg.orientation.y = math.cos(self.time * 0.1) * 0.1
        imu_msg.orientation.z = 0.0
        imu_msg.orientation.w = math.sqrt(1 - (imu_msg.orientation.x**2 + imu_msg.orientation.y**2))
        imu_msg.angular_velocity.x = random.uniform(-0.1, 0.1)
        imu_msg.angular_velocity.y = random.uniform(-0.1, 0.1)
        imu_msg.angular_velocity.z = random.uniform(-0.05, 0.05)
        imu_msg.linear_acceleration.x = random.uniform(-0.5, 0.5)
        imu_msg.linear_acceleration.y = random.uniform(-0.5, 0.5)
        imu_msg.linear_acceleration.z = 9.8 + random.uniform(-0.2, 0.2)
        
        self.imu_pub.publish(imu_msg)
        
        # Publish joint temperature
        temp_msg = Float32()
        temp_msg.data = 25.0 + random.uniform(-2.0, 5.0)  # Base temp + fluctuation
        self.joint_status_pub.publish(temp_msg)
        
        self.time += 0.02  # Increment time based on timer period

def main(args=None):
    rclpy.init(args=args)
    sensor_sim = SensorSimulator()
    
    try:
        rclpy.spin(sensor_sim)
    except KeyboardInterrupt:
        sensor_sim.get_logger().info('Node stopped by user')
    finally:
        sensor_sim.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**data_processor.py**:
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String
import math

class DataProcessor(Node):
    def __init__(self):
        super().__init__('data_processor')
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)
        self.temp_sub = self.create_subscription(
            Float32, 'joint_temperature', self.temp_callback, 10)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, 'robot_status', 10)
        
        # Internal state
        self.joint_data = None
        self.imu_data = None
        self.temperature = 25.0
        
        self.get_logger().info('Data processor initialized')

    def joint_callback(self, msg):
        self.joint_data = msg
        self.process_joint_data()

    def imu_callback(self, msg):
        self.imu_data = msg
        self.process_imu_data()

    def temp_callback(self, msg):
        self.temperature = msg.data
        self.check_temperature()

    def process_joint_data(self):
        if self.joint_data is not None:
            # Calculate joint velocities and positions
            avg_pos = sum(self.joint_data.position) / len(self.joint_data.position) if self.joint_data.position else 0.0
            
            # Send a command based on joint positions
            cmd = Twist()
            cmd.linear.x = 0.1 + avg_pos * 0.5  # Move forward based on joint positions
            cmd.angular.z = avg_pos * 0.3  # Turn based on joint positions
            
            self.cmd_pub.publish(cmd)
            
            status = String()
            status.data = f"Joint avg pos: {avg_pos:.3f}, temp: {self.temperature:.1f}C"
            self.status_pub.publish(status)

    def process_imu_data(self):
        if self.imu_data is not None:
            # Calculate tilt from IMU orientation
            tilt_x = math.atan2(2.0 * (self.imu_data.orientation.w * self.imu_data.orientation.x + 
                                      self.imu_data.orientation.y * self.imu_data.orientation.z),
                               1.0 - 2.0 * (self.imu_data.orientation.x**2 + self.imu_data.orientation.y**2))
            
            # If tilted too far, send corrective command
            if abs(tilt_x) > 0.3:  # 0.3 radians ≈ 17 degrees
                correction_cmd = Twist()
                correction_cmd.angular.z = -tilt_x * 2.0  # Counter the tilt
                self.cmd_pub.publish(correction_cmd)

    def check_temperature(self):
        if self.temperature > 60.0:
            status = String()
            status.data = f"HIGH TEMPERATURE WARNING: {self.temperature:.1f}C"
            self.status_pub.publish(status)

def main(args=None):
    rclpy.init(args=args)
    processor = DataProcessor()
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        processor.get_logger().info('Node stopped by user')
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2. Exercise 2: Integrating URDF with ROS 2

### Objective
Create a system that loads a URDF model and visualizes the robot state in RViz.

### Implementation

First, create a launch file that combines everything:

**launch/humanoid_system.launch.py**:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    gui = LaunchConfiguration('gui', default='true')
    namespace = LaunchConfiguration('namespace', default='')
    
    # Get URDF file path
    urdf_file_path = os.path.join(
        get_package_share_directory('humanoid_description'),
        'urdf',
        'simple_humanoid.urdf'
    )
    
    # Read URDF file
    with open(urdf_file_path, 'r') as infp:
        robot_description = infp.read()
    
    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description
        }]
    )
    
    # Joint State Publisher node
    joint_state_publisher = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher',
        condition=IfCondition(gui),
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )
    
    # RViz node (optional)
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory('humanoid_description'), 'rviz', 'config.rviz')],
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )
    
    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        rviz
    ])
```

And the RViz configuration file:

**rviz/config.rviz**:
```yaml
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /TF1
      Splitter Ratio: 0.5
    Tree Height: 617
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
    - Class: rviz_default_plugins/SetGoal
    - Class: rviz_default_plugins/PublishPoint
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 3.0
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5
      Target Frame: base_link
      Value: Orbit (rviz)
      Yaw: 0.5
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000002f4fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d000002f4000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000002f4fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073000000003d000002f4000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d00650100000000000004500000000000000000000003a2000002f400000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1200
  X: 725
  Y: 158
```

## 3. Exercise 3: Complete Humanoid System Integration

### Objective
Create a complete system that combines ROS 2 communication, URDF modeling, and Python-based AI agents.

### Implementation

Now let's create a more complex system that includes an AI agent for humanoid behavior control:

**humanoid_behavior_controller.py**:
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32, String, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math
import numpy as np

class HumanoidBehaviorController(Node):
    def __init__(self):
        super().__init__('humanoid_behavior_controller')
        
        # Subscribers for sensor data
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)
        self.joint_temp_sub = self.create_subscription(
            Float32, 'joint_temperature', self.temperature_callback, 10)
        
        # Publishers for control commands
        self.twist_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_traj_pub = self.create_publisher(JointTrajectory, 'joint_trajectory', 10)
        self.status_pub = self.create_publisher(String, 'behavior_status', 10)
        self.fall_detector_pub = self.create_publisher(Bool, 'is_fallen', 10)
        
        # Timer for behavior control (10 Hz)
        self.behavior_timer = self.create_timer(0.1, self.behavior_control_callback)
        
        # Internal state
        self.current_joint_states = JointState()
        self.imu_orientation = Vector3()
        self.temperature = 25.0
        self.current_behavior = 'idle'
        self.fall_threshold = 0.5  # Radians for fall detection
        
        self.get_logger().info('Humanoid Behavior Controller Initialized')

    def joint_callback(self, msg):
        """Process joint state messages"""
        self.current_joint_states = msg
        self.detect_fall()

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_orientation.x = msg.orientation.x
        self.imu_orientation.y = msg.orientation.y
        self.imu_orientation.z = msg.orientation.z
        
    def temperature_callback(self, msg):
        """Process temperature data"""
        self.temperature = msg.data

    def detect_fall(self):
        """Detect if the robot has fallen based on IMU data"""
        # Calculate tilt magnitude from IMU
        tilt_magnitude = math.sqrt(
            self.imu_orientation.x**2 + 
            self.imu_orientation.y**2
        )
        
        is_fallen = Bool()
        is_fallen.data = tilt_magnitude > self.fall_threshold
        
        self.fall_detector_pub.publish(is_fallen)
        
        if is_fallen.data:
            self.get_logger().warn('FALL DETECTED! Initiating recovery sequence.')

    def behavior_control_callback(self):
        """Main behavior control loop"""
        # Determine current behavior based on sensor data
        new_behavior = self.determine_behavior()
        
        if new_behavior != self.current_behavior:
            self.get_logger().info(f'Behavior transition: {self.current_behavior} -> {new_behavior}')
            self.current_behavior = new_behavior
        
        # Execute current behavior
        self.execute_behavior()
        
        # Publish behavior status
        status_msg = String()
        status_msg.data = f"Behavior: {self.current_behavior}, Temp: {self.temperature:.1f}C"
        self.status_pub.publish(status_msg)

    def determine_behavior(self):
        """Determine appropriate behavior based on sensor data"""
        # Check if robot has fallen
        tilt_magnitude = math.sqrt(
            self.imu_orientation.x**2 + 
            self.imu_orientation.y**2
        )
        
        if tilt_magnitude > self.fall_threshold:
            return 'recover_from_fall'
        
        # Check temperature
        if self.temperature > 60.0:
            return 'cooling_down'
        
        # Check joint configuration
        if self.current_joint_states.name:
            # Check if joints are in safe configuration
            angles = list(self.current_joint_states.position)
            if angles and any(abs(angle) > 2.0 for angle in angles):
                return 'safe_mode'
        
        # Default behavior
        return 'exploring'

    def execute_behavior(self):
        """Execute the current behavior"""
        if self.current_behavior == 'recover_from_fall':
            self.execute_recovery_sequence()
        elif self.current_behavior == 'cooling_down':
            self.execute_cooling_sequence()
        elif self.current_behavior == 'safe_mode':
            self.execute_safe_mode()
        elif self.current_behavior == 'exploring':
            self.execute_exploration_sequence()

    def execute_recovery_sequence(self):
        """Execute a recovery sequence when robot falls"""
        self.get_logger().info('Executing recovery sequence...')
        
        # Send joint trajectory to stand up
        traj = JointTrajectory()
        traj.joint_names = ['left_knee', 'right_knee', 'left_hip_pitch', 'right_hip_pitch']
        
        # Create trajectory points for recovery
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0, 0.0, 0.0]  # Return to neutral position
        point.velocities = [0.5, 0.5, 0.5, 0.5]
        point.time_from_start = Duration(sec=2, nanosec=0)
        
        traj.points = [point]
        
        self.joint_traj_pub.publish(traj)
        
        # Stop any movement commands
        stop_cmd = Twist()
        self.twist_pub.publish(stop_cmd)

    def execute_cooling_sequence(self):
        """Execute cooling down sequence"""
        self.get_logger().info('Executing cooling sequence...')
        
        # Reduce movement to minimize heating
        slow_cmd = Twist()
        slow_cmd.linear.x = 0.1  # Move very slowly
        slow_cmd.angular.z = 0.0
        self.twist_pub.publish(slow_cmd)

    def execute_safe_mode(self):
        """Execute safe mode when joints are in dangerous positions"""
        self.get_logger().info('Executing safe mode...')
        
        # Move joints to safe positions
        safe_angles = [0.0] * len(self.current_joint_states.name)
        
        traj = JointTrajectory()
        traj.joint_names = self.current_joint_states.name
        
        point = JointTrajectoryPoint()
        point.positions = safe_angles
        point.velocities = [1.0] * len(safe_angles)
        point.time_from_start = Duration(sec=1, nanosec=0)
        
        traj.points = [point]
        self.joint_traj_pub.publish(traj)
        
        # Stop movement
        stop_cmd = Twist()
        self.twist_pub.publish(stop_cmd)

    def execute_exploration_sequence(self):
        """Execute normal exploration behavior"""
        # Generate exploratory movement
        cmd = Twist()
        cmd.linear.x = 0.3  # Move forward
        cmd.angular.z = 0.1 * math.sin(self.get_clock().now().nanoseconds * 1e-9)  # Gentle turning
        self.twist_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidBehaviorController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Humanoid controller stopped by user')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4. Exercise 4: System Integration and Testing

### Objective
Combine all components and test the integrated humanoid system.

### Testing Framework

Let's create a test script to validate the complete system:

**test_humanoid_system.py**:
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String, Bool
import time

class HumanoidSystemTester(Node):
    def __init__(self):
        super().__init__('humanoid_system_tester')
        
        # QoS profile for reliable communication
        qos_profile = QoSProfile(depth=10)
        
        # Publishers for commands
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        
        # Subscribers for status
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, qos_profile)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, qos_profile)
        self.status_sub = self.create_subscription(
            String, 'behavior_status', self.status_callback, qos_profile)
        self.fall_sub = self.create_subscription(
            Bool, 'is_fallen', self.fall_callback, qos_profile)
        
        # Internal state
        self.joint_states = None
        self.imu_data = None
        self.status = None
        self.is_fallen = False
        
        self.test_stage = 0
        self.test_start_time = time.time()
        
        # Timer for testing sequence
        self.test_timer = self.create_timer(1.0, self.run_test_sequence)
        
        self.get_logger().info('Humanoid System Tester Initialized')

    def joint_callback(self, msg):
        self.joint_states = msg

    def imu_callback(self, msg):
        self.imu_data = msg

    def status_callback(self, msg):
        self.status = msg.data
        self.get_logger().info(f'System Status: {msg.data}')

    def fall_callback(self, msg):
        self.is_fallen = msg.data

    def run_test_sequence(self):
        """Run the test sequence"""
        current_time = time.time()
        
        if self.test_stage == 0:
            # Test 1: Check basic communication
            self.get_logger().info('Test 1: Checking basic communication...')
            self.test_communication()
            self.test_stage = 1
            
        elif self.test_stage == 1:
            # Test 2: Check sensor data
            self.get_logger().info('Test 2: Checking sensor data...')
            self.test_sensor_data()
            self.test_stage = 2
            
        elif self.test_stage == 2:
            # Test 3: Send commands and verify response
            self.get_logger().info('Test 3: Sending commands...')
            self.test_command_response()
            self.test_stage = 3
            
        elif self.test_stage == 3:
            # Test 4: Check system behavior
            self.get_logger().info('Test 4: Checking behavior system...')
            self.test_behavior_system()
            self.test_stage = 4
            
        elif self.test_stage == 4:
            # All tests complete
            self.get_logger().info('All tests complete!')
            self.get_logger().info('System integration successful.')
            self.test_timer.cancel()

    def test_communication(self):
        """Test basic ROS communication"""
        if self.joint_states is not None:
            self.get_logger().info(f'✓ Joint states received: {len(self.joint_states.name)} joints')
        else:
            self.get_logger().warning('✗ No joint states received')
            
        if self.imu_data is not None:
            self.get_logger().info('✓ IMU data received')
        else:
            self.get_logger().warning('✗ No IMU data received')

    def test_sensor_data(self):
        """Test sensor data validity"""
        if self.joint_states and self.joint_states.position:
            joint_range_ok = all(-3.14 <= pos <= 3.14 for pos in self.joint_states.position)
            if joint_range_ok:
                self.get_logger().info('✓ Joint positions in valid range')
            else:
                self.get_logger().warning('✗ Joint positions outside valid range')
        
        if self.imu_data:
            # Check if orientation quaternion is normalized
            norm = (self.imu_data.orientation.x**2 + 
                   self.imu_data.orientation.y**2 + 
                   self.imu_data.orientation.z**2 + 
                   self.imu_data.orientation.w**2)
            if abs(norm - 1.0) < 0.1:
                self.get_logger().info('✓ IMU orientation quaternion is normalized')
            else:
                self.get_logger().warning('✗ IMU orientation quaternion not normalized')

    def test_command_response(self):
        """Test command response"""
        cmd = Twist()
        cmd.linear.x = 0.1
        cmd.angular.z = 0.1
        self.cmd_pub.publish(cmd)
        self.get_logger().info('Sent movement command')

    def test_behavior_system(self):
        """Test behavior system response"""
        if self.is_fallen:
            self.get_logger().info('✓ Fall detection working')
        else:
            self.get_logger().info('Fall detection: Not fallen (expected in normal operation)')

def main(args=None):
    rclpy.init(args=args)
    tester = HumanoidSystemTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        tester.get_logger().info('Tester stopped by user')
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 5. Exercise 5: Advanced Integration with AI

### Objective
Implement a more sophisticated AI agent that learns from sensor data and improves robot behavior over time.

### Implementation

**ai_learning_controller.py**:
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String
from builtin_interfaces.msg import Duration
import numpy as np
import random
from collections import deque

class AILearningController(Node):
    def __init__(self):
        super().__init__('ai_learning_controller')
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu_data', self.imu_callback, 10)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.reward_pub = self.create_publisher(Float32, 'reward_signal', 10)
        self.status_pub = self.create_publisher(String, 'ai_status', 10)
        
        # Timer for AI decision making (10 Hz)
        self.ai_timer = self.create_timer(0.1, self.ai_decision_callback)
        
        # Internal state
        self.joint_states = None
        self.imu_data = None
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.action_history = deque(maxlen=100)
        
        # Simple Q-learning setup
        self.state_size = 6  # joint angles, IMU orientation
        self.action_size = 4  # [forward, backward, turn_left, turn_right]
        self.q_table = {}  # State-action value table
        
        # Exploration parameters
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.get_logger().info('AI Learning Controller Initialized')

    def joint_callback(self, msg):
        self.joint_states = msg

    def imu_callback(self, msg):
        self.imu_data = msg

    def get_state(self):
        """Convert sensor data to state representation"""
        if self.joint_states is None or self.imu_data is None:
            # Return default state if no sensor data
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Extract relevant state features
        # Using first 3 joint angles and IMU orientation as state
        joint_angles = list(self.joint_states.position[:3]) if len(self.joint_states.position) >= 3 else [0.0, 0.0, 0.0]
        imu_orient = [
            self.imu_data.orientation.x,
            self.imu_data.orientation.y,
            self.imu_data.orientation.z
        ]
        
        # Pad or truncate to ensure state size
        state = (joint_angles + imu_orient)[:6]
        return tuple(round(s, 2) for s in state)  # Round for discrete state space

    def get_reward(self, state, action):
        """Calculate reward based on state and action"""
        # Calculate reward based on stability (less IMU tilt = higher reward)
        imu_tilt = abs(state[3]) + abs(state[4])  # Sum of orientation x and y
        stability_reward = max(0, 1.0 - imu_tilt)  # Higher reward for more stable
        
        # Movement reward (encourage movement but penalize instability)
        action_movement = [0.0, -0.5, 0.2, 0.2][action]  # Rewards based on action
        movement_reward = action_movement if stability_reward > 0.3 else -0.5  # Heavy penalty if unstable
        
        return stability_reward + movement_reward

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(range(self.action_size))
        
        # Exploit: best known action
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_size
        
        return self.q_table[state].index(max(self.q_table[state]))

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning algorithm"""
        learning_rate = 0.1
        discount_factor = 0.95
        
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_size
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * self.action_size
        
        # Q-learning update rule
        best_next_action_value = max(self.q_table[next_state])
        td_target = reward + discount_factor * best_next_action_value
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += learning_rate * td_error

    def ai_decision_callback(self):
        """Main AI decision making loop"""
        current_state = self.get_state()
        
        # Choose action
        action = self.choose_action(current_state)
        self.action_history.append(action)
        
        # Execute action
        cmd = self.action_to_command(action)
        self.cmd_pub.publish(cmd)
        
        # Calculate reward
        reward = self.get_reward(current_state, action)
        self.total_reward += reward
        
        # Log status
        status_msg = String()
        status_msg.data = f"Episode: {self.episode_count}, Step: {self.step_count}, Reward: {reward:.2f}, Epsilon: {self.epsilon:.3f}"
        self.status_pub.publish(status_msg)
        
        # Publish reward signal
        reward_msg = Float32()
        reward_msg.data = reward
        self.reward_pub.publish(reward_msg)
        
        # Update learning (using next state approximation)
        next_state = self.get_state()
        self.update_q_table(current_state, action, reward, next_state)
        
        # Update counters
        self.step_count += 1
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # End of episode condition (for demonstration)
        if self.step_count % 1000 == 0:
            self.get_logger().info(f'Episode {self.episode_count} completed. Total reward: {self.total_reward:.2f}')
            self.episode_count += 1
            self.step_count = 0
            self.total_reward = 0.0

    def action_to_command(self, action):
        """Convert action index to Twist command"""
        cmd = Twist()
        
        if action == 0:  # Move forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
        elif action == 1:  # Move backward
            cmd.linear.x = -0.3
            cmd.angular.z = 0.0
        elif action == 2:  # Turn left
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
        elif action == 3:  # Turn right
            cmd.linear.x = 0.0
            cmd.angular.z = -0.3
        
        return cmd

def main(args=None):
    rclpy.init(args=args)
    ai_controller = AILearningController()
    
    try:
        rclpy.spin(ai_controller)
    except KeyboardInterrupt:
        ai_controller.get_logger().info('AI Learning Controller stopped')
    finally:
        ai_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 6. Running the Complete System

### Launch Configuration

Create a comprehensive launch file to bring up the entire system:

**launch/complete_humanoid_system.launch.py**:
```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Include robot state publisher (for URDF)
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('humanoid_description'),
                'launch',
                'robot_state_publisher.launch.py'
            ])
        ])
    )
    
    return LaunchDescription([
        # Launch nodes in the required order
        Node(
            package='humanoid_system',
            executable='sensor_simulator',
            name='sensor_simulator',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),
        Node(
            package='humanoid_system',
            executable='data_processor',
            name='data_processor',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),
        Node(
            package='humanoid_system',
            executable='humanoid_behavior_controller',
            name='humanoid_behavior_controller',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),
        Node(
            package='humanoid_system',
            executable='ai_learning_controller',
            name='ai_learning_controller',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', PathJoinSubstitution([FindPackageShare('humanoid_description'), 'rviz', 'config.rviz'])],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
```

## 7. Validation and Troubleshooting

### Common Issues and Solutions

1. **Node Communication Problems**
   - Ensure all nodes are on the same namespace
   - Check topic names match between publishers and subscribers
   - Verify QoS settings are compatible

2. **URDF/TF Issues**
   - Validate URDF with `check_urdf`
   - Ensure robot_state_publisher is running
   - Check joint_state_publisher is providing data

3. **AI Learning Problems**
   - Ensure sufficient exploration during learning
   - Verify reward function is appropriate
   - Check state representation is informative

### Testing Checklist

- [ ] All nodes launch without errors
- [ ] Topics are properly connected
- [ ] Sensor data is being published and received
- [ ] URDF is loaded and visualized correctly
- [ ] AI controller is making decisions
- [ ] Safety systems are functioning
- [ ] Behavior transitions are working

## 8. Performance Optimization

### For Real-time Systems

When deploying on real hardware, consider:

1. **Efficient Data Processing**: Use NumPy for numerical operations
2. **Threading**: Separate sensor processing from AI decision making
3. **Memory Management**: Be mindful of memory usage in long-running applications
4. **Computational Complexity**: Simplify algorithms if needed for real-time performance

## Summary

In this lesson, we've integrated all the concepts from Module 1 into practical exercises:

1. We created a basic communication system with multiple nodes
2. We integrated URDF models with ROS 2 systems
3. We built a complete humanoid behavior controller
4. We implemented an AI learning system for adaptive behavior
5. We validated the integrated system through comprehensive testing

These exercises demonstrate how the individual concepts of ROS 2 (nodes, topics, services), Python integration (rclpy), and robot modeling (URDF) work together to create complete humanoid robotic systems. The skills developed in these exercises form the foundation for more advanced humanoid robotics applications.

## References and Further Reading

- ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html
- URDF Tutorials: http://wiki.ros.org/urdf/Tutorials
- MoveIt! Motion Planning: http://moveit.ros.org/
- Gazebo Simulation: http://gazebosim.org/tutorials

## APA Citations for This Lesson

Open Robotics. (2023). *ROS 2 Documentation*. Retrieved from https://docs.ros.org/en/humble/

When referencing this educational content:

Author, A. A. (2025). Lesson 4: Practical Exercises Integrating ROS 2 Concepts. In *Physical AI & Humanoid Robotics Course*. Physical AI Educational Initiative.