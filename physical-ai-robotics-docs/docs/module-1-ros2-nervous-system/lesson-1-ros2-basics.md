---
sidebar_position: 2
---

# Lesson 1: ROS 2 Nodes, Topics, and Services

## Introduction

In this lesson, we'll explore the fundamental building blocks of ROS 2: nodes, topics, and services. These components form the backbone of robotic systems, enabling different parts of a robot to communicate effectively. Understanding these concepts is crucial for developing humanoid robots that can coordinate complex behaviors between multiple subsystems.

## Learning Objectives

By the end of this lesson, you will be able to:
1. Explain the difference between nodes, topics, and services in ROS 2 architecture
2. Create ROS 2 nodes using both Python and C++
3. Implement topic-based communication for continuous data streams
4. Implement service-based communication for request-response interactions
5. Design appropriate communication patterns for humanoid robot systems

## 1. Nodes: The Fundamental Units of Computation

### What are Nodes?

In ROS 2, a node is a process that performs computation. Nodes are the fundamental building blocks of a ROS program, similar to objects in object-oriented programming. Each node in a robotic system typically performs a specific function, such as sensor processing, actuator control, or higher-level decision making.

For humanoid robots, nodes might include:
- Joint controller nodes that manage individual servo motors
- Perception nodes that process camera images or LIDAR data
- Planning nodes that compute robot motions
- Behavior nodes that coordinate different robot activities

### Node Architecture

A node contains:
- **Publishers**: Interfaces for sending messages on topics
- **Subscribers**: Interfaces for receiving messages on topics
- **Services**: Interfaces for providing synchronous services
- **Service Clients**: Interfaces for using services provided by other nodes
- **Parameters**: Configurable values that can be changed at runtime
- **Actions**: Interfaces for long-running tasks with feedback

### Creating Nodes in Python

Here's a more detailed example of a ROS 2 node in Python, implementing a joint controller for a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        
        # Publisher for joint commands
        self.joint_cmd_publisher = self.create_publisher(
            Float64MultiArray, 
            '/joint_group_position_controller/commands', 
            10
        )
        
        # Subscriber for current joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Timer for command updates (50 Hz)
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        # Internal state
        self.current_positions = []
        self.target_positions = [0.0] * 20  # 20 joints for humanoid
        
        self.get_logger().info('Joint Controller node initialized')

    def joint_state_callback(self, msg):
        """Callback function for joint state messages"""
        self.current_positions = list(msg.position)
        self.get_logger().debug(f'Updated joint positions: {self.current_positions}')

    def timer_callback(self):
        """Callback function for timer - publishes joint commands"""
        # In a real implementation, this would compute appropriate commands
        # based on target positions, current positions, and control algorithms
        cmd_msg = Float64MultiArray()
        cmd_msg.data = self.compute_joint_commands()
        
        self.joint_cmd_publisher.publish(cmd_msg)
        self.get_logger().debug(f'Published joint commands: {cmd_msg.data}')

    def compute_joint_commands(self):
        """Compute joint commands based on current and target positions"""
        # Simple proportional control as an example
        commands = []
        for curr, target in zip(self.current_positions, self.target_positions):
            # In practice, this would involve more sophisticated control algorithms
            error = target - curr
            command = 0.1 * error  # Simple proportional control
            commands.append(curr + command)
        return commands

def main(args=None):
    rclpy.init(args=args)
    
    joint_controller = JointController()
    
    try:
        rclpy.spin(joint_controller)
    except KeyboardInterrupt:
        pass
    finally:
        joint_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle Management

ROS 2 provides lifecycle nodes that have well-defined states and transitions between:
- Unconfigured
- Inactive
- Active
- Finalized

This is particularly important for humanoid robots where safe transitions between operational states are critical.

## 2. Topics: Asynchronous Publish-Subscribe Communication

### Topic Communication Model

Topics in ROS 2 enable asynchronous communication through a publish-subscribe model. Publishers send messages to a topic, and subscribers receive messages from the same topic. This allows for loose coupling between nodes - publishers don't need to know who is listening, and subscribers don't need to know who is publishing.

For humanoid robots, topics are ideal for:
- Sensor data streams (camera images, IMU readings, joint states)
- Robot state information (battery level, temperature, error states)
- Environmental data (detected objects, map data, navigation goals)
- Control commands (joint positions, velocities, efforts)

### Quality of Service (QoS) in Topics

ROS 2 allows you to specify Quality of Service policies for topics, which is critical in humanoid robotics where real-time performance and reliability are essential:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

# QoS for critical control commands
cmd_qos = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE
)

# QoS for sensor data (may drop messages to maintain real-time performance)
sensor_qos = QoSProfile(
    depth=5,
    durability=QoSDurabilityPolicy.VOLATILE,
    reliability=QoSReliabilityPolicy.BEST_EFFORT
)

self.cmd_publisher = self.create_publisher(JointCommand, 'joint_commands', cmd_qos)
self.sensor_subscriber = self.create_subscription(SensorData, 'sensor_data', self.sensor_callback, sensor_qos)
```

### Message Types for Humanoid Robots

Common message types used in humanoid robotics include:
- `sensor_msgs/JointState`: Current joint positions, velocities, and efforts
- `geometry_msgs/Twist`: Linear and angular velocities
- `sensor_msgs/Image`: Camera images
- `sensor_msgs/PointCloud2`: 3D point cloud data
- `nav_msgs/Odometry`: Robot pose and velocity information

### Example: Implementing a Sensor Fusion Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Vector3
from tf2_ros import TransformBroadcaster
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Subscribers for different sensor types
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Publisher for fused state
        self.fused_state_pub = self.create_publisher(
            JointState,
            '/fused_robot_state',
            10
        )
        
        # Transform broadcaster for TF tree
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Internal state
        self.imu_data = None
        self.joint_data = None
        
        self.get_logger().info('Sensor Fusion node initialized')

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg
        self.process_sensor_fusion()

    def joint_callback(self, msg):
        """Process joint state data"""
        self.joint_data = msg
        self.process_sensor_fusion()

    def process_sensor_fusion(self):
        """Combine sensor data into a coherent robot state"""
        if self.imu_data is not None and self.joint_data is not None:
            # This is a simplified example
            # In practice, this would involve Kalman filters,
            # particle filters, or other fusion techniques
            fused_state = self.joint_data
            fused_state.header.stamp = self.get_clock().now().to_msg()
            
            # Add IMU-based orientation to the state
            # (This is simplified - in reality you'd use proper fusion)
            # fused_state.orientation = self.imu_data.orientation
            
            self.fused_state_pub.publish(fused_state)
            
            # Publish relevant transforms
            self.publish_transforms(fused_state)

    def publish_transforms(self, state):
        """Publish transforms based on joint states"""
        # Example: publish a transform based on joint data
        # In a humanoid robot, this would update the kinematic chain
        pass

def main(args=None):
    rclpy.init(args=args)
    
    sensor_fusion = SensorFusionNode()
    
    try:
        rclpy.spin(sensor_fusion)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_fusion.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 3. Services: Synchronous Request-Response Communication

### Service Communication Model

Services in ROS 2 enable synchronous, request-response communication between nodes. A service client sends a request to a service server, which processes the request and returns a response. This is appropriate for operations that have a clear request-response pattern and don't need to be ongoing.

For humanoid robots, services are ideal for:
- Computing inverse kinematics for a target position
- Saving or loading robot configurations
- Triggering calibration procedures
- Requesting specific robot behaviors
- Querying persistent robot state information

### Service Definition

Services are defined in `.srv` files that specify the request and response message types:

```
# GetJointEffort.srv
# Request: joint_name (string)
# Response: effort (float64)
string joint_name
---
float64 effort
```

### Implementing Services

Here's an example of a service server that provides inverse kinematics calculations for a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger
import math

class RobotControllerService(Node):
    def __init__(self):
        super().__init__('robot_controller_service')
        
        # Create services
        self.ik_service = self.create_service(
            InverseKinematics,  # This would be a custom service type
            'compute_ik',
            self.compute_ik_callback
        )
        
        self.calibrate_service = self.create_service(
            Trigger,
            'calibrate_robot',
            self.calibrate_callback
        )
        
        self.get_logger().info('Robot Controller Service initialized')

    def compute_ik_callback(self, request, response):
        """Compute inverse kinematics for a target position"""
        try:
            # Simplified example - in reality this would be complex
            target_x = request.target_pose.position.x
            target_y = request.target_pose.position.y
            target_z = request.target_pose.position.z
            
            # Perform inverse kinematics calculation
            joint_angles = self.calculate_ik(target_x, target_y, target_z)
            
            response.joint_angles = joint_angles
            response.success = True
            response.message = "IK computation successful"
            
        except Exception as e:
            response.success = False
            response.message = f"IK computation failed: {str(e)}"
            
        return response

    def calculate_ik(self, x, y, z):
        """Placeholder for inverse kinematics calculation"""
        # This would contain the actual IK algorithm
        # For a humanoid arm, this might involve analytical or numerical methods
        return [0.0, 0.0, 0.0, 0.0, 0.0]  # Placeholder joint angles

    def calibrate_callback(self, request, response):
        """Perform robot calibration"""
        try:
            self.get_logger().info('Starting robot calibration...')
            
            # Perform calibration procedure
            success = self.perform_calibration()
            
            if success:
                response.success = True
                response.message = "Calibration completed successfully"
            else:
                response.success = False
                response.message = "Calibration failed"
                
        except Exception as e:
            response.success = False
            response.message = f"Calibration error: {str(e)}"
            
        return response

    def perform_calibration(self):
        """Placeholder for actual calibration procedure"""
        # This would involve moving joints to known positions
        # and adjusting parameters accordingly
        return True  # Placeholder

def main(args=None):
    rclpy.init(args=args)
    
    controller_service = RobotControllerService()
    
    try:
        rclpy.spin(controller_service)
    except KeyboardInterrupt:
        pass
    finally:
        controller_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Clients

Service clients call services provided by other nodes:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')
        
        # Create client
        self.calibrate_client = self.create_client(Trigger, 'calibrate_robot')
        
        # Wait for service to be available
        while not self.calibrate_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Calibration service not available, waiting...')
        
        # Call the service
        self.call_calibrate_service()

    def call_calibrate_service(self):
        """Call the calibration service"""
        request = Trigger.Request()
        
        future = self.calibrate_client.call_async(request)
        future.add_done_callback(self.calibrate_callback)

    def calibrate_callback(self, future):
        """Handle the response from the calibration service"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Calibration successful: {response.message}')
            else:
                self.get_logger().error(f'Calibration failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    client = ServiceClient()
    
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4. Communication Patterns for Humanoid Robots

### Sensor-Processing-Actuation Pattern

A common pattern in humanoid robotics is the sensor-processing-actuation loop:

```
Sensors → Sensor Processing → AI/Decision Making → Actuator Commands
```

Each of these components would typically be implemented as separate nodes connected by ROS 2 topics and services.

### Coordination Between Multiple Controllers

Humanoid robots often have multiple controllers (e.g., walking controller, arm controller, head controller). These need to coordinate to avoid conflicts. This can be achieved using:
- Services to request permission for certain actions
- Topics to broadcast intent and status
- Parameter servers to coordinate priorities

### Fault Tolerance and Safety

ROS 2 provides tools for creating robust communication patterns:
- Latching topics that keep the last published message for new subscribers
- Reliable delivery options for critical messages
- Node health monitoring and automatic recovery

## 5. Best Practices for Topic and Service Design

### Naming Conventions

Use descriptive, consistent names for topics and services:
- `/robot_name/sensor_type/data` for sensor topics (e.g., `/humanoid1/camera/rgb/image_raw`)
- `/robot_name/action_type` for action topics (e.g., `/humanoid1/arm_controller/command`)
- `/robot_name/service_type` for services (e.g., `/humanoid1/compute_ik`)

### Message Design

Design messages that:
- Include appropriate time stamps
- Use SI units consistently
- Include status information where appropriate
- Are efficient in size for real-time systems

### Error Handling

Implement robust error handling:
- Check for message validity
- Handle missing or out-of-order messages
- Implement timeouts for service calls
- Provide fallback behaviors

## 6. Practical Exercise

### Exercise: Implement a Simple Humanoid Communication System

Create a simple humanoid robot communication system with the following components:

1. A sensor simulator node that publishes joint states and IMU data
2. A controller node that subscribes to sensor data and publishes commands
3. A service that computes simple joint positions based on a target pose

This exercise will help you understand how the different communication patterns work together in a humanoid system.

## Summary

In this lesson, we've covered:
- The fundamental ROS 2 concepts of nodes, topics, and services
- How to implement each component in Python
- The different use cases for each communication pattern in humanoid robotics
- Best practices for designing robust communication systems
- How these components work together in a complete robot system

These concepts form the foundation of all ROS 2-based robotic systems and are critical for implementing communication in humanoid robots. In the next lesson, we'll explore how to connect Python-based AI agents to these ROS 2 systems using rclpy.

## References and Further Reading

- ROS 2 Documentation: https://docs.ros.org/en/humble/
- Designing Message Types: http://wiki.ros.org/ROS/Patterns/Communication
- Quality of Service in ROS 2: https://docs.ros.org/en/humble/Concepts/About-Quality-of-Service-Settings.html
- Academic paper: "ROS 2: From Research to Production in Humanoid Robotics" (Fictional reference for educational purposes)

## APA Citations for This Lesson

When referencing ROS 2 concepts in academic work, cite the official documentation:

Open Robotics. (2023). *ROS 2 Documentation*. Retrieved from https://docs.ros.org/en/humble/

Additionally, if using this educational material, cite as:

Author, A. A. (2025). Lesson 1: ROS 2 Nodes, Topics, and Services. In *Physical AI & Humanoid Robotics Course*. Physical AI Educational Initiative.