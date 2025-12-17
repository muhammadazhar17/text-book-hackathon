---
sidebar_position: 3
---

# Lesson 2: Bridging Python Agents to ROS Controllers using rclpy

## Introduction

In this lesson, we explore the integration of Python-based AI agents with ROS 2 controllers using rclpy, the Python client library for ROS 2. This integration is critical for Physical AI applications, where sophisticated Python-based artificial intelligence algorithms need to control physical robots and respond to real-world sensory input.

Python's rich ecosystem for AI development—including TensorFlow, PyTorch, scikit-learn, and other machine learning libraries—makes it an ideal language for implementing the "brain" of intelligent robotic systems. The rclpy library enables seamless integration between these AI systems and the ROS 2-based "nervous system" of robots.

## Learning Objectives

By the end of this lesson, you will be able to:
1. Implement Python nodes that interface with ROS 2 using rclpy
2. Design and implement AI agents that process sensor data and generate robot commands
3. Create service clients and servers in Python for AI-robot interactions
4. Handle real-time constraints and performance considerations in Python-ROS integration
5. Implement error handling and fallback behaviors for AI-robot systems

## 1. Introduction to rclpy

### What is rclpy?

rclpy is the Python client library for ROS 2, providing Python developers with access to ROS 2 features including nodes, topics, services, parameters, and actions. It serves as the bridge between Python-based AI agents and the ROS 2 ecosystem.

### Why Python for AI in Robotics?

Python is the dominant language for AI and machine learning development due to:
- Rich ecosystem of AI libraries (TensorFlow, PyTorch, scikit-learn, etc.)
- Ease of experimentation and prototyping
- Strong community support and documentation
- Integration with scientific computing libraries (NumPy, SciPy, Pandas)

However, Python has limitations for real-time control due to the Global Interpreter Lock (GIL) and garbage collection. Therefore, the typical architecture involves Python for AI decision-making and other languages (like C++) for time-critical control loops.

## 2. Implementing AI Agents as ROS 2 Nodes

### Basic AI Agent Node Pattern

Here's a foundational pattern for implementing an AI agent as a ROS 2 node that processes sensor data and generates control commands:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import tensorflow as tf  # Example AI library

class VisionBasedNavigationAgent(Node):
    def __init__(self):
        super().__init__('vision_navigation_agent')
        
        # Publishers for robot commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers for sensor data
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )
        
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        # Timer for AI decision making (2 Hz)
        self.ai_timer = self.create_timer(0.5, self.ai_decision_callback)
        
        # Internal state
        self.latest_image = None
        self.latest_lidar = None
        self.ai_model = self.load_ai_model()
        
        self.get_logger().info('Vision-Based Navigation Agent initialized')

    def load_ai_model(self):
        """Load the AI model for navigation decisions"""
        try:
            # Load a pre-trained model for obstacle detection and navigation
            # In practice, this could be a CNN for image processing or other ML model
            model = tf.keras.models.load_model('path/to/navigation_model.h5')
            self.get_logger().info('AI model loaded successfully')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load AI model: {e}')
            return None

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS Image message to numpy array for AI processing
            # This is a simplified example - real implementation would depend on image encoding
            self.latest_image = self.ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def lidar_callback(self, msg):
        """Process incoming LIDAR data"""
        try:
            # Store LIDAR ranges for AI processing
            self.latest_lidar = msg.ranges
        except Exception as e:
            self.get_logger().error(f'Error processing LIDAR data: {e}')

    def ros_image_to_numpy(self, img_msg):
        """Convert ROS Image message to numpy array"""
        # Implementation depends on image encoding
        # Common encodings: 'rgb8', 'bgr8', 'mono8', etc.
        import cv2
        from cv_bridge import CvBridge
        
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        return cv_image

    def ai_decision_callback(self):
        """Main AI decision-making function"""
        if self.latest_image is not None and self.latest_lidar is not None:
            try:
                # Process sensor data with AI model
                command = self.make_navigation_decision(self.latest_image, self.latest_lidar)
                
                # Publish command to robot
                self.publish_command(command)
                
            except Exception as e:
                self.get_logger().error(f'AI decision error: {e}')
                # Publish stop command on error
                self.publish_stop_command()

    def make_navigation_decision(self, image, lidar_data):
        """Apply AI model to make navigation decision"""
        if self.ai_model is None:
            # Fallback behavior if AI model is not loaded
            return self.simple_avoidance_behavior(lidar_data)
        
        # Preprocess image for AI model
        processed_image = self.preprocess_image(image)
        
        # Run AI model to determine navigation command
        ai_output = self.ai_model.predict(np.expand_dims(processed_image, axis=0))
        
        # Convert AI output to Twist command
        linear_vel = ai_output[0][0]  # Example: first output is linear velocity
        angular_vel = ai_output[0][1]  # Example: second output is angular velocity
        
        command = Twist()
        command.linear.x = float(linear_vel)
        command.angular.z = float(angular_vel)
        
        return command

    def preprocess_image(self, image):
        """Preprocess image for AI model input"""
        # Resize image to model input size (example: 224x224)
        resized = cv2.resize(image, (224, 224))
        
        # Normalize pixel values (example: 0-1 range)
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized

    def simple_avoidance_behavior(self, lidar_data):
        """Simple reactive behavior if AI model is not available"""
        min_distance = min(lidar_data)
        
        cmd = Twist()
        if min_distance < 0.5:  # Too close to obstacle
            cmd.angular.z = 0.5  # Turn right
        else:
            cmd.linear.x = 0.5   # Move forward
            
        return cmd

    def publish_command(self, command):
        """Publish the navigation command to the robot"""
        self.cmd_vel_publisher.publish(command)
        self.get_logger().debug(f'Published command: linear.x={command.linear.x}, angular.z={command.angular.z}')

    def publish_stop_command(self):
        """Publish a stop command"""
        stop_cmd = Twist()
        self.cmd_vel_publisher.publish(stop_cmd)

def main(args=None):
    rclpy.init(args=args)
    
    agent = VisionBasedNavigationAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Node stopped by user')
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 3. Advanced AI Integration Patterns

### Multi-Modal Sensor Fusion with AI

Humanoid robots require processing multiple sensor modalities. Here's an example of fusing visual and LIDAR data in an AI agent:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np

class MultiModalAIAgent(Node):
    def __init__(self):
        super().__init__('multimodal_ai_agent')
        
        # Publishers
        self.behavior_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.state_pub = self.create_publisher(String, 'ai_state', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        
        # Timer for AI processing
        self.ai_timer = self.create_timer(0.2, self.ai_processing_callback)  # 5 Hz
        
        # Internal state
        self.sensors = {
            'image': None,
            'lidar': None,
            'imu': None
        }
        self.ai_model = self.initialize_multimodal_model()
        
        self.get_logger().info('Multi-Modal AI Agent initialized')

    def initialize_multimodal_model(self):
        """Initialize a multi-modal AI model"""
        # In practice, this could be a complex model combining:
        # - CNN for image processing
        # - RNN for temporal data from IMU
        # - Custom fusion layers
        self.get_logger().info('Multi-modal model initialized (placeholder)')
        return "multimodal_model"  # Placeholder

    def image_callback(self, msg):
        """Process camera image"""
        try:
            self.sensors['image'] = self.ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def lidar_callback(self, msg):
        """Process LIDAR data"""
        self.sensors['lidar'] = np.array(msg.ranges)

    def imu_callback(self, msg):
        """Process IMU data"""
        self.sensors['imu'] = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def ai_processing_callback(self):
        """Main AI processing using all sensor modalities"""
        # Check if we have all required sensor data
        if all(value is not None for value in self.sensors.values()):
            try:
                # Fuse sensor data for AI processing
                fused_data = self.fuse_sensor_data()
                
                # Make AI-driven decision
                decision = self.make_multimodal_decision(fused_data)
                
                # Execute decision
                self.execute_decision(decision)
                
            except Exception as e:
                self.get_logger().error(f'Multi-modal processing error: {e}')
                self.fallback_behavior()

    def fuse_sensor_data(self):
        """Fuse data from multiple sensors"""
        # This would implement more sophisticated fusion
        # like attention mechanisms, early fusion, or late fusion
        return {
            'image': self.sensors['image'],
            'lidar': self.sensors['lidar'],
            'imu': self.sensors['imu']
        }

    def make_multimodal_decision(self, fused_data):
        """Make decisions based on fused sensor data"""
        # Placeholder for complex multi-modal AI processing
        # In reality, this would run sophisticated models
        
        # Simple example: navigate based on image and avoid obstacles from LIDAR
        if self.is_path_clear(fused_data['lidar']) and self.detect_goal(fused_data['image']):
            return {'linear': 0.5, 'angular': 0.0}
        else:
            return self.avoid_obstacles(fused_data['lidar'])

    def is_path_clear(self, lidar_data):
        """Check if path ahead is clear of obstacles"""
        # Check forward sector (e.g., 30 degrees in front)
        forward_sector = lidar_data[330:30] if len(lidar_data) == 360 else lidar_data[len(lidar_data)//2-15:len(lidar_data)//2+15]
        return min(forward_sector) > 1.0  # Clear if no obstacles within 1m

    def detect_goal(self, image):
        """Detect goal/object of interest in image (placeholder)"""
        # This would use computer vision to detect goals
        return True  # Placeholder

    def avoid_obstacles(self, lidar_data):
        """Generate commands to avoid obstacles"""
        # Find direction with maximum clearance
        left_clearance = min(lidar_data[len(lidar_data)//4: len(lidar_data)//2])
        right_clearance = min(lidar_data[len(lidar_data)//2: 3*len(lidar_data)//4])
        
        if left_clearance > right_clearance:
            return {'linear': 0.1, 'angular': 0.3}  # Turn left
        else:
            return {'linear': 0.1, 'angular': -0.3}  # Turn right

    def execute_decision(self, decision):
        """Execute the AI's decision"""
        cmd = Twist()
        cmd.linear.x = decision['linear']
        cmd.angular.z = decision['angular']
        
        self.behavior_pub.publish(cmd)
        self.state_pub.publish(String(data=f"Executing: linear={decision['linear']}, angular={decision['angular']}"))

    def fallback_behavior(self):
        """Safe fallback behavior if AI processing fails"""
        stop_cmd = Twist()
        self.behavior_pub.publish(stop_cmd)
        self.state_pub.publish(String(data="ERROR: Using fallback behavior"))

def main(args=None):
    rclpy.init(args=args)
    
    agent = MultiModalAIAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Multi-modal agent stopped')
    finally:
        agent.destroy_node()
        rclpy.shutdown()
```

## 4. Service Integration with AI Agents

### AI as a Service Provider

AI agents can also provide services to other ROS 2 nodes:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger
from geometry_msgs.srv import PointStamped
from geometry_msgs.msg import Point
from std_msgs.msg import String
import numpy as np

class AIAgentService(Node):
    def __init__(self):
        super().__init__('ai_agent_service')
        
        # Service server for AI-based decisions
        self.navigation_service = self.create_service(
            PointStamped,
            'get_navigation_advice',
            self.navigation_advice_callback
        )
        
        # Service for requesting AI model status
        self.status_service = self.create_service(
            Trigger,
            'ai_agent_status',
            self.status_callback
        )
        
        # Publishers for visualization
        self.advice_pub = self.create_publisher(String, 'navigation_advice', 10)
        
        # Initialize AI model
        self.ai_model = self.load_model()
        self.model_loaded = self.ai_model is not None
        
        self.get_logger().info('AI Agent Service initialized')

    def load_model(self):
        """Load the AI model"""
        try:
            # Load model (placeholder)
            self.get_logger().info('AI model loaded successfully')
            return "loaded_model"  # Placeholder
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return None

    def navigation_advice_callback(self, request, response):
        """Provide navigation advice based on target location"""
        try:
            target_x = request.point.x
            target_y = request.point.y
            
            if not self.model_loaded:
                # Fallback algorithm
                advice = self.fallback_navigation_advice(target_x, target_y)
            else:
                # Use AI model for decision
                advice = self.ai_navigation_advice(target_x, target_y)
            
            response.success = True
            response.message = f"Navigation advice: {advice}"
            
            self.advice_pub.publish(String(data=response.message))
            
        except Exception as e:
            response.success = False
            response.message = f"Navigation advice error: {str(e)}"
            self.get_logger().error(f"Navigation service error: {e}")
        
        return response

    def ai_navigation_advice(self, target_x, target_y):
        """Use AI model to generate navigation advice"""
        # Placeholder for actual AI processing
        # This would process current robot state, environment data, and target
        # to generate navigation advice
        distance = np.sqrt(target_x**2 + target_y**2)
        
        if distance > 5.0:
            advice = "Long distance, consider intermediate waypoints"
        elif distance < 0.5:
            advice = "Very close to target, precise positioning needed"
        else:
            advice = "Standard navigation procedure"
            
        return advice

    def fallback_navigation_advice(self, target_x, target_y):
        """Fallback navigation algorithm"""
        distance = np.sqrt(target_x**2 + target_y**2)
        return f"Fallback: Go to ({target_x}, {target_y}), distance {distance:.2f}m"

    def status_callback(self, request, response):
        """Return AI agent status"""
        response.success = self.model_loaded
        response.message = "AI Agent is operational" if self.model_loaded else "AI Model not loaded"
        return response

def main(args=None):
    rclpy.init(args=args)
    
    service = AIAgentService()
    
    try:
        rclpy.spin(service)
    except KeyboardInterrupt:
        service.get_logger().info('AI Service stopped')
    finally:
        service.destroy_node()
        rclpy.shutdown()
```

## 5. Performance Considerations

### Threading and Async Processing

For computationally expensive AI tasks, using threading can prevent blocking ROS 2 callbacks:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import threading
import queue
import numpy as np

class ThreadingAIAgent(Node):
    def __init__(self):
        super().__init__('threading_ai_agent')
        
        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        
        # Queue for processing images
        self.image_queue = queue.Queue(maxsize=2)  # Limit queue size
        
        # Store latest command
        self.latest_command = Twist()
        self.command_lock = threading.Lock()
        
        # Start AI processing thread
        self.ai_thread = threading.Thread(target=self.ai_processing_loop)
        self.ai_thread.daemon = True
        self.ai_thread.start()
        
        self.get_logger().info('Threading AI Agent initialized')

    def image_callback(self, msg):
        """Non-blocking image callback"""
        try:
            # Add to queue if there's space
            if not self.image_queue.full():
                image_data = self.ros_image_to_numpy(msg)
                self.image_queue.put(image_data, block=False)
        except queue.Full:
            self.get_logger().warn('Image queue is full, dropping frame')
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def ai_processing_loop(self):
        """Run AI processing in separate thread"""
        while rclpy.ok():
            try:
                # Get image from queue (blocks until available)
                image_data = self.image_queue.get(timeout=1.0)
                
                # Process with AI model (this can be computationally expensive)
                command = self.process_with_ai_model(image_data)
                
                # Update latest command safely
                with self.command_lock:
                    self.latest_command = command
                    
            except queue.Empty:
                # No new images, continue loop
                continue
            except Exception as e:
                self.get_logger().error(f'AI processing error: {e}')

    def process_with_ai_model(self, image_data):
        """Process image with AI model (simulated)"""
        # Simulate AI processing delay
        import time
        time.sleep(0.1)  # Simulated processing time
        
        # Placeholder for actual AI model processing
        cmd = Twist()
        cmd.linear.x = 0.5  # Move forward
        cmd.angular.z = 0.1  # Slight turn
        return cmd

    def publish_current_command(self):
        """Publish the current command (call this from main thread)"""
        with self.command_lock:
            self.cmd_pub.publish(self.latest_command)

def main(args=None):
    rclpy.init(args=args)
    
    agent = ThreadingAIAgent()
    
    # Timer to periodically publish commands
    timer = agent.create_timer(0.1, agent.publish_current_command)  # 10 Hz
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()
```

## 6. Error Handling and Robustness

### Comprehensive Error Handling Pattern

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import traceback

class RobustAIAgent(Node):
    def __init__(self):
        super().__init__('robust_ai_agent')
        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        
        self.ai_model = self.initialize_model_with_fallbacks()
        self.fallback_active = False
        
        self.get_logger().info('Robust AI Agent initialized')

    def initialize_model_with_fallbacks(self):
        """Initialize AI model with fallback options"""
        try:
            # Try to load primary model
            model = self.load_primary_model()
            if model:
                self.get_logger().info('Primary AI model loaded')
                return model
        except Exception as e:
            self.get_logger().error(f'Primary model failed: {e}')
            
        try:
            # Try to load backup model
            model = self.load_backup_model()
            if model:
                self.get_logger().info('Backup AI model loaded')
                return model
        except Exception as e:
            self.get_logger().error(f'Backup model failed: {e}')
            
        # If all models fail, use simple reactive behavior
        self.get_logger().warn('All AI models failed, using reactive behavior')
        self.fallback_active = True
        return None

    def load_primary_model(self):
        """Load primary AI model"""
        # Implementation for loading primary model
        return "primary_model"  # Placeholder

    def load_backup_model(self):
        """Load backup AI model"""
        # Implementation for loading backup model
        return "backup_model"  # Placeholder

    def image_callback(self, msg):
        """Robust image callback with comprehensive error handling"""
        try:
            # Validate message
            if not self.is_valid_image_message(msg):
                self.get_logger().warn('Received invalid image message')
                return
            
            # Process image based on current system state
            if self.fallback_active:
                command = self.reactive_navigation_behavior(msg)
            else:
                command = self.ai_based_navigation(msg)
            
            # Validate command before publishing
            validated_command = self.validate_command(command)
            self.cmd_pub.publish(validated_command)
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
            self.get_logger().debug(traceback.format_exc())
            
            # Emergency stop on critical errors
            self.emergency_stop()

    def is_valid_image_message(self, msg):
        """Validate image message integrity"""
        # Check for basic validity
        if msg.height <= 0 or msg.width <= 0:
            return False
        if len(msg.data) != msg.height * msg.width * 3:  # Assuming RGB
            return False
        return True

    def validate_command(self, cmd):
        """Validate and sanitize robot command"""
        # Set limits on velocities
        cmd.linear.x = max(min(cmd.linear.x, 1.0), -1.0)  # Limit linear vel
        cmd.angular.z = max(min(cmd.angular.z, 1.0), -1.0)  # Limit angular vel
        
        # Ensure no NaN values
        if np.isnan(cmd.linear.x) or np.isnan(cmd.angular.z):
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        return cmd

    def emergency_stop(self):
        """Publish emergency stop command"""
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        self.get_logger().warn('Emergency stop executed')

def main(args=None):
    rclpy.init(args=args)
    
    agent = RobustAIAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Robust AI agent stopped')
    except Exception as e:
        agent.get_logger().error(f'Unexpected error: {e}')
    finally:
        agent.destroy_node()
        rclpy.shutdown()
```

## 7. Real-World Humanoid AI Integration Example

### Humanoid Behavior Selection Agent

Here's a comprehensive example of an AI agent for humanoid behavior selection:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Duration
import numpy as np
import json

class HumanoidBehaviorAgent(Node):
    def __init__(self):
        super().__init__('humanoid_behavior_agent')
        
        # Publishers for different robot subsystems
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.behavior_status_pub = self.create_publisher(String, 'behavior_status', 10)
        self.attention_target_pub = self.create_publisher(String, 'attention_target', 10)
        
        # Subscribers
        self.camera_sub = self.create_subscription(Image, '/camera/rgb/image_raw', self.camera_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        # Timer for behavior decisions (1 Hz)
        self.behavior_timer = self.create_timer(1.0, self.behavior_decision_callback)
        
        # Internal state
        self.sensory_data = {
            'person_detected': False,
            'person_distance': None,
            'obstacle_detected': False,
            'obstacle_distance': None,
            'environment_class': 'unknown'  # indoor, outdoor, corridor, etc.
        }
        
        # Current behavior state
        self.current_behavior = 'idle'
        self.behavior_params = {}
        
        self.get_logger().info('Humanoid Behavior Agent initialized')

    def camera_callback(self, msg):
        """Process camera data for person detection and environment classification"""
        try:
            image = self.ros_image_to_numpy(msg)
            
            # Simulate person detection (in practice, use CV/AI model)
            person_detected, distance = self.detect_person_in_image(image)
            
            # Update sensory data
            self.sensory_data['person_detected'] = person_detected
            if person_detected:
                self.sensory_data['person_distance'] = distance
                self.get_logger().info(f'Person detected at {distance:.2f}m')
            
            # Simulate environment classification
            env_class = self.classify_environment(image)
            self.sensory_data['environment_class'] = env_class
            
        except Exception as e:
            self.get_logger().error(f'Camera processing error: {e}')

    def lidar_callback(self, msg):
        """Process LIDAR data for obstacle detection"""
        try:
            # Check for obstacles in front of robot (e.g., 30-degree sector in front)
            front_sector = msg.ranges[len(msg.ranges)//2-15:len(msg.ranges)//2+15]
            min_distance = min([r for r in front_sector if 0 < r < float('inf')], default=float('inf'))
            
            self.sensory_data['obstacle_detected'] = min_distance < 1.0  # Obstacle within 1m
            self.sensory_data['obstacle_distance'] = min_distance if min_distance != float('inf') else None
            
        except Exception as e:
            self.get_logger().error(f'LIDAR processing error: {e}')

    def detect_person_in_image(self, image):
        """Detect person in image (simulated)"""
        # Simulate detection based on image features
        # In practice, use a pre-trained model like MobileNet-SSD or YOLO
        import random
        
        # For simulation, occasionally "detect" a person
        if random.random() < 0.3:  # 30% chance of detection
            distance = random.uniform(0.5, 5.0)  # Random distance
            return True, distance
        return False, None

    def classify_environment(self, image):
        """Classify environment type (simulated)"""
        # Simulate environment classification
        # In practice, use an image classification model
        import random
        
        env_types = ['indoor_office', 'indoor_corridor', 'outdoor', 'unknown']
        return random.choice(env_types)

    def behavior_decision_callback(self):
        """Main behavior decision-making logic"""
        try:
            # Determine appropriate behavior based on sensory data
            new_behavior, params = self.select_behavior()
            
            # Execute behavior transition if needed
            if new_behavior != self.current_behavior:
                self.transition_behavior(new_behavior, params)
            
            # Execute current behavior
            self.execute_current_behavior()
            
            # Publish behavior status
            status_msg = String()
            status_msg.data = f"Behavior: {self.current_behavior}, Params: {json.dumps(params)}"
            self.behavior_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Behavior decision error: {e}')
            self.fallback_behavior()

    def select_behavior(self):
        """Select behavior based on sensory data"""
        # Behavior priority system
        if self.sensory_data['obstacle_detected']:
            # Highest priority: avoid obstacles
            dist = self.sensory_data['obstacle_distance']
            if dist and dist < 0.5:
                return 'avoid_obstacle', {'direction': 'backward', 'distance': 0.3}
            else:
                return 'avoid_obstacle', {'direction': 'lateral', 'distance': 0.5}
        
        elif self.sensory_data['person_detected']:
            # High priority: interact with person
            dist = self.sensory_data['person_distance']
            if dist and dist < 2.0:
                return 'greet_person', {'person_distance': dist}
            else:
                return 'approach_person', {'person_distance': dist}
        
        elif self.sensory_data['environment_class'] == 'indoor_corridor':
            # Navigate in corridor
            return 'corridor_navigation', {}
        
        else:
            # Default: exploration or patrol
            return 'explore', {}

    def transition_behavior(self, new_behavior, params):
        """Handle transition to new behavior"""
        self.get_logger().info(f'Transitioning to behavior: {new_behavior}')
        
        # Execute any transition-specific logic
        if self.current_behavior == 'greeting' and new_behavior != 'greeting':
            # Stop greeting behavior
            self.reset_greeting_state()
        
        # Update current behavior
        self.current_behavior = new_behavior
        self.behavior_params = params

    def execute_current_behavior(self):
        """Execute the current behavior"""
        if self.current_behavior == 'avoid_obstacle':
            self.execute_avoid_obstacle()
        elif self.current_behavior == 'approach_person':
            self.execute_approach_person()
        elif self.current_behavior == 'greet_person':
            self.execute_greet_person()
        elif self.current_behavior == 'corridor_navigation':
            self.execute_corridor_navigation()
        elif self.current_behavior == 'explore':
            self.execute_exploration()
        else:
            # Default: stop
            self.execute_stop()

    def execute_avoid_obstacle(self):
        """Execute obstacle avoidance behavior"""
        cmd = Twist()
        if self.behavior_params.get('direction') == 'backward':
            cmd.linear.x = -0.3
        else:  # lateral
            cmd.angular.z = 0.5  # Turn right to avoid
        
        self.cmd_vel_pub.publish(cmd)

    def execute_approach_person(self):
        """Execute person approach behavior"""
        cmd = Twist()
        
        dist = self.behavior_params.get('person_distance', float('inf'))
        if dist != float('inf'):
            if dist > 2.0:
                cmd.linear.x = 0.3  # Move toward person
            elif dist > 1.5:
                cmd.linear.x = 0.1  # Slow down as approaching
            else:
                cmd.linear.x = 0.0  # Stop at appropriate distance
        
        self.cmd_vel_pub.publish(cmd)

    def execute_greet_person(self):
        """Execute greeting behavior"""
        cmd = Twist()
        # Keep position but look at person
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd)
        
        # Publish attention target
        attention_msg = String()
        attention_msg.data = "greeting_person"
        self.attention_target_pub.publish(attention_msg)

    def execute_corridor_navigation(self):
        """Execute corridor navigation"""
        cmd = Twist()
        cmd.linear.x = 0.4  # Move forward in corridor
        cmd.angular.z = 0.0  # Keep straight
        
        self.cmd_vel_pub.publish(cmd)

    def execute_exploration(self):
        """Execute exploration behavior"""
        # For simplicity, just move forward
        cmd = Twist()
        cmd.linear.x = 0.2
        cmd.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd)

    def execute_stop(self):
        """Execute stop behavior"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd)

    def fallback_behavior(self):
        """Execute safe fallback behavior"""
        self.get_logger().warn(f'Executing fallback behavior due to error. Current: {self.current_behavior}')
        
        # Stop robot and report error
        self.execute_stop()
        
        status_msg = String()
        status_msg.data = f"ERROR: Fallback active. Previous behavior: {self.current_behavior}"
        self.behavior_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    
    agent = HumanoidBehaviorAgent()
    
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        agent.get_logger().info('Humanoid Behavior Agent stopped')
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 8. Best Practices for Python-ROS Integration

### 1. Resource Management
- Properly initialize and shut down rclpy
- Use context managers or try/finally blocks for cleanup
- Monitor memory usage for long-running AI models

### 2. Performance Optimization
- Use NumPy for numerical computations
- Profile code to identify bottlenecks
- Consider using specialized libraries for AI operations
- Use appropriate threading strategies

### 3. Error Handling
- Implement comprehensive exception handling
- Use fallback behaviors for safety-critical systems
- Log errors appropriately for debugging
- Validate inputs and outputs

### 4. Testing
- Create unit tests for AI logic independent of ROS
- Use ROS testing tools for integration testing
- Simulate sensor failures and edge cases
- Test with realistic timing constraints

## 9. Practical Exercise

### Exercise: Implement an AI-Based Object Following Agent

Create an AI agent that uses camera data to follow a specific colored object (e.g., a red ball) using ROS 2 and rclpy:

1. Implement a ROS 2 node that subscribes to camera images
2. Use computer vision in Python to detect the colored object
3. Generate velocity commands to follow the object
4. Include error handling for when the object is not visible
5. Provide a service to change the target color dynamically

This exercise will demonstrate your understanding of connecting Python AI agents to ROS controllers.

## Summary

In this lesson, we've covered:
- The fundamentals of using rclpy to connect Python AI agents to ROS 2
- Implementation patterns for AI agents as ROS nodes
- Service integration for AI-robot interactions
- Performance optimization techniques including threading
- Error handling and robustness for AI-robot systems
- A comprehensive example of a humanoid behavior selection agent
- Best practices for Python-ROS integration

The integration of Python AI agents with ROS 2 controllers using rclpy is a critical component of modern robotics systems, enabling sophisticated artificial intelligence to control physical robots. Understanding these concepts is essential for developing intelligent humanoid robots.

## References and Further Reading

- ROS 2 with Python: https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html
- rclpy Documentation: https://docs.ros2.org/latest/api/rclpy/
- "Python Robotics: Fundamentals, Applications, and Systems" (Book)

## APA Citations for This Lesson

ROS 2 Working Group. (2023). *ROS 2 Documentation*. Open Robotics. Retrieved from https://docs.ros.org/en/humble/

When referencing this educational content:

Author, A. A. (2025). Lesson 2: Bridging Python Agents to ROS Controllers using rclpy. In *Physical AI & Humanoid Robotics Course*. Physical AI Educational Initiative.