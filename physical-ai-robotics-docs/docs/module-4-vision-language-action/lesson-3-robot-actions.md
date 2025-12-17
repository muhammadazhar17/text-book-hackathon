---
sidebar_position: 4
---

# Lesson 3: Robot Actions - From High-Level Goals to Physical Execution

## Learning Objectives

By the end of this lesson, you will be able to:

1. Design action execution architectures that translate high-level goals into low-level robot commands
2. Implement multi-step task planning and execution with error handling
3. Create robust action execution systems with recovery mechanisms
4. Implement adaptive execution that responds to environmental feedback
5. Design human-robot interaction protocols for action confirmation and monitoring

## Introduction

The action execution layer forms the crucial bridge between high-level planning (from LLMs and other planners) and the physical robot. This layer is responsible for translating abstract goals and plans into specific, executable movements and operations that the robot performs in the physical world. 

For humanoid robots, action execution is particularly complex because it must coordinate multiple degrees of freedom, handle diverse interaction modalities, and operate safely in human environments. The action execution system must be robust, adaptive, and capable of handling uncertainty in both the environment and the robot's own state.

Key challenges in action execution include:

- **Motion Planning**: Converting goals into specific joint trajectories
- **Control**: Executing movements with appropriate dynamics and forces
- **Perception Integration**: Using real-time perception to adapt actions
- **Error Handling**: Managing failures and uncertainties during execution
- **Human Safety**: Ensuring safe operation around humans

## Action Execution Architecture

The action execution system is typically organized as a hierarchical architecture, where high-level goals are progressively refined into low-level control commands:

### 1. Task Planning Level

At the highest level, the system decomposes complex goals into sequences of subtasks:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import time
import threading
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"

class Task:
    """Base class for tasks in the execution system"""
    def __init__(self, name: str, description: str, priority: int = 1):
        self.name = name
        self.description = description
        self.priority = priority  # Higher number = higher priority
        self.status = TaskStatus.PENDING
        self.created_at = time.time()
        self.start_time = None
        self.end_time = None
        self.subtasks: List['Task'] = []
        self.parameters: Dict[str, Any] = {}
        self.estimated_duration = 0.0  # seconds

    def execute(self) -> bool:
        """Execute the task and return success status"""
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        
        try:
            result = self._execute_impl()
            self.status = TaskStatus.SUCCESS if result else TaskStatus.FAILURE
            self.end_time = time.time()
            return result
        except Exception as e:
            print(f"Task {self.name} failed with error: {e}")
            self.status = TaskStatus.FAILURE
            self.end_time = time.time()
            return False

    @abstractmethod
    def _execute_impl(self) -> bool:
        """Implementation of the specific task execution"""
        pass

    def add_subtask(self, subtask: 'Task'):
        """Add a subtask to this task"""
        self.subtasks.append(subtask)

    def get_duration(self) -> float:
        """Get the execution duration"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

class NavigationTask(Task):
    """Task for navigating to a specific location"""
    def __init__(self, destination: str, approach_distance: float = 0.5, speed: float = 0.3):
        super().__init__(f"navigate_to_{destination}", f"Navigate to {destination}")
        self.destination = destination
        self.approach_distance = approach_distance
        self.speed = speed
        self.estimated_duration = 60.0  # Estimated average duration in seconds

    def _execute_impl(self) -> bool:
        """Execute navigation task"""
        print(f"Navigating to {self.destination}")
        
        # In a real implementation, this would interface with the navigation system
        # For simulation, we'll just sleep for a while
        time.sleep(2)  # Simulate navigation time
        
        # In a real system, we would check if navigation succeeded
        # success = self.robot_interface.navigate_to(self.destination)
        # return success
        
        # Simulated success
        return True

class ManipulationTask(Task):
    """Task for manipulating objects"""
    def __init__(self, action: str, object_name: str, location: str = None):
        super().__init__(f"{action}_{object_name}", f"{action.title()} {object_name}")
        self.action = action  # 'pick', 'place', 'grasp', 'release'
        self.object_name = object_name
        self.location = location
        self.estimated_duration = 15.0

    def _execute_impl(self) -> bool:
        """Execute manipulation task"""
        print(f"Performing manipulation: {self.action} {self.object_name}")
        
        # In a real implementation, this would interface with the manipulation system
        time.sleep(1.5)  # Simulate manipulation time
        
        # In a real system, we would check if manipulation succeeded
        # success = self.robot_interface.manipulate_object(self.action, self.object_name, self.location)
        # return success
        
        # Simulated success
        return True

class CommunicationTask(Task):
    """Task for robot communication"""
    def __init__(self, message: str, modality: str = "speech"):
        super().__init__(f"communicate_{hash(message)}", f"Communicate: {message[:30]}...")
        self.message = message
        self.modality = modality  # 'speech', 'gesture', 'display'
        self.estimated_duration = len(message.split()) * 0.2  # Rough estimate based on word count

    def _execute_impl(self) -> bool:
        """Execute communication task"""
        print(f"Communicating: {self.message}")
        
        # In a real implementation, this would interface with the communication system
        time.sleep(0.5)
        
        # Simulated success
        return True
```

### 2. Motion Planning Level

The motion planning level converts task goals into specific movement trajectories:

```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class JointState:
    """Represents robot joint states"""
    positions: List[float]
    velocities: List[float]
    efforts: List[float]

@dataclass
class CartesianPose:
    """Represents pose in Cartesian space"""
    position: Tuple[float, float, float]  # x, y, z
    orientation: Tuple[float, float, float, float]  # x, y, z, w (quaternion)

class MotionPlanner:
    """Motion planning component of the action execution system"""
    
    def __init__(self):
        self.robot_urdf_path = "path/to/robot.urdf"  # Path to robot model
        self.joint_limits = self.load_joint_limits()
        self.workspace_bounds = {
            'min': (-1, -1, 0),
            'max': (1, 1, 2)  # meters
        }
        
    def load_joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """Load joint limits from robot description"""
        # In real implementation, would load from URDF/SRDF
        return {
            'joint_1': (-2.0, 2.0),
            'joint_2': (-1.5, 1.5),
            'joint_3': (-3.0, 3.0),
            'joint_4': (-2.0, 2.0),
            'joint_5': (-2.5, 2.5),
            'joint_6': (-3.0, 3.0)
        }
    
    def plan_navigation_trajectory(self, start_pose: CartesianPose, goal_pose: CartesianPose, 
                                   environment_map: Dict[str, Any] = None) -> List[CartesianPose]:
        """Plan a navigation trajectory from start to goal"""
        # This would typically use algorithms like A*, RRT, or navigation stack
        # For this example, we'll create a simple linear trajectory
        
        # Calculate number of waypoints based on distance
        dx = goal_pose.position[0] - start_pose.position[0]
        dy = goal_pose.position[1] - start_pose.position[1]
        dz = goal_pose.position[2] - start_pose.position[2]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        num_waypoints = max(int(distance * 10), 10)  # 10 waypoints per meter
        trajectory = []
        
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            waypoint = CartesianPose(
                position=(
                    start_pose.position[0] + t * dx,
                    start_pose.position[1] + t * dy,
                    start_pose.position[2] + t * dz
                ),
                orientation=self.interpolate_orientation(start_pose.orientation, goal_pose.orientation, t)
            )
            trajectory.append(waypoint)
        
        return trajectory
    
    def interpolate_orientation(self, start_quat: Tuple[float, float, float, float], 
                                end_quat: Tuple[float, float, float, float], 
                                t: float) -> Tuple[float, float, float, float]:
        """Linearly interpolate between two orientations"""
        # Use SLERP (Spherical Linear Interpolation) for proper quaternion interpolation
        # For simplicity, we'll use linear interpolation here
        interpolated = [
            start_quat[i] + t * (end_quat[i] - start_quat[i]) 
            for i in range(4)
        ]
        # Normalize quaternion
        norm = np.linalg.norm(interpolated)
        return tuple(val/norm for val in interpolated)
    
    def plan_manipulation_trajectory(self, 
                                   start_pose: CartesianPose,
                                   goal_pose: CartesianPose,
                                   object_pose: CartesianPose = None,
                                   approach_distance: float = 0.1) -> List[CartesianPose]:
        """Plan a manipulation trajectory with approach and retreat"""
        trajectory = []
        
        # Approach trajectory - move to approach point before grasping
        approach_pose = self.calculate_approach_pose(goal_pose, approach_distance)
        approach_traj = self.plan_cartesian_trajectory(start_pose, approach_pose)
        trajectory.extend(approach_traj)
        
        # Grasp trajectory - move from approach to grasp position
        grasp_traj = self.plan_cartesian_trajectory(approach_pose, goal_pose)
        trajectory.extend(grasp_traj)
        
        # If object is being moved, plan trajectory to destination
        if object_pose:
            # Retreat from grasp position
            retreat_pose = self.calculate_approach_pose(goal_pose, approach_distance)
            # Go to destination
            destination_traj = self.plan_cartesian_trajectory(retreat_pose, object_pose)
            trajectory.extend(destination_traj)
        
        # Calculate joint trajectory from Cartesian trajectory
        return trajectory
    
    def calculate_approach_pose(self, target_pose: CartesianPose, distance: float) -> CartesianPose:
        """Calculate approach pose at a distance from target"""
        # Move along the approach vector (e.g., along the z-axis of the gripper)
        # For simplicity, we'll just translate along the z-axis
        approach_pos = (
            target_pose.position[0],
            target_pose.position[1],
            target_pose.position[2] + distance
        )
        return CartesianPose(position=approach_pos, orientation=target_pose.orientation)
    
    def plan_cartesian_trajectory(self, start: CartesianPose, end: CartesianPose, 
                                  num_waypoints: int = 20) -> List[CartesianPose]:
        """Plan a simple Cartesian trajectory from start to end"""
        trajectory = []
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            waypoint_pos = [
                start.position[j] + t * (end.position[j] - start.position[j])
                for j in range(3)
            ]
            
            waypoint_orient = self.interpolate_orientation(start.orientation, end.orientation, t)
            
            trajectory.append(CartesianPose(
                position=tuple(waypoint_pos),
                orientation=waypoint_orient
            ))
        
        return trajectory

class JointTrajectoryExecutor:
    """Convert Cartesian trajectories to joint-space trajectories and execute"""
    
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.motion_planner = MotionPlanner()
        self.inverse_kinematics_solver = self.initialize_ik_solver()
    
    def initialize_ik_solver(self):
        """Initialize inverse kinematics solver"""
        # In real implementation, this could be PyKDL, KDL, or other IK solver
        # For this example, we'll simulate an IK solver
        class MockIKSolver:
            def solve_ik(self, pose: CartesianPose) -> Optional[JointState]:
                # In a real implementation, this would solve inverse kinematics
                # For simulation, return a mock joint state
                return JointState(
                    positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    velocities=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    efforts=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                )
        return MockIKSolver()
    
    def execute_cartesian_trajectory(self, cartesian_trajectory: List[CartesianPose], 
                                   velocity_scale: float = 1.0) -> bool:
        """Execute a Cartesian trajectory by converting to joint space"""
        joint_trajectory = []
        
        # Convert each Cartesian pose to joint state
        for pose in cartesian_trajectory:
            joint_state = self.inverse_kinematics_solver.solve_ik(pose)
            if joint_state is None:
                print("IK solution not found for pose, trajectory execution failed")
                return False
            joint_trajectory.append(joint_state)
        
        # Execute the joint trajectory
        return self.execute_joint_trajectory(joint_trajectory, velocity_scale)
    
    def execute_joint_trajectory(self, joint_trajectory: List[JointState], 
                               velocity_scale: float = 1.0) -> bool:
        """Execute a joint-space trajectory"""
        try:
            # In a real implementation, this would send trajectory to robot controller
            for joint_state in joint_trajectory:
                # Simulate sending command to robot controller
                time.sleep(0.05)  # Simulate time to execute each waypoint
                
                # In real implementation:
                # self.robot_interface.send_joint_trajectory(joint_state, velocity_scale)
                
            return True
        except Exception as e:
            print(f"Error executing joint trajectory: {e}")
            return False
```

### 3. Control Level

The control level handles low-level motor control and feedback:

```python
class RobotController:
    """Low-level robot controller interface"""
    
    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.joint_controllers = {}
        self.control_frequency = 100  # Hz
        self.is_active = False
        
    def initialize_controllers(self):
        """Initialize all joint controllers"""
        # In real implementation, would initialize hardware interfaces
        print("Initializing robot controllers...")
        self.is_active = True
        
    def send_joint_position(self, joint_name: str, position: float, duration: float = 0.1):
        """Send position command to a specific joint"""
        if not self.is_active:
            raise RuntimeError("Controller not active")
        
        # In a real implementation, this would interface with hardware
        print(f"Sending position command to {joint_name}: {position} rad")
        # self.robot_interface.send_position_command(joint_name, position, duration)
    
    def send_joint_trajectory(self, joint_names: List[str], trajectory_points: List[List[float]], 
                            time_points: List[float]):
        """Send a complete trajectory to multiple joints"""
        if not self.is_active:
            raise RuntimeError("Controller not active")
        
        # In a real implementation, this would send trajectory to robot
        print(f"Sending trajectory for joints: {joint_names}")
        # self.robot_interface.send_joint_trajectory(joint_names, trajectory_points, time_points)
    
    def get_joint_state(self, joint_name: str) -> JointState:
        """Get current state of a specific joint"""
        if not self.is_active:
            raise RuntimeError("Controller not active")
        
        # In a real implementation, this would query hardware
        # For simulation, return mock data
        return JointState(
            positions=[0.0],
            velocities=[0.0],
            efforts=[0.0]
        )
    
    def get_all_joint_states(self) -> Dict[str, JointState]:
        """Get current state of all joints"""
        if not self.is_active:
            raise RuntimeError("Controller not active")
        
        # In a real implementation, this would query hardware
        # For simulation, return mock data
        return {
            "joint_1": JointState([0.1], [0.0], [0.0]),
            "joint_2": JointState([0.2], [0.0], [0.0]),
            "joint_3": JointState([0.3], [0.0], [0.0])
        }
    
    def stop_all_motion(self):
        """Stop all robot motion immediately"""
        print("Stopping all robot motion")
        # self.robot_interface.stop_all_motors()
        self.is_active = False

class SafetyController:
    """Safety system for action execution"""
    
    def __init__(self, robot_controller: RobotController):
        self.robot_controller = robot_controller
        self.emergency_stop = False
        self.safety_limits = self.define_safety_limits()
        self.monitoring_thread = None
        self.is_monitoring = False
        
    def define_safety_limits(self) -> Dict[str, Any]:
        """Define safety limits for robot operation"""
        return {
            'joint_limits': {
                'position': {'min': -3.0, 'max': 3.0},
                'velocity': {'max': 2.0},
                'effort': {'max': 50.0}
            },
            'workspace_limits': {
                'min': (-2.0, -2.0, 0.0),  # x, y, z
                'max': (2.0, 2.0, 2.0)
            },
            'force_limits': {
                'gripper': {'max': 100.0},  # Newtons
                'wrist': {'max': 50.0}
            },
            'human_proximity': {
                'minimum_distance': 0.5  # meters
            }
        }
    
    def start_monitoring(self):
        """Start safety monitoring thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            print("Safety monitoring started")
    
    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("Safety monitoring stopped")
    
    def monitoring_loop(self):
        """Continuous monitoring loop for safety"""
        while self.is_monitoring:
            try:
                # Check joint limits
                if self.check_joint_limits():
                    self.trigger_safety_stop("Joint limit exceeded")
                    continue
                
                # Check workspace limits
                if self.check_workspace_limits():
                    self.trigger_safety_stop("Workspace limit exceeded")
                    continue
                
                # Check force limits
                if self.check_force_limits():
                    self.trigger_safety_stop("Force limit exceeded")
                    continue
                
                # Check for humans in proximity
                if self.check_human_proximity():
                    self.trigger_safety_stop("Human too close")
                    continue
                
                time.sleep(0.01)  # Monitor at 100 Hz
                
            except Exception as e:
                print(f"Error in safety monitoring: {e}")
                time.sleep(0.1)
    
    def check_joint_limits(self) -> bool:
        """Check if any joints are outside safe limits"""
        # This would check current joint states against limits
        # Return True if unsafe, False if safe
        return False  # Simplified for example
    
    def check_workspace_limits(self) -> bool:
        """Check if end-effector is outside workspace limits"""
        # This would check current pose against workspace bounds
        # Return True if unsafe, False if safe
        return False  # Simplified for example
    
    def check_force_limits(self) -> bool:
        """Check if forces are exceeding safe limits"""
        # This would check force/torque sensors
        # Return True if unsafe, False if safe
        return False  # Simplified for example
    
    def check_human_proximity(self) -> bool:
        """Check if humans are too close to robot"""
        # This would check perception system for human detection
        # Return True if unsafe, False if safe
        return False  # Simplified for example
    
    def trigger_safety_stop(self, reason: str):
        """Trigger emergency stop with reason"""
        print(f"SAFETY STOP TRIGGERED: {reason}")
        self.emergency_stop = True
        self.robot_controller.stop_all_motion()
        
        # In a real implementation, might also:
        # - Log the incident
        # - Notify operators
        # - Enter safe state
```

## Multi-Step Task Execution

Complex tasks require coordination of multiple actions with proper sequencing and error handling:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue

class MultiStepTaskExecutor:
    """Executor for complex multi-step tasks"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)  # For parallel execution of independent tasks
        self.active_tasks = []
        self.safety_controller = None
        self.perception_system = None
        self.navigation_system = None
        self.manipulation_system = None
        
    def set_systems(self, perception=None, navigation=None, manipulation=None, safety_controller=None):
        """Set references to other systems"""
        self.perception_system = perception
        self.navigation_system = navigation
        self.manipulation_system = manipulation
        self.safety_controller = safety_controller
        
    async def execute_complex_task(self, tasks: List[Task], parallelizable: bool = False) -> bool:
        """Execute a list of tasks, either sequentially or in parallel if possible"""
        if not tasks:
            return True
            
        if parallelizable:
            return await self._execute_parallel_tasks(tasks)
        else:
            return await self._execute_sequential_tasks(tasks)
    
    async def _execute_sequential_tasks(self, tasks: List[Task]) -> bool:
        """Execute tasks in sequential order"""
        for task in tasks:
            print(f"Executing task: {task.name}")
            
            # Check safety before each task
            if self.safety_controller and self.safety_controller.emergency_stop:
                print("Emergency stop active, cancelling task execution")
                return False
            
            success = await asyncio.get_event_loop().run_in_executor(
                self.executor, task.execute
            )
            
            if not success:
                print(f"Task {task.name} failed, stopping execution")
                return False
                
            print(f"Task {task.name} completed successfully")
        
        return True
    
    async def _execute_parallel_tasks(self, tasks: List[Task]) -> bool:
        """Execute tasks in parallel where possible"""
        # For true parallel execution, we'd need to identify which tasks can run in parallel
        # For now, we'll run them all but using async for better resource utilization
        tasks_to_execute = [asyncio.get_event_loop().run_in_executor(self.executor, task.execute) for task in tasks]
        
        results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
        
        # Check results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {tasks[i].name} raised exception: {result}")
                return False
            elif not result:
                print(f"Task {tasks[i].name} failed")
                return False
        
        return True
    
    async def execute_fetch_task(self, item: str, destination: str) -> bool:
        """Execute a fetch task: locate, navigate, grasp, return"""
        # Step 1: Locate the item
        if not await self.locate_item(item):
            print(f"Could not locate {item}")
            return False
        
        # Step 2: Navigate to the item
        navigate_task = NavigationTask(f"near_{item}")
        if not await asyncio.get_event_loop().run_in_executor(self.executor, navigate_task.execute):
            print(f"Failed to navigate to {item}")
            return False
        
        # Step 3: Grasp the item
        grasp_task = ManipulationTask("grasp", item)
        if not await asyncio.get_event_loop().run_in_executor(self.executor, grasp_task.execute):
            print(f"Failed to grasp {item}")
            return False
        
        # Step 4: Navigate to destination
        navigate_destination_task = NavigationTask(destination)
        if not await asyncio.get_event_loop().run_in_executor(self.executor, navigate_destination_task.execute):
            print(f"Failed to navigate to {destination}")
            return False
        
        # Step 5: Release the item
        release_task = ManipulationTask("place", item, destination)
        if not await asyncio.get_event_loop().run_in_executor(self.executor, release_task.execute):
            print(f"Failed to place {item} at {destination}")
            return False
        
        print(f"Successfully fetched {item} and placed at {destination}")
        return True
    
    async def locate_item(self, item_name: str) -> bool:
        """Locate an item in the environment using perception system"""
        try:
            if self.perception_system:
                # Use perception system to locate item
                # result = self.perception_system.find_object(item_name)
                # return result is not None
                print(f"Locating {item_name}...")
                time.sleep(1)  # Simulate search time
                return True  # Simulate finding the item
            else:
                print("Perception system not available")
                return False
        except Exception as e:
            print(f"Error locating item: {e}")
            return False

class AdaptiveTaskExecutor(MultiStepTaskExecutor):
    """Task executor with adaptation capabilities based on execution feedback"""
    
    def __init__(self):
        super().__init__()
        self.execution_history = []
        self.adaptation_rules = self.define_adaptation_rules()
        
    def define_adaptation_rules(self) -> Dict[str, Any]:
        """Define rules for adapting execution based on feedback"""
        return {
            'navigation_failure': {
                'retry_with_different_path': True,
                'update_map': True,
                'try_alternative_location': True
            },
            'grasp_failure': {
                'try_different_grasp': True,
                'adjust_approach': True,
                'retry_with_compensation': True
            },
            'perception_failure': {
                'reposition_robot': True,
                'change_sensor_params': True,
                'request_human_assistance': False
            }
        }
    
    async def adaptive_execute_task(self, task: Task, max_attempts: int = 3) -> bool:
        """Execute a task with adaptive recovery"""
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1} of {max_attempts} for task: {task.name}")
            
            success = await asyncio.get_event_loop().run_in_executor(self.executor, task.execute)
            
            if success:
                print(f"Task {task.name} succeeded on attempt {attempt + 1}")
                self.record_execution(task, attempt + 1, True)
                return True
            else:
                print(f"Task {task.name} failed on attempt {attempt + 1}")
                
                if attempt < max_attempts - 1:  # Not the last attempt
                    # Apply adaptation strategy
                    adapted_task = await self.adapt_task_for_retry(task, attempt)
                    if adapted_task:
                        task = adapted_task
                    else:
                        # If no adaptation possible, continue with original task
                        pass
                else:
                    print(f"Task {task.name} failed after {max_attempts} attempts")
                    self.record_execution(task, attempt + 1, False)
                    return False
        
        return False
    
    def record_execution(self, task: Task, attempt: int, success: bool):
        """Record execution result for learning"""
        record = {
            'task_name': task.name,
            'attempt': attempt,
            'success': success,
            'timestamp': time.time(),
            'duration': task.get_duration()
        }
        self.execution_history.append(record)
        
        # Trim history to last 100 executions to prevent memory issues
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    async def adapt_task_for_retry(self, original_task: Task, attempt: int) -> Optional[Task]:
        """Adapt task for retry based on execution history"""
        # For different task types, apply different adaptations
        if isinstance(original_task, NavigationTask):
            return await self.adapt_navigation_task(original_task, attempt)
        elif isinstance(original_task, ManipulationTask):
            return await self.adapt_manipulation_task(original_task, attempt)
        else:
            # For other task types, try with slightly different parameters
            return self.modify_task_parameters(original_task, attempt)
    
    async def adapt_navigation_task(self, task: NavigationTask, attempt: int) -> Optional[Task]:
        """Adapt navigation task for retry"""
        # Increase approach distance on retries to avoid obstacles
        new_approach_distance = task.approach_distance * (1 + attempt * 0.2)
        new_speed = max(task.speed * 0.8, 0.1)  # Slow down with each attempt
        
        adapted_task = NavigationTask(
            destination=task.destination,
            approach_distance=new_approach_distance,
            speed=new_speed
        )
        
        print(f"Adapted navigation task: new approach distance {new_approach_distance}, speed {new_speed}")
        return adapted_task
    
    async def adapt_manipulation_task(self, task: ManipulationTask, attempt: int) -> Optional[Task]:
        """Adapt manipulation task for retry"""
        # Add small offsets to approach position to try different angles
        offset = attempt * 0.05  # 5cm offset per attempt
        
        # Create a modified version of the task with adaptation
        adapted_task = ManipulationTask(
            action=task.action,
            object_name=task.object_name,
            location=task.location
        )
        
        # Store adaptation info in parameters
        adapted_task.parameters = {
            'approach_offset': offset,
            'attempt_number': attempt
        }
        
        print(f"Adapted manipulation task with offset: {offset}")
        return adapted_task
    
    def modify_task_parameters(self, task: Task, attempt: int) -> Task:
        """Apply general parameter modification for retries"""
        # Create a copy of the task with modified parameters
        adapted_task = type(task)(**task.__dict__)
        
        # Add attempt-specific parameters
        adapted_task.parameters['retry_attempt'] = attempt
        
        return adapted_task
```

## Error Handling and Recovery

Robust action execution requires sophisticated error handling and recovery mechanisms:

```python
class RecoveryStrategy(ABC):
    """Base class for recovery strategies"""
    
    @abstractmethod
    def apply(self, failed_task: Task, error_context: Dict[str, Any]) -> bool:
        """Apply recovery strategy and return if it succeeded"""
        pass

class RetryWithBackoff(RecoveryStrategy):
    """Retry strategy with exponential backoff"""
    
    def apply(self, failed_task: Task, error_context: Dict[str, Any]) -> bool:
        max_attempts = error_context.get('max_attempts', 3)
        current_attempt = error_context.get('current_attempt', 1)
        
        if current_attempt >= max_attempts:
            return False  # Already at max attempts
        
        # Calculate delay with exponential backoff
        base_delay = error_context.get('base_delay', 0.5)
        delay = base_delay * (2 ** (current_attempt - 1))  # Exponential backoff
        
        print(f"Waiting {delay}s before retry (attempt {current_attempt + 1})")
        time.sleep(delay)
        
        # Try to execute the task again
        return failed_task.execute()

class AlternativeAction(RecoveryStrategy):
    """Use an alternative action to achieve the same goal"""
    
    def apply(self, failed_task: Task, error_context: Dict[str, Any]) -> bool:
        alternative = error_context.get('alternative_action')
        if not alternative:
            return False
            
        print(f"Trying alternative action: {alternative}")
        return alternative.execute()

class TaskDelegation(RecoveryStrategy):
    """Delegate the task to another agent or human"""
    
    def apply(self, failed_task: Task, error_context: Dict[str, Any]) -> bool:
        print(f"Delegating task {failed_task.name} to alternative agent")
        # In a real implementation, would delegate to human operator or other robot
        return False  # For now, just report the delegation

class ErrorHandlingSystem:
    """System for managing errors and applying recovery strategies"""
    
    def __init__(self):
        self.recovery_strategies = {
            'retry_backoff': RetryWithBackoff(),
            'alternative_action': AlternativeAction(),
            'delegate': TaskDelegation()
        }
        self.error_log = []
        
    def handle_error(self, failed_task: Task, exception: Exception) -> bool:
        """Handle an error in task execution"""
        print(f"Handling error in task {failed_task.name}: {exception}")
        
        # Log the error
        error_record = {
            'task_name': failed_task.name,
            'error': str(exception),
            'timestamp': time.time(),
            'task_state': failed_task.status.value
        }
        self.error_log.append(error_record)
        
        # Determine appropriate recovery strategy
        strategy_name = self.select_recovery_strategy(failed_task, exception)
        if strategy_name:
            strategy = self.recovery_strategies.get(strategy_name)
            if strategy:
                error_context = self.create_error_context(failed_task, exception)
                return strategy.apply(failed_task, error_context)
        
        # No recovery possible
        return False
    
    def select_recovery_strategy(self, failed_task: Task, exception: Exception) -> Optional[str]:
        """Select appropriate recovery strategy based on error type"""
        error_msg = str(exception).lower()
        
        # Match error type to recovery strategy
        if "timeout" in error_msg or "connection" in error_msg:
            return "retry_backoff"
        elif "navigation" in error_msg or "obstacle" in error_msg:
            return "alternative_action"
        elif "grasp" in error_msg or "manipulation" in error_msg:
            return "alternative_action"
        elif "critical" in error_msg or "safety" in error_msg:
            return "delegate"
        else:
            # Default to retry for most errors
            return "retry_backoff"
    
    def create_error_context(self, failed_task: Task, exception: Exception) -> Dict[str, Any]:
        """Create context for error recovery"""
        return {
            'task': failed_task,
            'exception': exception,
            'max_attempts': 3,
            'current_attempt': 1,
            'base_delay': 0.5
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors"""
        if not self.error_log:
            return {'total_errors': 0}
        
        error_types = {}
        for record in self.error_log:
            error_type = record['error']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'recent_errors': self.error_log[-10:]  # Last 10 errors
        }

class RobustActionExecutor:
    """Action executor with comprehensive error handling and recovery"""
    
    def __init__(self):
        self.task_executor = AdaptiveTaskExecutor()
        self.error_handler = ErrorHandlingSystem()
        self.max_recovery_attempts = 3
        
    async def execute_robustly(self, task: Task, max_execution_attempts: int = 3) -> bool:
        """Execute a task with comprehensive error handling and recovery"""
        execution_attempts = 0
        
        while execution_attempts < max_execution_attempts:
            execution_attempts += 1
            print(f"Execution attempt {execution_attempts}/{max_execution_attempts}")
            
            try:
                # Execute the task
                success = await self.task_executor.adaptive_execute_task(
                    task, 
                    max_attempts=2  # Use adaptive execution internally
                )
                
                if success:
                    print(f"Task {task.name} completed successfully after {execution_attempts} attempt(s)")
                    return True
                    
            except Exception as primary_exception:
                print(f"Primary execution failed with: {primary_exception}")
                
                # Try to recover from the error
                recovery_success = await self.attempt_recovery(task, primary_exception)
                
                if not recovery_success and execution_attempts >= max_execution_attempts:
                    print(f"Task {task.name} failed after all recovery attempts")
                    return False
                
                # If recovery failed but we have more execution attempts, continue
                continue
        
        return False
    
    async def attempt_recovery(self, failed_task: Task, exception: Exception) -> bool:
        """Attempt to recover from task failure"""
        print(f"Attempting recovery for task {failed_task.name} after error: {exception}")
        
        recovery_attempts = 0
        while recovery_attempts < self.max_recovery_attempts:
            recovery_attempts += 1
            
            try:
                recovery_success = self.error_handler.handle_error(failed_task, exception)
                if recovery_success:
                    print(f"Recovery succeeded after {recovery_attempts} attempt(s)")
                    return True
                else:
                    print(f"Recovery attempt {recovery_attempts} failed")
            except Exception as recovery_exception:
                print(f"Recovery attempt {recovery_attempts} raised exception: {recovery_exception}")
            
            # Brief delay between recovery attempts
            time.sleep(0.1)
        
        print("All recovery attempts failed")
        return False
    
    def execute_complex_operation(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute a complex operation composed of multiple steps with error handling"""
        for op in operations:
            task_type = op['type']
            params = op.get('params', {})
            
            # Create task based on type
            if task_type == 'navigate':
                task = NavigationTask(params['destination'])
            elif task_type == 'grasp':
                task = ManipulationTask('grasp', params['object'])
            elif task_type == 'communicate':
                task = CommunicationTask(params['message'])
            else:
                print(f"Unknown task type: {task_type}")
                return False
            
            # Execute task robustly
            success = asyncio.run(self.execute_robustly(task))
            if not success:
                print(f"Operation failed at step: {task_type} with params {params}")
                return False
        
        print("Complex operation completed successfully")
        return True
```

## Human-Robot Interaction for Action Monitoring

Effective action execution systems include mechanisms for human oversight and interaction:

```python
from typing import Callable, Awaitable
import json

class HumanInteractionManager:
    """Manages human-robot interaction for action execution"""
    
    def __init__(self):
        self.human_confirmation_required = True
        self.notification_callbacks = []
        self.confirmation_timeout = 30  # seconds
        self.active_processes = {}
        
    def add_notification_callback(self, callback: Callable[[str, str], None]):
        """Add callback to be notified of action events"""
        self.notification_callbacks.append(callback)
    
    def notify_human(self, event_type: str, message: str):
        """Notify humans of action events"""
        print(f"[HRI] {event_type}: {message}")
        
        for callback in self.notification_callbacks:
            try:
                callback(event_type, message)
            except Exception as e:
                print(f"Error in notification callback: {e}")
    
    def request_human_confirmation(self, action_description: str) -> bool:
        """Request human confirmation before executing an action"""
        if not self.human_confirmation_required:
            return True  # Skip confirmation if not required
            
        self.notify_human("ACTION_REQUEST", f"Confirm action: {action_description}")
        
        # In a real implementation, this would wait for human input
        # For simulation, we'll return True
        print(f"Waiting for confirmation for: {action_description}")
        time.sleep(1)  # Simulate waiting for human input
        return True  # Simulated confirmation
    
    def report_action_status(self, task: Task, status: str, details: str = ""):
        """Report action status to human operators"""
        message = f"{task.name} - {status}"
        if details:
            message += f": {details}"
            
        self.notify_human("ACTION_STATUS", message)
    
    def enable_human_override(self, process_id: str, task: Task):
        """Enable human override for a running process"""
        self.active_processes[process_id] = {
            'task': task,
            'start_time': time.time(),
            'can_override': True
        }
    
    def handle_human_override(self, process_id: str, override_action: str):
        """Handle human override request"""
        if process_id not in self.active_processes:
            return False
            
        process = self.active_processes[process_id]
        if not process['can_override']:
            return False
            
        # Handle the override action
        if override_action == "pause":
            print(f"Pausing process {process_id}")
            process['status'] = "paused"
            return True
        elif override_action == "cancel":
            print(f"Cancelling process {process_id}")
            process['status'] = "cancelled"
            return True
        elif override_action == "modify":
            print(f"Modifying process {process_id}")
            # Handle modification
            return True
        else:
            print(f"Unknown override action: {override_action}")
            return False

class InteractiveActionExecutor(RobustActionExecutor):
    """Action executor with human interaction capabilities"""
    
    def __init__(self):
        super().__init__()
        self.hri_manager = HumanInteractionManager()
        self.enable_interactive_mode = True
    
    async def execute_interactive_task(self, task: Task) -> bool:
        """Execute a task with human interaction"""
        # Request confirmation if interactive mode is enabled
        if self.enable_interactive_mode:
            if not self.hri_manager.request_human_confirmation(task.description):
                print("Action not confirmed by human, cancelling")
                self.hri_manager.report_action_status(task, "CANCELLED", "Human denied confirmation")
                return False
        
        # Report start of task to human
        self.hri_manager.report_action_status(task, "STARTED")
        self.hri_manager.enable_human_override(task.name, task)
        
        # Register for status updates
        self.register_for_status_updates(task.name)
        
        try:
            # Execute the task with error handling
            success = await self.execute_robustly(task)
            
            # Report result
            status = "COMPLETED" if success else "FAILED"
            self.hri_manager.report_action_status(task, status)
            
            return success
            
        except Exception as e:
            self.hri_manager.report_action_status(task, "ERROR", str(e))
            raise e
    
    def register_for_status_updates(self, process_id: str):
        """Register to receive status updates"""
        self.hri_manager.add_notification_callback(
            lambda event_type, message: self.handle_status_update(process_id, event_type, message)
        )
    
    def handle_status_update(self, process_id: str, event_type: str, message: str):
        """Handle status update notifications"""
        if event_type == "ACTION_STATUS":
            # Update internal tracking or UI
            print(f"Process {process_id} status: {message}")
    
    async def execute_with_human_feedback(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute operations with human feedback between steps"""
        for i, op in enumerate(operations):
            task_type = op['type']
            params = op.get('params', {})
            
            # Create task
            if task_type == 'navigate':
                task = NavigationTask(params['destination'])
            elif task_type == 'grasp':
                task = ManipulationTask('grasp', params['object'])
            elif task_type == 'communicate':
                task = CommunicationTask(params['message'])
            else:
                print(f"Unknown task type: {task_type}")
                return False
            
            # Execute with human interaction
            success = await self.execute_interactive_task(task)
            if not success:
                print(f"Operation failed at step {i+1}: {task_type}")
                return False
            
            # In interactive mode, wait for human go-ahead for next step
            if self.enable_interactive_mode and i < len(operations) - 1:
                self.hri_manager.request_human_confirmation(
                    f"Continue to next step: {operations[i+1].get('type', 'unknown')}"
                )
        
        return True

class ActionExecutionMonitor:
    """Monitor for action execution with visualization and analytics"""
    
    def __init__(self):
        self.execution_log = []
        self.performance_metrics = {}
        self.active_monitoring = False
        
    def start_monitoring(self):
        """Start execution monitoring"""
        self.active_monitoring = True
        
    def stop_monitoring(self):
        """Stop execution monitoring"""
        self.active_monitoring = False
    
    def log_execution(self, task: Task, success: bool, execution_time: float, details: Dict[str, Any] = None):
        """Log an execution event"""
        log_entry = {
            'task_id': task.name,
            'task_type': type(task).__name__,
            'success': success,
            'execution_time': execution_time,
            'timestamp': time.time(),
            'details': details or {}
        }
        self.execution_log.append(log_entry)
        
        # Maintain only the last 1000 entries to prevent memory issues
        if len(self.execution_log) > 1000:
            self.execution_log = self.execution_log[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of execution performance"""
        if not self.execution_log:
            return {'total_executions': 0}
        
        total_executions = len(self.execution_log)
        successful_executions = sum(1 for entry in self.execution_log if entry['success'])
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        # Calculate average execution time
        execution_times = [entry['execution_time'] for entry in self.execution_log if entry['execution_time'] > 0]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # Execution time by task type
        execution_by_type = {}
        for entry in self.execution_log:
            task_type = entry['task_type']
            if task_type not in execution_by_type:
                execution_by_type[task_type] = []
            execution_by_type[task_type].append(entry['execution_time'])
        
        # Average times by type
        avg_by_type = {}
        for task_type, times in execution_by_type.items():
            avg_by_type[task_type] = sum(times) / len(times)
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'execution_time_by_type': avg_by_type
        }
    
    def visualize_execution_tree(self):
        """Visualize the execution tree"""
        # In a real implementation, this would create a visualization
        # For now, we'll just print a text representation
        print("Execution Tree Visualization:")
        for i, entry in enumerate(self.execution_log[-10:]):  # Last 10 entries
            status = "✓" if entry['success'] else "✗"
            print(f"  {i+1}. {status} {entry['task_id']} ({entry['execution_time']:.2f}s)")
```

## Complete System Integration

Here's how all components work together in a complete system:

```python
class CompleteActionExecutionSystem:
    """Complete action execution system integrating all components"""
    
    def __init__(self):
        # Core components
        self.executor = InteractiveActionExecutor()
        self.monitor = ActionExecutionMonitor()
        self.safety_controller = SafetyController(None)  # Will set robot interface later
        
        # Set up systems in executor
        self.executor.task_executor.set_systems(
            perception=None,  # Will be set later
            navigation=None,  # Will be set later
            manipulation=None,  # Will be set later
            safety_controller=self.safety_controller
        )
        
        # Start monitoring
        self.monitor.start_monitoring()
        
    def set_robot_interfaces(self, robot_interface, perception_system, navigation_system, manipulation_system):
        """Set interfaces to actual robot systems"""
        # Update safety controller with real robot interface
        self.safety_controller.robot_controller = RobotController(robot_interface)
        
        # Update executor systems
        self.executor.task_executor.set_systems(
            perception=perception_system,
            navigation=navigation_system,
            manipulation=manipulation_system,
            safety_controller=self.safety_controller
        )
        
    def execute_task_with_monitoring(self, task: Task) -> bool:
        """Execute a task with full monitoring and logging"""
        start_time = time.time()
        
        try:
            success = asyncio.run(self.executor.execute_interactive_task(task))
        except Exception as e:
            success = False
            print(f"Task execution failed with exception: {e}")
        
        execution_time = time.time() - start_time
        
        # Log the execution
        self.monitor.log_execution(
            task=task,
            success=success,
            execution_time=execution_time,
            details={'task_params': task.parameters}
        )
        
        return success
    
    def execute_complex_operation(self, operations: List[Dict[str, Any]]) -> bool:
        """Execute a complex operation with monitoring"""
        start_time = time.time()
        
        try:
            success = asyncio.run(self.executor.execute_with_human_feedback(operations))
        except Exception as e:
            success = False
            print(f"Complex operation failed with exception: {e}")
        
        execution_time = time.time() - start_time
        
        # Log the overall operation
        task = Task("complex_operation", "Complex operation execution")
        self.monitor.log_execution(
            task=task,
            success=success,
            execution_time=execution_time,
            details={'total_steps': len(operations), 'operations': operations}
        )
        
        return success
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'safety_status': {
                'emergency_stop': self.safety_controller.emergency_stop,
                'monitoring_active': self.safety_controller.is_monitoring
            },
            'performance_metrics': self.monitor.get_performance_summary(),
            'error_statistics': self.executor.error_handler.get_error_statistics(),
            'active_tasks': len(self.executor.task_executor.active_tasks)
        }
    
    def start_system(self):
        """Start the action execution system"""
        print("Starting action execution system...")
        
        # Initialize safety controller
        self.safety_controller.start_monitoring()
        
        print("Action execution system ready")
    
    def shutdown_system(self):
        """Shutdown the action execution system"""
        print("Shutting down action execution system...")
        
        # Stop safety monitoring
        self.safety_controller.stop_monitoring()
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        print("Action execution system shutdown complete")

# Example usage
def example_usage():
    """Example of using the action execution system"""
    # Create the system
    system = CompleteActionExecutionSystem()
    system.start_system()
    
    # Create some example operations
    operations = [
        {'type': 'navigate', 'params': {'destination': 'kitchen'}},
        {'type': 'communicate', 'params': {'message': 'I have reached the kitchen.'}},
        {'type': 'grasp', 'params': {'object': 'water_bottle'}}
    ]
    
    # Execute the sequence
    success = system.execute_complex_operation(operations)
    print(f"Operations completed successfully: {success}")
    
    # Get performance summary
    status = system.get_system_status()
    print(json.dumps(status, indent=2))
    
    # Shutdown
    system.shutdown_system()

if __name__ == "__main__":
    example_usage()
```

## Summary

This lesson covered the implementation of action execution systems for humanoid robots, including:

1. Hierarchical action execution architecture from high-level goals to low-level control
2. Motion planning components that convert tasks to robot trajectories
3. Multi-step task execution with proper sequencing and error handling
4. Robust error handling and recovery mechanisms with various strategies
5. Human-robot interaction protocols for action confirmation and monitoring
6. Performance monitoring and analytics for continuous improvement
7. Safety considerations throughout the execution pipeline

The action execution system is critical for translating high-level plans and goals into physical robot behaviors. By properly implementing these components, humanoid robots can perform complex tasks reliably and safely in human environments.

## Exercises

1. Implement a custom task type for your specific robot platform and integrate it with the execution system
2. Create a simulation environment to test action execution without physical hardware
3. Implement a learning mechanism that adapts action parameters based on success/failure history
4. Design safety monitors specific to your robot's capabilities and environment
5. Create a visualization system to show action execution progress in real-time

## Further Reading

- Research papers on robot action execution and control architectures
- ROS2 documentation on action libraries (rclpy.action)
- Studies on human-robot interaction in action execution
- Safety standards for collaborative robotics (ISO/TS 15066)