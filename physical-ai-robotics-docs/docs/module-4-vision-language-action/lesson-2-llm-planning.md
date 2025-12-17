---
sidebar_position: 3
---

# Lesson 2: Large Language Models (LLMs) for Robot Planning

## Learning Objectives

By the end of this lesson, you will be able to:

1. Connect large language models (LLMs) to robot planning and control systems
2. Implement natural language to action mapping using LLMs
3. Design task decomposition systems using LLM reasoning capabilities
4. Handle uncertainty and context in LLM-based planning
5. Create feedback loops between robot execution and LLM reasoning

## Introduction

Large Language Models (LLMs) have revolutionized how we approach natural language understanding and generation. For humanoid robots, LLMs serve as high-level reasoning engines that can interpret complex, natural language commands and decompose them into sequences of executable actions. This lesson explores how to integrate LLMs with robotic systems to create intelligent, human-like interfaces.

The key challenge in LLM-robot integration is translating the rich, contextual understanding that LLMs provide into concrete, executable robot behaviors. This requires careful consideration of:

- **Robot capabilities**: What the robot can actually do
- **Environment constraints**: What is possible in the current environment
- **Safety requirements**: Ensuring all planned actions are safe
- **Execution feedback**: Updating plans based on robot execution results

## LLM Integration Architecture

### 1. Basic LLM-Robot Interface

The fundamental architecture for LLM-robot integration involves creating a bridge between the natural language understanding of the LLM and the action execution capabilities of the robot:

```python
import openai
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ActionStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"

@dataclass
class RobotAction:
    """Represents a single robot action"""
    action_type: str
    parameters: Dict[str, Any]
    description: str

@dataclass
class TaskPlan:
    """Represents a complete task plan"""
    id: str
    original_command: str
    actions: List[RobotAction]
    context: Dict[str, Any]  # Environmental context
    created_at: float

class LLMRobotInterface:
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model_name = model_name
        self.robot_capabilities = self.initialize_robot_capabilities()
        self.conversation_history = []
        
    def initialize_robot_capabilities(self) -> Dict[str, Any]:
        """Define what the robot can do"""
        return {
            "navigation": {
                "supported": True,
                "max_distance": 100.0,  # meters
                "known_locations": ["kitchen", "living room", "bedroom", "office", "dining room"]
            },
            "manipulation": {
                "supported": True,
                "grasp_types": ["cylindrical", "spherical", "rectangular"],
                "weight_limit_kg": 2.0
            },
            "communication": {
                "supported": True,
                "greetings": ["wave", "nod", "speak"],
                "languages": ["en", "es", "fr"]
            },
            "sensors": {
                "camera": True,
                "lidar": True,
                "microphone": True,
                "touch_sensors": False
            }
        }
    
    def interpret_command(self, command: str, environment_context: Dict[str, Any]) -> Optional[TaskPlan]:
        """Use LLM to interpret command and create task plan"""
        # Build a detailed prompt with robot capabilities and environment context
        prompt = self.build_interpretation_prompt(command, environment_context)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                functions=[{
                    "name": "create_task_plan",
                    "description": "Create a detailed task plan for the robot",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "action_type": {"type": "string", "description": "Type of action"},
                                        "parameters": {"type": "object", "description": "Action parameters"},
                                        "description": {"type": "string", "description": "Human-readable action description"}
                                    },
                                    "required": ["action_type", "parameters", "description"]
                                }
                            },
                            "reasoning": {"type": "string", "description": "Explanation of planning decisions"}
                        },
                        "required": ["actions", "reasoning"]
                    }
                }],
                function_call={"name": "create_task_plan"}
            )
            
            # Extract function arguments
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            actions_data = function_args["actions"]
            
            # Create RobotAction objects
            actions = [
                RobotAction(
                    action_type=action_data["action_type"],
                    parameters=action_data["parameters"],
                    description=action_data["description"]
                )
                for action_data in actions_data
            ]
            
            # Create and return task plan
            import uuid
            import time
            task_plan = TaskPlan(
                id=str(uuid.uuid4()),
                original_command=command,
                actions=actions,
                context=environment_context,
                created_at=time.time()
            )
            
            return task_plan
            
        except Exception as e:
            print(f"Error interpreting command: {e}")
            return None
    
    def build_interpretation_prompt(self, command: str, environment_context: Dict[str, Any]) -> str:
        """Build prompt for LLM interpretation"""
        return f"""
        Human command: "{command}"
        
        Robot capabilities:
        {json.dumps(self.robot_capabilities, indent=2)}
        
        Current environment context:
        {json.dumps(environment_context, indent=2)}
        
        Please create a detailed task plan with specific, executable actions that the robot can perform to fulfill this command.
        Consider the robot's capabilities, current environment, and safety requirements.
        """
    
    def get_system_prompt(self) -> str:
        """Get system prompt for LLM"""
        return """
        You are an expert robot task planner. Your role is to interpret human commands and create detailed, executable task plans for a humanoid robot.
        
        Guidelines:
        1. Always consider the robot's actual capabilities and limitations
        2. Break complex commands into simple, sequential actions
        3. Include safety checks and validation steps where appropriate
        4. Use precise language for action parameters
        5. Only suggest actions that the robot is capable of performing
        6. Consider the environmental context when planning
        """
```

### 2. Context-Aware Planning

LLMs are particularly powerful when they can incorporate context about the current environment and robot state:

```python
from datetime import datetime

class ContextualLLMPlanner(LLMRobotInterface):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model_name)
        self.context_history = []  # Maintain context for conversation continuity
        self.known_objects = set()  # Objects the robot has learned about
        self.navigation_map = {}    # Learned navigation locations

    def get_comprehensive_context(self, environment_sensors: Dict[str, Any]) -> Dict[str, Any]:
        """Gather comprehensive context for planning"""
        return {
            "robot_state": {
                "location": environment_sensors.get("location", "unknown"),
                "battery_level": environment_sensors.get("battery", 100),
                "gripper_status": environment_sensors.get("gripper", "open"),
                "current_task": environment_sensors.get("current_task", "idle")
            },
            "environment": {
                "objects": environment_sensors.get("objects", []),
                "navigation": environment_sensors.get("navigation_map", {}),
                "humans": environment_sensors.get("humans", []),
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "capabilities": self.robot_capabilities,
            "recent_interactions": self.context_history[-5:],  # Last 5 interactions
            "known_objects": list(self.known_objects),
            "learned_locations": list(self.navigation_map.keys())
        }

    def plan_with_context(self, command: str, environment_sensors: Dict[str, Any]) -> Optional[TaskPlan]:
        """Plan actions using comprehensive environmental context"""
        context = self.get_comprehensive_context(environment_sensors)
        
        # For complex commands, use a more detailed prompt
        detailed_prompt = self.build_contextual_prompt(command, context)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_contextual_system_prompt()},
                    {"role": "user", "content": detailed_prompt}
                ],
                temperature=0.2,  # Slightly higher for more adaptive planning
                max_tokens=1000,
                functions=[{
                    "name": "generate_detailed_plan",
                    "description": "Generate a detailed, multi-step plan with safety considerations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action_sequence": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "action_type": {"type": "string"},
                                        "parameters": {"type": "object"},
                                        "safety_check": {"type": "string", "description": "What to check before executing"},
                                        "success_criteria": {"type": "string", "description": "How to verify success"},
                                        "fallback_action": {"type": "string", "description": "What to do if action fails"}
                                    },
                                    "required": ["action_type", "parameters", "safety_check", "success_criteria"]
                                }
                            },
                            "reasoning": {"type": "string"},
                            "estimated_duration_seconds": {"type": "number"}
                        },
                        "required": ["action_sequence", "reasoning"]
                    }
                }],
                function_call={"name": "generate_detailed_plan"}
            )
            
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            action_sequence = function_args["action_sequence"]
            
            # Create RobotAction objects with safety and fallback info
            actions = []
            for action_data in action_sequence:
                # Create action with metadata
                action = RobotAction(
                    action_type=action_data["action_type"],
                    parameters=action_data["parameters"],
                    description=action_data.get("description", f"Perform {action_data['action_type']}")
                )
                actions.append(action)
                
                # Store safety and fallback information for execution
                action.metadata = {
                    "safety_check": action_data.get("safety_check", "none"),
                    "success_criteria": action_data.get("success_criteria", "none"),
                    "fallback_action": action_data.get("fallback_action", "none")
                }
            
            # Create and return task plan
            import uuid
            import time
            task_plan = TaskPlan(
                id=str(uuid.uuid4()),
                original_command=command,
                actions=actions,
                context=context,
                created_at=time.time()
            )
            
            # Update context history
            self.context_history.append({
                "command": command,
                "plan": [action.action_type for action in actions],
                "timestamp": time.time()
            })
            
            return task_plan
            
        except Exception as e:
            print(f"Error in contextual planning: {e}")
            return None

    def build_contextual_prompt(self, command: str, context: Dict[str, Any]) -> str:
        """Build a detailed prompt with full context"""
        return f"""
        Human command: "{command}"

        Current context:
        - Robot state: {json.dumps(context['robot_state'], indent=2)}
        - Environmental observations: {json.dumps(context['environment'], indent=2)}
        - Robot capabilities: {json.dumps(context['capabilities'], indent=2)}
        - Recent interactions: {json.dumps(context['recent_interactions'], indent=2)}

        Please generate a detailed action sequence that:
        1. Considers the robot's current state (especially battery level and location)
        2. Uses environmental information appropriately
        3. Includes safety checks before risky operations
        4. Has fallback plans for likely failure scenarios
        5. Optimizes for efficiency given the current context
        6. Includes success criteria for each action
        """
    
    def get_contextual_system_prompt(self) -> str:
        """System prompt for contextual planning"""
        return """
        You are an advanced robot task planner with access to comprehensive environmental context.
        Your planning must be precise, safe, and efficient, taking full advantage of available context.

        When planning:
        1. Always consider the robot's current location and battery level
        2. Use environmental information to inform navigation and manipulation decisions
        3. Include appropriate safety checks before physical actions
        4. Provide fallback options for common failure scenarios
        5. Account for potential human presence in the environment
        6. Optimize for time and energy efficiency based on current conditions
        """
```

## Handling Ambiguity and Uncertainty

One of the key strengths of LLMs is their ability to handle ambiguous commands by asking clarifying questions or making reasonable assumptions:

```python
class AmbiguityHandler:
    def __init__(self, llm_interface: ContextualLLMPlanner):
        self.llm_planner = llm_interface
        self.ambiguity_threshold = 0.7  # Confidence threshold

    def detect_ambiguity(self, command: str, context: Dict[str, Any]) -> tuple[bool, str]:
        """Detect if a command is ambiguous and needs clarification"""
        prompt = f"""
        Analyze this robot command for ambiguity: "{command}"
        
        Robot context:
        - Capabilities: {context.get('capabilities', {})}
        - Current environment: {context.get('environment', {})}
        - Current state: {context.get('robot_state', {})}
        
        Is this command ambiguous? If so, what specific information is missing or unclear?
        Provide your response as JSON:
        {{
            "ambiguous": true/false,
            "clarification_needed": "brief description of what is unclear"
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_planner.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying ambiguous instructions for robots. Be precise about what information is missing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('ambiguous', False), result.get('clarification_needed', '')
            
        except Exception as e:
            print(f"Error detecting ambiguity: {e}")
            return False, ""

    def generate_clarification_request(self, command: str, ambiguity_info: str, context: Dict[str, Any]) -> str:
        """Generate a specific clarification request for the user"""
        prompt = f"""
        The following command is ambiguous: "{command}"
        The ambiguity is: {ambiguity_info}
        
        Robot capabilities: {json.dumps(context.get('capabilities', {}), indent=2)}
        Current environment: {json.dumps(context.get('environment', {}), indent=2)}
        
        Generate a specific, helpful question to ask the user to resolve this ambiguity.
        The question should:
        1. Be natural and human-friendly
        2. Focus on the specific information needed
        3. Reference the robot's capabilities when possible
        4. Consider the environmental context
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_planner.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful robot assistant. Ask natural questions to clarify ambiguous commands."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating clarification: {e}")
            return "I'm not sure what you mean. Could you please clarify your command?"

    def resolve_ambiguity(self, command: str, context: Dict[str, Any]) -> tuple[str, bool]:
        """Attempt to resolve ambiguity in the command"""
        is_ambiguous, ambiguity_info = self.detect_ambiguity(command, context)
        
        if is_ambiguous:
            clarification_request = self.generate_clarification_request(command, ambiguity_info, context)
            return clarification_request, True  # Needs clarification
        else:
            return command, False  # No clarification needed
```

## Task Decomposition and Execution Planning

LLMs excel at breaking down complex tasks into manageable steps:

```python
class TaskDecompositionSystem:
    def __init__(self, llm_interface: ContextualLLMPlanner):
        self.llm_planner = llm_interface

    def decompose_task(self, high_level_task: str, context: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Decompose a high-level task into specific subtasks"""
        prompt = f"""
        Decompose this high-level task into specific, actionable subtasks: "{high_level_task}"
        
        Robot context:
        - Capabilities: {json.dumps(context.get('capabilities', {}), indent=2)}
        - Current state: {json.dumps(context.get('robot_state'), indent=2)}
        - Environment: {json.dumps(context.get('environment', {}), indent=2)}
        
        Decompose the task into 3-7 specific subtasks that the robot can execute.
        For each subtask, specify:
        1. Action type
        2. Parameters needed
        3. Expected outcome
        4. Potential challenges
        5. How it contributes to the overall goal
        """
        
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_planner.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at breaking down complex tasks into actionable steps for robots. Be specific and practical."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                functions=[{
                    "name": "decompose_task",
                    "description": "Break down a complex task into specific subtasks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subtasks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "order": {"type": "integer", "description": "Execution order"},
                                        "action_type": {"type": "string", "description": "Type of action"},
                                        "parameters": {"type": "object", "description": "Action parameters"},
                                        "description": {"type": "string", "description": "What this subtask does"},
                                        "expected_outcome": {"type": "string", "description": "What should happen after this subtask"},
                                        "challenges": {"type": "array", "items": {"type": "string"}, "description": "Potential challenges"},
                                        "contributes_to": {"type": "string", "description": "How this helps the overall goal"}
                                    },
                                    "required": ["order", "action_type", "parameters", "description", "expected_outcome"]
                                }
                            }
                        },
                        "required": ["subtasks"]
                    }
                }],
                function_call={"name": "decompose_task"}
            )
            
            function_args = json.loads(response.choices[0].message.function_call.arguments)
            return sorted(function_args["subtasks"], key=lambda x: x["order"])
            
        except Exception as e:
            print(f"Error decomposing task: {e}")
            return None

    def create_execution_plan(self, subtasks: List[Dict[str, Any]], context: Dict[str, Any]) -> Optional[TaskPlan]:
        """Create a complete execution plan from subtasks"""
        # Convert subtasks to RobotAction objects
        actions = []
        for subtask in subtasks:
            action = RobotAction(
                action_type=subtask["action_type"],
                parameters=subtask["parameters"],
                description=subtask["description"]
            )
            actions.append(action)

        # Create and return task plan
        import uuid
        import time
        task_plan = TaskPlan(
            id=str(uuid.uuid4()),
            original_command=f"Execute: {', '.join([st['description'] for st in subtasks])}",
            actions=actions,
            context=context,
            created_at=time.time()
        )
        
        return task_plan
```

## Integration with Robot Control Systems

LLMs generate plans that must be executed by the robot's control system:

```python
import asyncio
from typing import Callable

class PlanExecutor:
    def __init__(self, robot_interface: Any):
        self.robot_interface = robot_interface  # Interface to actual robot
        self.action_library = {
            'navigate': self.execute_navigation,
            'grasp': self.execute_grasping,
            'place': self.execute_placement,
            'greet': self.execute_greeting,
            'communicate': self.execute_communication,
            'find_object': self.execute_object_search,
            'wait': self.execute_wait
        }
        self.event_callbacks = {}  # Store callbacks for plan events

    def execute_plan(self, task_plan: TaskPlan, on_progress: Callable = None) -> bool:
        """Execute a plan generated by the LLM"""
        print(f"Starting execution of plan: {task_plan.original_command}")
        
        success = True
        completed_actions = 0
        
        for i, action in enumerate(task_plan.actions):
            print(f"Executing action {i+1}/{len(task_plan.actions)}: {action.action_type}")
            
            # Check for safety conditions before executing
            if not self.check_safety_conditions(action, task_plan.context):
                print(f"Safety check failed for action: {action.action_type}")
                success = False
                break
            
            # Execute the action
            try:
                action_success = self.execute_single_action(action, task_plan.context)
                
                if not action_success:
                    print(f"Action failed: {action.action_type}")
                    # Try fallback action if available
                    fallback_success = self.execute_fallback_action(action)
                    if not fallback_success:
                        success = False
                        break
                
                completed_actions += 1
                
                # Report progress if callback is provided
                if on_progress:
                    on_progress(i+1, len(task_plan.actions), action.action_type, action_success)
                    
            except Exception as e:
                print(f"Error executing action {action.action_type}: {e}")
                success = False
                break
        
        print(f"Plan execution completed. Success: {success}, Actions completed: {completed_actions}/{len(task_plan.actions)}")
        return success

    def execute_single_action(self, action: RobotAction, context: Dict[str, Any]) -> bool:
        """Execute a single robot action"""
        action_type = action.action_type
        
        if action_type in self.action_library:
            # Validate action parameters before execution
            if self.validate_action_parameters(action, context):
                return self.action_library[action_type](action, context)
            else:
                print(f"Invalid parameters for action: {action_type}")
                return False
        else:
            print(f"Unknown action type: {action_type}")
            return False

    def validate_action_parameters(self, action: RobotAction, context: Dict[str, Any]) -> bool:
        """Validate action parameters before execution"""
        # Check if robot has required capabilities
        if action.action_type == 'grasp':
            required_capability = self.robot_interface.get_capability('manipulation')
            if not required_capability.get('supported', False):
                return False
                
        # Check if parameters are within robot limits
        if action.action_type == 'navigate':
            destination = action.parameters.get('location')
            known_locations = context.get('capabilities', {}).get('navigation', {}).get('known_locations', [])
            if destination not in known_locations:
                return False
            
        # Add more validation rules as needed
        return True

    def check_safety_conditions(self, action: RobotAction, context: Dict[str, Any]) -> bool:
        """Check safety conditions before executing an action"""
        # Check battery level for navigation tasks
        if action.action_type == 'navigate':
            battery_level = context.get('robot_state', {}).get('battery_level', 100)
            if battery_level < 15:  # Require at least 15% battery for navigation
                return False
        
        # Check human presence for navigation near people
        if action.action_type in ['navigate', 'manipulate']:
            humans_nearby = context.get('environment', {}).get('humans', [])
            # Add safety logic to ensure safe operation around humans
        
        # Add more safety checks as needed
        return True

    def execute_navigation(self, action: RobotAction, context: Dict[str, Any]) -> bool:
        """Execute navigation action"""
        destination = action.parameters.get('location')
        if not destination:
            print("Navigation action missing destination parameter")
            return False

        print(f"Navigating to {destination}")
        # In real implementation, call robot's navigation system
        # success = self.robot_interface.navigate_to(destination)
        # return success
        
        # Simulated navigation
        import time
        time.sleep(1)  # Simulate navigation time
        return True

    def execute_grasping(self, action: RobotAction, context: Dict[str, Any]) -> bool:
        """Execute grasping action"""
        object_name = action.parameters.get('object')
        if not object_name:
            print("Grasping action missing object parameter")
            return False

        print(f"Attempting to grasp {object_name}")
        # In real implementation, call robot's manipulation system
        # success = self.robot_interface.grasp_object(object_name)
        # return success
        
        # Simulated grasping
        import time
        time.sleep(1)  # Simulate grasping time
        return True

    def execute_greeting(self, action: RobotAction, context: Dict[str, Any]) -> bool:
        """Execute greeting action"""
        greeting_type = action.parameters.get('type', 'wave')
        print(f"Performing greeting: {greeting_type}")
        # In real implementation, call robot's communication system
        # success = self.robot_interface.perform_greeting(greeting_type)
        # return success
        
        # Simulated greeting
        import time
        time.sleep(0.5)
        return True

    def execute_communication(self, action: RobotAction, context: Dict[str, Any]) -> bool:
        """Execute communication action"""
        message = action.parameters.get('message')
        print(f"Communicating: {message}")
        # In real implementation, call robot's speech system
        # success = self.robot_interface.speak(message)
        # return success
        
        # Simulated communication
        import time
        time.sleep(0.5)
        return True

    def execute_fallback_action(self, action: RobotAction) -> bool:
        """Execute fallback action if primary action fails"""
        # Get fallback action from action metadata or predefined defaults
        fallback = getattr(action, 'metadata', {}).get('fallback_action')
        if fallback:
            print(f"Executing fallback action: {fallback}")
            # Parse and execute the fallback action
            return True  # Simplified
        else:
            print("No fallback action available")
            return False
```

## Feedback and Learning Loop

LLMs can improve their planning by learning from execution feedback:

```python
from typing import Literal

class LearningLLMPlanner:
    def __init__(self, llm_interface: ContextualLLMPlanner):
        self.llm_planner = llm_interface
        self.execution_history = []
        self.performance_metrics = {
            'success_rate': 0.0,
            'average_plan_length': 0,
            'common_failures': {}
        }

    def record_execution_result(self, task_plan: TaskPlan, success: bool, execution_log: List[Dict[str, Any]]):
        """Record the result of a plan execution"""
        result_record = {
            'plan_id': task_plan.id,
            'original_command': task_plan.original_command,
            'actions_count': len(task_plan.actions),
            'success': success,
            'execution_log': execution_log,
            'timestamp': task_plan.created_at
        }
        
        self.execution_history.append(result_record)
        
        # Update performance metrics
        self.update_performance_metrics(success, task_plan)
        
        # For successful executions, add to positive examples
        if success:
            self.add_to_positive_examples(task_plan)

    def update_performance_metrics(self, success: bool, task_plan: TaskPlan):
        """Update performance metrics based on execution result"""
        # Calculate success rate
        total_executions = len(self.execution_history)
        if total_executions > 0:
            successful_executions = sum(1 for record in self.execution_history if record['success'])
            self.performance_metrics['success_rate'] = successful_executions / total_executions
            
            # Calculate average plan length
            total_actions = sum(record['actions_count'] for record in self.execution_history)
            self.performance_metrics['average_plan_length'] = total_actions / total_executions

    def add_to_positive_examples(self, task_plan: TaskPlan):
        """Add successful plan to positive examples for future refinement"""
        # This would be used to fine-tune or improve future planning
        # For now, we'll just record it
        pass

    def refine_plan_with_feedback(self, command: str, context: Dict[str, Any], previous_attempts: List[Dict[str, Any]] = None) -> Optional[TaskPlan]:
        """Refine planning based on execution feedback"""
        if not previous_attempts:
            return self.llm_planner.plan_with_context(command, self.extract_sensor_data(context))
        
        # Build prompt with previous attempts for learning
        feedback_prompt = self.build_feedback_prompt(command, context, previous_attempts)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_planner.model_name,
                messages=[
                    {"role": "system", "content": self.get_learning_system_prompt()},
                    {"role": "user", "content": feedback_prompt}
                ],
                temperature=0.2
            )
            
            # In a real implementation, we would extract the plan from the response
            # For this example, we'll simulate a refined plan
            return self.create_refined_plan(command, context, previous_attempts, response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error in feedback-based planning: {e}")
            return None

    def build_feedback_prompt(self, command: str, context: Dict[str, Any], previous_attempts: List[Dict[str, Any]]) -> str:
        """Build a prompt that includes execution feedback"""
        return f"""
        Original command: "{command}"
        
        Robot context: {json.dumps(context, indent=2)}
        
        Previous execution attempts:
        {json.dumps(previous_attempts, indent=2)}
        
        Based on the previous attempts and their outcomes, create an improved plan for the robot to fulfill the command.
        Consider why previous attempts failed and how to address those issues.
        """

    def get_learning_system_prompt(self) -> str:
        """System prompt for learning-based planning"""
        return """
        You are an advanced robot planner that learns from execution feedback.
        When creating new plans, consider:
        1. Why previous attempts may have failed
        2. How to adapt the approach based on environmental feedback
        3. How to make plans more robust against likely failure points
        4. How to optimize for the specific execution environment
        """

    def extract_sensor_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sensor data from context"""
        # For now, returning the context as is
        # In real implementation, would extract specific sensor readings
        return context

    def create_refined_plan(self, command: str, context: Dict[str, Any], previous_attempts: List[Dict[str, Any]], llm_response: str) -> Optional[TaskPlan]:
        """Create a refined plan based on LLM response"""
        # For this example, we'll simulate creating a new plan
        # In a real implementation, we would parse the LLM response properly
        import uuid
        import time
        
        # Simulating a refined plan
        task_plan = TaskPlan(
            id=str(uuid.uuid4()),
            original_command=command,
            actions=[
                RobotAction(
                    action_type="navigate",
                    parameters={"location": "kitchen"},
                    description="Navigate to kitchen"
                )
            ],
            context=context,
            created_at=time.time()
        )
        
        return task_plan
```

## Integration Example

Here's how all components work together in a complete system:

```python
class CompleteLLMRobotSystem:
    def __init__(self, api_key: str):
        # Initialize components
        self.llm_interface = ContextualLLMPlanner(api_key)
        self.ambiguity_handler = AmbiguityHandler(self.llm_interface)
        self.task_decomposer = TaskDecompositionSystem(self.llm_interface)
        self.plan_executor = PlanExecutor(robot_interface=None)  # Will be set to actual robot interface
        self.learning_system = LearningLLMPlanner(self.llm_interface)
        
        # System state
        self.current_plan = None
        self.is_executing = False

    def process_command(self, command: str, environment_sensors: Dict[str, Any]) -> bool:
        """Process a command from start to finish"""
        print(f"Processing command: {command}")
        
        # Step 1: Check for ambiguity
        clarification, needs_clarification = self.ambiguity_handler.resolve_ambiguity(command, environment_sensors)
        if needs_clarification:
            print(f"Need clarification: {clarification}")
            return False  # In real system, would get clarification from user
        
        # Step 2: Create initial plan
        if self.is_executing:
            print("Cannot process new command while executing current plan")
            return False
            
        # Use learning system for adaptive planning
        task_plan = self.learning_system.refine_plan_with_feedback(command, environment_sensors)
        if not task_plan:
            # If learning system fails, try basic planning
            task_plan = self.llm_interface.plan_with_context(command, environment_sensors)
        
        if not task_plan:
            print("Could not generate a plan for the command")
            return False
        
        # Step 3: Execute the plan
        self.current_plan = task_plan
        self.is_executing = True
        
        try:
            execution_result = self.plan_executor.execute_plan(
                task_plan, 
                on_progress=self.on_execution_progress
            )
            
            # Record execution result
            self.learning_system.record_execution_result(
                task_plan, 
                execution_result, 
                self.get_execution_log(task_plan)
            )
            
            return execution_result
            
        finally:
            self.is_executing = False
            self.current_plan = None

    def on_execution_progress(self, current_step: int, total_steps: int, action: str, success: bool):
        """Callback for execution progress"""
        status = "✓" if success else "✗"
        print(f"Execution progress: {current_step}/{total_steps} - {action} {status}")

    def get_execution_log(self, task_plan: TaskPlan) -> List[Dict[str, Any]]:
        """Generate execution log for learning system"""
        # In real implementation, would contain actual execution data
        return [
            {
                "action": action.action_type,
                "parameters": action.parameters,
                "executed": True,
                "result": "success"
            }
            for action in task_plan.actions
        ]

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'is_executing': self.is_executing,
            'current_plan_id': self.current_plan.id if self.current_plan else None,
            'current_command': self.current_plan.original_command if self.current_plan else None,
            'performance_metrics': self.learning_system.performance_metrics
        }
```

## Safety and Error Handling

LLM-based robot planning systems must have robust safety measures:

```python
class SafeLLMRobotInterface:
    def __init__(self, llm_interface: ContextualLLMPlanner):
        self.llm_interface = llm_interface
        self.safety_constraints = self.define_safety_constraints()
        self.emergency_stop_active = False

    def define_safety_constraints(self) -> Dict[str, Any]:
        """Define safety constraints for LLM planning"""
        return {
            'navigation': {
                'no_go_zones': ['staircase', 'construction_area'],
                'minimum_distance_to_humans': 1.0,  # meter
                'speed_limits': {
                    'near_humans': 0.3,  # m/s
                    'open_space': 0.8
                }
            },
            'manipulation': {
                'weight_limit': 2.0,  # kg
                'temperature_limits': {'min': -10, 'max': 60},  # Celsius
                'safe_grasp_types': ['cylindrical', 'spherical', 'rectangular']
            },
            'communication': {
                'volume_limits': {'min': 0.1, 'max': 0.8},  # Fraction of max volume
                'inappropriate_content_filter': True
            }
        }

    def validate_plan_safety(self, task_plan: TaskPlan) -> tuple[bool, List[str]]:
        """Validate that a plan meets safety constraints"""
        safety_issues = []
        
        for action in task_plan.actions:
            action_issues = self.check_action_safety(action, task_plan.context)
            safety_issues.extend(action_issues)
        
        is_safe = len(safety_issues) == 0
        return is_safe, safety_issues

    def check_action_safety(self, action: RobotAction, context: Dict[str, Any]) -> List[str]:
        """Check if an action is safe to execute"""
        issues = []
        
        if action.action_type == 'navigate':
            destination = action.parameters.get('location', '').lower()
            no_go_zones = self.safety_constraints['navigation']['no_go_zones']
            if destination in no_go_zones:
                issues.append(f"Cannot navigate to {destination}, it's a restricted area")
                
            # Check if path is clear of humans
            humans = context.get('environment', {}).get('humans', [])
            if len(humans) > 0:
                # Check distance to humans
                pass
        
        elif action.action_type == 'grasp':
            # Check object weight
            object_info = self.get_object_info(action.parameters.get('object', ''))
            if object_info and object_info.get('weight', 0) > self.safety_constraints['manipulation']['weight_limit']:
                issues.append(f"Object too heavy to grasp: {object_info['weight']}kg > {self.safety_constraints['manipulation']['weight_limit']}kg")
        
        # Add more safety checks as needed
        
        return issues

    def get_object_info(self, object_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an object"""
        # In real implementation, would query perception system
        # For now, return None or some default values
        return None

    def safe_process_command(self, command: str, environment_sensors: Dict[str, Any]) -> bool:
        """Process command with safety validation"""
        if self.emergency_stop_active:
            print("Emergency stop is active. Cannot process commands.")
            return False
        
        # Generate plan
        task_plan = self.llm_interface.plan_with_context(command, environment_sensors)
        if not task_plan:
            print("Could not generate a plan")
            return False
        
        # Validate safety
        is_safe, safety_issues = self.validate_plan_safety(task_plan)
        if not is_safe:
            print(f"Plan failed safety validation: {', '.join(safety_issues)}")
            return False
        
        print("Plan passed safety validation, proceeding with execution")
        # In real implementation, would execute the plan
        return True
```

## Performance Optimization

For real-time applications, LLM-robot interfaces need performance optimization:

```python
import asyncio
from functools import lru_cache
import time

class OptimizedLLMRobotInterface:
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model_name = model_name
        self.robot_capabilities = self.initialize_robot_capabilities()
        
        # Caching for frequently used functions
        self.cached_capabilities = self.robot_capabilities
        self.response_cache = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds
    
    @lru_cache(maxsize=128)
    def cached_command_interpretation(self, command: str, capabilities_json: str, context_json: str) -> Optional[TaskPlan]:
        """Cached command interpretation for frequently used commands"""
        # This would perform the same interpretation as before but cached
        # For this example, we'll return None to indicate cache miss
        return None

    async def async_plan_command(self, command: str, environment_context: Dict[str, Any]) -> Optional[TaskPlan]:
        """Asynchronously plan a command to avoid blocking"""
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - (current_time - self.last_request_time))
        
        # Throttling
        self.last_request_time = time.time()
        
        # Perform planning
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._plan_command_sync, command, environment_context)
    
    def _plan_command_sync(self, command: str, environment_context: Dict[str, Any]) -> Optional[TaskPlan]:
        """Synchronous command planning (for use with executor)"""
        # Implementation of command planning
        import uuid
        import time
        
        # Simulated planning
        task_plan = TaskPlan(
            id=str(uuid.uuid4()),
            original_command=command,
            actions=[RobotAction(
                action_type="example_action",
                parameters={"example": "params"},
                description="Example action"
            )],
            context=environment_context,
            created_at=time.time()
        )
        
        return task_plan

    def batch_process_commands(self, commands: List[str], environment_context: Dict[str, Any]) -> List[Optional[TaskPlan]]:
        """Process multiple commands efficiently"""
        # In a real implementation, we might batch requests to the LLM API
        # or process commands in parallel
        results = []
        for command in commands:
            plan = self._plan_command_sync(command, environment_context)
            results.append(plan)
        
        return results
```

## Summary

This lesson covered the implementation of LLM-based planning systems for humanoid robots, including:

1. Basic LLM-robot interface architecture for natural language understanding
2. Context-aware planning that considers robot state and environment
3. Ambiguity detection and resolution for robust command interpretation
4. Task decomposition to break complex commands into executable steps
5. Integration with robot control systems for plan execution
6. Learning from execution feedback to improve future planning
7. Safety measures and error handling for reliable operation
8. Performance optimization techniques for real-time applications

LLM-robot integration enables sophisticated natural language interfaces that allow humans to interact with robots using everyday language. By properly implementing these components, we can create robots that understand complex, high-level commands and execute them safely and effectively in real-world environments.

## Exercises

1. Implement a custom prompt engineering approach to improve planning accuracy for your specific robot platform
2. Create a safety validation system that prevents the execution of potentially dangerous plans
3. Develop a learning system that adapts planning based on successful execution examples
4. Implement a fallback mechanism for when the LLM is unavailable
5. Design a multimodal interface that combines LLM planning with visual input

## Further Reading

- Papers on LLM-robot interaction and planning
- OpenAI API documentation for function calling
- Research on safe AI for robotics applications
- Studies on human-robot interaction with LLM interfaces