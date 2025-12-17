import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  // Manual sidebar structure for the Physical AI & Humanoid Robotics Course
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'intro',
        // removed 'assessments' since it doesn't exist
      ],
      link: {
        type: 'generated-index',
      },
    },
    {
      type: 'category',
      label: 'Hardware Requirements',
      items: [
        'hardware-requirements/digital-twin-workstation',
        'hardware-requirements/physical-ai-edge-kit',
        'hardware-requirements/robot-lab-options',
        'hardware-requirements/architecture-summary',
      ],
      link: {
        type: 'generated-index',
      },
    },
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2-nervous-system/overview',
        'module-1-ros2-nervous-system/lesson-1-ros2-basics',
        'module-1-ros2-nervous-system/lesson-2-nodes-topics-services',
        'module-1-ros2-nervous-system/lesson-3-rclpy-python',
        'module-1-ros2-nervous-system/lesson-3-urdf-humanoids',
        'module-1-ros2-nervous-system/lesson-4-practical-exercises',
      ],
      link: {
        type: 'generated-index',
      },
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin Simulation',
      items: [
        'module-2-digital-twin-simulation/overview',
        'module-2-digital-twin-simulation/lesson-1-gazebo-physics',
        'module-2-digital-twin-simulation/lesson-2-collisions-gravity',
        'module-2-digital-twin-simulation/lesson-3-unity-rendering',
        'module-2-digital-twin-simulation/lesson-4-simulated-sensors',
      ],
      link: {
        type: 'generated-index',
      },
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac AI Brain',
      items: [
        'module-3-nvidia-isaac-ai-brain/overview',
        'module-3-nvidia-isaac-ai-brain/lesson-1-isaac-sim',
        'module-3-nvidia-isaac-ai-brain/lesson-2-isaac-ros-vslam',
        'module-3-nvidia-isaac-ai-brain/lesson-3-nav2-path-planning',
        'module-3-nvidia-isaac-ai-brain/lesson-4-ai-perception',
      ],
      link: {
        type: 'generated-index',
      },
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action',
      items: [
        'module-4-vision-language-action/overview',
        'module-4-vision-language-action/lesson-1-whisper-voice-commands',
        'module-4-vision-language-action/lesson-2-llm-planning',
        'module-4-vision-language-action/lesson-3-robot-actions',
        'module-4-vision-language-action/capstone-autonomous-humanoid',
      ],
      link: {
        type: 'generated-index',
      },
    },
  ],
};

export default sidebars;
