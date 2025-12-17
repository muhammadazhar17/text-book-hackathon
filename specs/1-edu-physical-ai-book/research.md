# Research: Physical AI & Humanoid Robotics Course Book

## Docusaurus Setup and Configuration

### Decision: Use Docusaurus 3.x with TypeScript
- Rationale: Docusaurus 3.x offers improved performance, better TypeScript support, and modern tooling compared to version 2.x. The TypeScript support will provide better development experience and maintainability.

### Decision: Use Better Auth for Authentication
- Rationale: Better Auth provides a simple, secure authentication solution that works well with Docusaurus sites. It supports multiple providers and has good documentation.

### Decision: Backend for Progress Tracking
- Rationale: Since we need advanced progress tracking with detailed analytics stored on a server backend, a backend solution is required for storing user progress, quiz results, and analytics data.

## Content Development Phase

### Decision: Modular Content Structure
- Rationale: Organizing content into modules and lessons allows for independent development, easier maintenance, and clear progression for students.

### Decision: Markdown-based Content with MDX
- Rationale: Docusaurus supports Markdown and MDX (Markdown with JSX), which allows for rich interactive content while keeping content creation accessible to educators without deep technical knowledge.

## File Structure for Modules and Lessons

### Decision: Hierarchical Folder Structure
- Rationale: A clear hierarchical structure with separate folders for each module and lesson will make content management easier and allow for independent deployment of modules if needed.

```
docs/
│   ├── intro.md                      # What is Physical AI?
│   ├── why-physical-ai-matters.md    # Embodied intelligence explanation
│   ├── learning-outcomes.md
│   ├── weekly-breakdown.md
│   ├── assessments.md
│   ├── hardware-requirements/
│   │   ├── digital-twin-workstation.md
│   │   ├── physical-ai-edge-kit.md
│   │   ├── robot-lab-options.md
│   │   └── architecture-summary.md
│   │
│   ├── module-1-ros2-nervous-system/
│   │   ├── overview.md
│   │   ├── lesson-1-ros2-basics.md
│   │   ├── lesson-2-nodes-topics-services.md
│   │   ├── lesson-3-rclpy-python.md
│   │   └── lesson-4-urdf-humanoids.md
│   │
│   ├── module-2-digital-twin-simulation/
│   │   ├── overview.md
│   │   ├── lesson-1-gazebo-physics.md
│   │   ├── lesson-2-collisions-gravity.md
│   │   ├── lesson-3-unity-rendering.md
│   │   └── lesson-4-simulated-sensors.md
│   │
│   ├── module-3-nvidia-isaac-ai-brain/
│   │   ├── overview.md
│   │   ├── lesson-1-isaac-sim.md
│   │   ├── lesson-2-isaac-ros-vslam.md
│   │   ├── lesson-3-nav2-path-planning.md
│   │   └── lesson-4-ai-perception.md
│   │
│   ├── module-4-vision-language-action/
│   │   ├── overview.md
│   │   ├── lesson-1-whisper-voice-commands.md
│   │   ├── lesson-2-llm-planning.md
│   │   ├── lesson-3-robot-actions.md
│   │   └── capstone-autonomous-humanoid.md
│   │
│   └── glossary.md
```

## Accessibility and Internationalization

### Decision: Implement WCAG 2.1 AA Compliance
- Rationale: This is required by the feature specification and ensures the course is accessible to all students, including those with disabilities.

### Decision: Urdu Language Support
- Rationale: Required by the feature specification, this will involve using Docusaurus' built-in i18n capabilities to provide Urdu translations alongside the English content.

## Technology Stack Research

### Decision: Node.js Backend with Express
- Rationale: Node.js with Express provides a lightweight solution that integrates well with the Docusaurus frontend. It's also a common technology that many developers are familiar with.

### Decision: Database for User Data
- Rationale: A database (likely PostgreSQL or MongoDB) will be needed to store user accounts, progress tracking data, and other dynamic information. The exact choice will depend on deployment requirements and scale.

## Deployment Strategy

### Decision: Static Site Generation with Dynamic Features
- Rationale: Docusaurus' static site generation provides excellent performance and SEO, while API routes can handle dynamic features like progress tracking and authentication.