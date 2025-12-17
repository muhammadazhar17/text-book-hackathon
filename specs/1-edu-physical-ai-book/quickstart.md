# Quickstart Guide: Physical AI & Humanoid Robotics Course Book

## Prerequisites

- Node.js 18.x or higher
- npm or yarn package manager
- Git version control system
- A code editor (VS Code recommended)

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd docusaurus-book
```

### 2. Install Dependencies

```bash
npm install
# or
yarn install
```

### 3. Environment Setup

Create a `.env` file in the root directory with the following variables:

```env
DATABASE_URL=your_database_connection_string
AUTH_SECRET=your_auth_secret_key
NODE_ENV=development
```

### 4. Initialize the Database

```bash
# Run database migrations
npm run db:migrate
# or
yarn db:migrate
```

### 5. Run the Development Server

```bash
npm run dev
# or
yarn dev
```

The application will be available at `http://localhost:3000`.

## Project Structure

```
physical-ai-robotics-docs/
├── blog/
│   └── 2025-01-01-welcome.md
│
├── docs/
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
│
├── src/
│   ├── components/
│   │   ├── Hero.js
│   │   └── Callout.js
│   │
│   ├── css/
│   │   └── custom.css              # Default Docusaurus CSS override
│   │
│   ├── pages/
│   │   ├── index.js                # Home landing page
│   │   └── robots.js               # Optional extra pages
│   │
│   └── theme/
│       └── Navbar/
│           └── Logo.js
│
├── static/
│   ├── img/
│   │   ├── humanoid.png
│   │   ├── ros2.png
│   │   ├── gazebo.png
│   │   └── isaac.png
│   └── files/
│       └── syllabus.pdf
│
├── i18n/
│   ├── en/
│   │   └── docusaurus-plugin-content-docs/
│   └── ur/
│       └── docusaurus-plugin-content-docs/
│
├── docusaurus.config.js
├── sidebars.js
├── package.json
├── package-lock.json
├── README.md
└── .gitignore

```

## Adding New Content

### Creating a New Lesson

1. Create a new directory in the appropriate module folder:
   ```
   docs/module-1/lesson-5/
   ```

2. Add an `index.md` file in the lesson directory with the content:
   ```md
   ---
   title: Lesson Title
   sidebar_position: 5
   description: Brief description of the lesson
   ---

   # Lesson Title

   Content goes here...
   ```

3. Update the `sidebars.js` file to include the new lesson in the navigation.

### Creating a New Module

1. Create a new directory in the `docs/` folder:
   ```
   docs/module-2/
   ```

2. Add an `index.md` file in the module directory:
   ```md
   ---
   title: Module Title
   sidebar_position: 2
   ---

   import DocCardList from '@theme/DocCardList';
   import { useCurrentSidebarCategory } from '@docusaurus/theme-common';

   # Module Title

   <DocCardList items={useCurrentSidebarCategory().items}/>
   ```

3. Add lessons to the module following the lesson creation process above.

## Internationalization (i18n)

The application supports both English and Urdu. To add translations:

1. Navigate to the `i18n/` directory
2. Add or update translation files in either the `en/` or `ur/` folders
3. Translation files follow the structure:
   ```
   i18n/
   ├── en/
   │   └── docusaurus-theme-classic/
   │       └── navbar.json
   └── ur/
       └── docusaurus-theme-classic/
           └── navbar.json
   ```

## API Endpoints

The application provides several API endpoints for functionality like progress tracking and authentication. See the API contracts document for full details.

## Running in Production

```bash
# Build the static site
npm run build
# or
yarn build

# Serve the built site
npm run serve
# or
yarn serve
```

## Testing

Run the test suite to ensure everything works correctly:

```bash
npm test
# or
yarn test
```

For detailed component testing:

```bash
npm run test:components
# or
yarn test:components
```

For end-to-end testing:

```bash
npm run test:e2e
# or
yarn test:e2e
```

## Common Issues

### Port Already in Use

If you get an error about the port being in use, you can specify a different port:

```bash
npm run dev -- --port 3001
# or
yarn dev --port 3001
```

### Dependency Errors

If you encounter dependency errors, try clearing the node_modules and reinstalling:

```bash
rm -rf node_modules package-lock.json
npm install
# or
rm -rf node_modules yarn.lock
yarn install
```