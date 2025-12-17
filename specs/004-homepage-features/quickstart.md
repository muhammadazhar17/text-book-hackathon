# Quickstart: Premium Homepage Features Section

## Setup Instructions

### 1. Create the Component Directory
```bash
mkdir -p src/components/HomepageFeatures
```

### 2. Create the FeatureList Data
Create `src/components/HomepageFeatures/FeatureList.tsx` with the 6 feature objects:

```typescript
// FeatureList.tsx
import { FeatureItem } from './types';

export const FeatureList: FeatureItem[] = [
  {
    title: 'AI-Powered Learning',
    description: 'Experience interactive learning with artificial intelligence guidance',
    svg: '/img/ai-learning-icon.svg',
    link: '/docs/ai-learning',
    isAI: true,
    color: '#4ECDC4'
  },
  {
    title: 'Robotic Simulation',
    description: 'Test your code in realistic robotic environments before deployment',
    svg: '/img/robot-sim-icon.svg',
    link: '/docs/robotic-simulation',
    isAI: true,
    color: '#FF6B6B'
  },
  {
    title: 'Hardware Integration',
    description: 'Connect your code directly to physical robotics platforms',
    svg: '/img/hardware-icon.svg',
    link: '/docs/hardware-integration',
    isAI: false,
    color: '#45B7D1'
  },
  {
    title: 'Real-time Analytics',
    description: 'Monitor your robot\'s performance with comprehensive dashboards',
    svg: '/img/analytics-icon.svg',
    link: '/docs/analytics',
    isAI: false,
    color: '#96DEDA'
  },
  {
    title: 'Collaborative Coding',
    description: 'Work together on robotics projects with version control',
    svg: '/img/collab-icon.svg',
    link: '/docs/collaboration',
    isAI: false,
    color: '#F9CA24'
  },
  {
    title: 'Safety Protocols',
    description: 'Built-in safety checks to protect your robotics projects',
    svg: '/img/safety-icon.svg',
    link: '/docs/safety',
    isAI: false,
    color: '#6C5CE7'
  }
];
```

### 3. Create TypeScript Types
Create `src/components/HomepageFeatures/types.ts`:

```typescript
// types.ts
export interface FeatureItem {
  title: string;
  description: string;
  svg: string;
  link: string;
  isAI: boolean;
  color: string;
}
```

### 4. Create the Feature Component
Create `src/components/HomepageFeatures/Feature.tsx`:

```tsx
// Feature.tsx
import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useIsBrowser from '@docusaurus/useIsBrowser';
import { FeatureItem } from './types';

import styles from './styles.module.css';

interface FeatureProps {
  feature: FeatureItem;
  isFloating?: boolean;
}

export default function Feature({ feature, isFloating = false }: FeatureProps): JSX.Element {
  const isBrowser = useIsBrowser();

  // In browser, we'll load the SVG dynamically
  // In SSR, we'll render a placeholder
  const SvgComponent = isBrowser ? React.lazy(() => import(feature.svg)) : null;

  return (
    <div
      className={clsx(
        styles.featureCard,
        isFloating && styles.floating
      )}
      style={{ '--feature-color': feature.color } as React.CSSProperties}
    >
      <div className={styles.cardContent}>
        <div className={styles.svgContainer}>
          {SvgComponent ? (
            <React.Suspense fallback={<div className={styles.svgPlaceholder}>SVG</div>}>
              <SvgComponent className={styles.featureSvg} />
            </React.Suspense>
          ) : (
            <div className={styles.svgPlaceholder}>SVG</div>
          )}
        </div>

        {feature.isAI && (
          <div className={styles.aiBadge}>
            🤖 AI-Powered
          </div>
        )}

        <h3 className={styles.featureTitle}>
          {feature.title}
        </h3>

        <p className={styles.featureDescription}>
          {feature.description}
        </p>

        <Link
          to={feature.link}
          className={styles.learnMoreButton}
        >
          Learn More →
        </Link>
      </div>
    </div>
  );
}
```

### 5. Create the Main Component
Create `src/components/HomepageFeatures/HomepageFeatures.tsx`:

```tsx
// HomepageFeatures.tsx
import React from 'react';
import clsx from 'clsx';
import { FeatureList } from './FeatureList';

import Feature from './Feature';
import styles from './styles.module.css';

export default function HomepageFeatures(): JSX.Element {
  // Mark one of the AI features as floating (e.g., the first AI-powered one)
  const featuresWithFloating = FeatureList.map((feature, index) => {
    // Make the first AI-powered feature float
    const shouldFloat = feature.isAI && !FeatureList.slice(0, index).some(f => f.isAI);

    return { feature, isFloating: shouldFloat };
  });

  return (
    <section className={clsx(styles.featuresSection, 'margin-vert--lg')}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.sectionTitle}>Explore Physical AI</h2>
          </div>
        </div>
        <div className={styles.featuresGrid}>
          {featuresWithFloating.map(({ feature, isFloating }, index) => (
            <div key={index} className={styles.featureCol}>
              <Feature feature={feature} isFloating={isFloating} />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
```

### 6. Create CSS Module
Create `src/components/HomepageFeatures/styles.module.css`:

```css
/* styles.module.css */
.featuresSection {
  position: relative;
  padding: 4rem 0;
  overflow: hidden;
}

.featuresSection::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(
    circle at center,
    var(--ifm-color-emphasis-100) 0%,
    transparent 70%
  );
  animation: gradientShift 15s ease-in-out infinite;
  z-index: -2;
}

@keyframes gradientShift {
  0%, 100% {
    transform: scale(1);
    opacity: 0.3;
  }
  50% {
    transform: scale(1.2);
    opacity: 0.5;
  }
}

.sectionTitle {
  text-align: center;
  margin-bottom: 3rem;
  font-size: 2.5rem;
  background: linear-gradient(90deg, var(--ifm-color-primary), var(--ifm-color-secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.featuresGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 2rem;
}

.featureCol {
  display: flex;
  justify-content: center;
}

.featureCard {
  position: relative;
  width: 100%;
  max-width: 380px;
  min-height: 380px;
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 24px;
  padding: 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  transition: all 0.3s ease;
  overflow: hidden;
  z-index: 1;
}

.featureCard::before {
  content: '';
  position: absolute;
  top: -2px;
  left: -2px;
  right: -2px;
  bottom: -2px;
  background: linear-gradient(45deg,
    var(--feature-color),
    var(--ifm-color-primary),
    var(--ifm-color-secondary)
  );
  border-radius: 26px;
  z-index: -1;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.featureCard:hover {
  transform: translateY(-10px) scale(1.03);
  box-shadow: 0 20px 30px rgba(0, 0, 0, 0.2);
  background: rgba(255, 255, 255, 0.2);
}

.featureCard:hover::before {
  opacity: 1;
}

.featureCard:hover::after {
  content: '';
  position: absolute;
  top: -5px;
  left: -5px;
  right: -5px;
  bottom: -5px;
  border-radius: 28px;
  background: linear-gradient(45deg,
    var(--feature-color),
    var(--ifm-color-primary),
    var(--ifm-color-secondary),
    var(--feature-color)
  );
  z-index: -2;
  animation: rotate 3s linear infinite;
  filter: blur(10px);
}

.floating {
  animation: float 4s ease-in-out infinite;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-15px);
  }
}

@keyframes rotate {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.cardContent {
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  z-index: 2;
}

.svgContainer {
  margin-bottom: 1.5rem;
  width: 120px;
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.featureSvg {
  width: 100%;
  height: 100%;
  filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
}

.featureSvg path,
.featureSvg circle,
.featureSvg rect {
  transition: filter 0.3s ease;
}

.featureCard:hover .featureSvg path,
.featureCard:hover .featureSvg circle,
.featureCard:hover .featureSvg rect {
  filter: drop-shadow(0 6px 8px var(--feature-color));
}

.aiBadge {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: linear-gradient(45deg, var(--feature-color), var(--ifm-color-primary));
  color: white;
  padding: 0.3rem 0.8rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: bold;
  z-index: 3;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(var(--feature-color-rgb), 0.4);
  }
  50% {
    transform: scale(1.05);
    box-shadow: 0 0 0 10px rgba(var(--feature-color-rgb), 0);
  }
}

.featureTitle {
  font-size: 1.5rem;
  margin: 0 0 1rem 0;
  background: linear-gradient(90deg, var(--ifm-color-primary), var(--feature-color));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.4;
}

.featureDescription {
  flex-grow: 1;
  margin: 0 0 1.5rem 0;
  color: var(--ifm-color-emphasis-700);
  line-height: 1.6;
}

.learnMoreButton {
  display: inline-block;
  padding: 0.7rem 1.5rem;
  background: transparent;
  border: 2px solid var(--feature-color);
  color: var(--feature-color);
  border-radius: 24px;
  text-decoration: none;
  font-weight: 600;
  transition: all 0.3s ease;
  align-self: center;
}

.learnMoreButton:hover {
  background: var(--feature-color);
  color: white;
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

/* Dark mode overrides */
html[data-theme='dark'] .featureCard {
  background: rgba(30, 30, 30, 0.25);
}

html[data-theme='dark'] .featureCard:hover {
  background: rgba(30, 30, 30, 0.4);
}

html[data-theme='dark'] .featureDescription {
  color: var(--ifm-color-emphasis-600);
}

/* Responsive breakpoints */
@media (max-width: 996px) {
  .featuresGrid {
    grid-template-columns: 1fr;
  }

  .sectionTitle {
    font-size: 2rem;
  }
}

@media (max-width: 768px) {
  .featureCard {
    max-width: 100%;
    min-height: auto;
    padding: 1.5rem;
  }

  .svgContainer {
    width: 100px;
    height: 100px;
  }

  .sectionTitle {
    font-size: 1.8rem;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .featureCard,
  .featureCard:hover,
  .floating,
  .aiBadge {
    animation: none;
    transition: none;
    transform: none;
  }

  .featuresSection::before {
    animation: none;
  }
}
```

### 7. Integrate with Homepage
Update your `src/pages/index.js` to include the new component:

```jsx
// src/pages/index.js
import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  // Your existing header code
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        {/* Your other homepage content */}
      </main>
    </Layout>
  );
}
```

### 8. Add SVG Icons
Place your SVG icons in the `static/img/` directory with appropriate names that match the paths in your FeatureList.

### 9. Testing
Run your Docusaurus site to verify the component works:
```bash
npm run start
```