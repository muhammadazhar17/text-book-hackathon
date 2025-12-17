import React from 'react';
import Link from '@docusaurus/Link';
import {useColorMode} from '@docusaurus/theme-common';
import clsx from 'clsx';

import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Physical AI Concepts',
    description: 'Explore the fundamental principles of Physical AI and why it matters for the future of robotics.',
    link: '/docs/why-physical-ai-matters',
    isAI: true,
    color: '#667eea',
  },
  {
    title: 'ROS 2 Integration',
    description: 'Learn how to integrate your robotics projects with the Robot Operating System 2 framework.',
    link: '/docs/module-1-ros2-nervous-system/overview',
    isAI: false,
    color: '#48bb78',
  },
  {
    title: 'Humanoid Robotics',
    description: 'Discover the latest advances in humanoid robotics and vision-language-action systems.',
    link: '/docs/module-4-vision-language-action/overview',
    isAI: true,
    color: '#ed8936',
  },
  {
    title: 'Computer Vision',
    description: 'Understand perception systems and computer vision techniques for robotics applications.',
    link: '/docs/module-2-digital-twin-simulation/overview',
    isAI: true,
    color: '#9f7aea',
  },
  {
    title: 'Motion Planning',
    description: 'Master motion planning and control algorithms for robotic systems.',
    link: '/docs/module-3-nvidia-isaac-ai-brain/overview',
    isAI: false,
    color: '#4299e1',
  },
  {
    title: 'AI in Robotics',
    description: 'Implement AI-powered chatbots that can interact with and control robotic systems.',
    link: '/docs/module-4-vision-language-action/overview',
    isAI: true,
    color: '#f56565',
  },
];

function FeatureCard({ feature, isFloating = false }) {
  const { color, title, description, link, isAI } = feature;
  const { isDarkTheme } = useColorMode();

  // Convert hex color to RGB for use in CSS variables
  const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  };

  const rgbColor = hexToRgb(color);
  const rgbColorStr = rgbColor ? `${rgbColor.r}, ${rgbColor.g}, ${rgbColor.b}` : '102, 126, 234';

  return (
    <div
      className={clsx(
        styles.featureCard,
        isFloating && styles.floating
      )}
      style={{
        '--feature-color': color,
        '--feature-color-rgb': rgbColorStr
      }}
    >
      {isAI && (
        <div className={styles.aiBadge}>
          🤖 AI-Powered
        </div>
      )}

      <div className={styles.cardContent}>
        <div className={styles.svgContainer}>
          <div className={styles.svgPlaceholder} style={{ color: color }}>
            {/* This would be replaced with actual SVGs in a real implementation */}
            <svg
              className={styles.featureSvg}
              viewBox="0 0 120 120"
              xmlns="http://www.w3.org/2000/svg"
            >
              <rect x="10" y="10" width="100" height="100" rx="10" fill="none" stroke="currentColor" strokeWidth="4"/>
              <circle cx="60" cy="60" r="25" fill="currentColor" opacity="0.2"/>
              <path d="M40,60 L80,60 M60,40 L60,80" stroke="currentColor" strokeWidth="4" strokeLinecap="round"/>
            </svg>
          </div>
        </div>

        <h3 className={styles.featureTitle}>
          {title}
        </h3>

        <p className={styles.featureDescription}>
          {description}
        </p>

        <Link
          to={link}
          className={styles.learnMoreButton}
        >
          Learn More →
        </Link>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  // Make the AI Chatbots card float (it's the last one in our list)
  const featuresWithFloating = FeatureList.map((feature, index) => ({
    feature,
    isFloating: index === 5 // AI Chatbots is at index 5
  }));

  return (
    <section className={clsx(styles.featuresSection, 'margin-vert--lg')}>
      <div className="container">
        <div className="row">
          <div className="col col--12">
            <h2 className={styles.sectionTitle}>
              Explore Physical AI
            </h2>
          </div>
        </div>
        <div className={styles.featuresGrid}>
          {featuresWithFloating.map(({ feature, isFloating }, index) => (
            <div key={index} className={styles.featureCol}>
              <FeatureCard feature={feature} isFloating={isFloating} />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}