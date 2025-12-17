# Data Model: Premium Homepage Features Section

## Feature Entity

Represents a single feature card in the homepage features section.

### Properties

- **title** (string): The display title of the feature
  - Required: Yes
  - Validation: Non-empty string, max 100 characters
  - Example: "AI-Powered Learning"

- **description** (string): The detailed description of the feature
  - Required: Yes
  - Validation: Non-empty string, max 300 characters
  - Example: "Experience interactive learning with artificial intelligence guidance"

- **svg** (string): Path to the SVG icon file
  - Required: Yes
  - Validation: Valid path to SVG file in static/img directory
  - Example: "/img/ai-learning-icon.svg"

- **link** (string): URL for the "Learn More" button
  - Required: Yes
  - Validation: Valid URL or relative path
  - Example: "/docs/ai-learning"

- **isAI** (boolean): Indicates if the feature is AI-powered
  - Required: Yes
  - Validation: Boolean value
  - Example: true

- **color** (string): Hex color code for the feature's theme
  - Required: Yes
  - Validation: Valid hex color code (e.g., "#FF6B6B")
  - Example: "#4ECDC4"

### Relationships

- The Feature entity is contained within the FeatureList array
- Each Feature is rendered by the Feature component

## FeatureList Entity

Represents the collection of features to be displayed in the section.

### Properties

- **features** (array of Feature): The array of feature objects
  - Required: Yes
  - Validation: Must contain exactly 6 Feature objects
  - Example: [Feature, Feature, Feature, Feature, Feature, Feature]

## State Transitions

- Hover state: Feature card transforms (lifts, scales) when user hovers over it
- Theme state: Feature appearance changes based on light/dark mode preference
- Animation state: AI badge pulses and one card floats when in view