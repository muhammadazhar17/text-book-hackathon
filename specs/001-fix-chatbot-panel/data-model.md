# Data Model: Chatbot Panel State

## Entities

### ChatPanelState
Represents the current state of the chat panel that needs to be fixed.

- **isPanelOpen** (boolean): Indicates whether the chat panel is currently visible or hidden
- **panelVisibility** (enum: 'visible' | 'hidden' | 'minimized'): More detailed visibility state of the panel
- **lastInteractionTime** (Date): Timestamp of the last user interaction (open/close)
- **buttonClickCount** (number): Count of button clicks (for debugging rapid clicking issues)

## State Transitions

### ChatPanelState Transitions
- `hidden` → `visible` (when floating button is clicked and panel should open)
- `visible` → `hidden` (when close button is clicked or toggled off)
- `visible` → `minimized` (when panel is minimized)
- `minimized` → `visible` (when panel is expanded)

## Validation Rules

### ChatPanelState Validation
- `isPanelOpen` must be consistent with `panelVisibility` value
- `lastInteractionTime` must be a valid date/time
- `buttonClickCount` must be a non-negative integer

## Relationships

- The ChatPanelState is managed within the ChatWidget component state
- The ChatButton component triggers updates to ChatPanelState
- The ChatPanel component renders based on ChatPanelState values