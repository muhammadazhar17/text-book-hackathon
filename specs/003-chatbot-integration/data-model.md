# Data Model: Chatbot Integration

## Entities

### ChatMessage
Represents a single message in the conversation between the user and the bot.

- **id** (string): Unique identifier for the message
- **content** (string): The actual text content of the message
- **sender** (enum: 'user' | 'bot'): Indicates whether the message was sent by the user or the bot
- **timestamp** (Date): When the message was created/sent
- **status** (enum: 'sent' | 'delivered' | 'error'): Status of message delivery (for user messages)

### ConversationSession
Represents a sequence of related chat messages that belong together.

- **id** (string): Unique identifier for the conversation session
- **messages** (ChatMessage[]): Array of messages in the conversation
- **createdAt** (Date): When the conversation was started
- **lastActiveAt** (Date): When the last message was sent in this conversation
- **isActive** (boolean): Whether the conversation is currently active

## State Transitions

### ChatMessage State Transitions
- `created` → `sent` (when message is sent to backend)
- `sent` → `delivered` (when backend confirms receipt)
- `sent` → `error` (when there's an error sending the message)

### ChatWidget State Transitions
- `hidden` → `visible` (when floating button is clicked)
- `visible` → `minimized` (when widget is minimized)
- `minimized` → `visible` (when widget is expanded)
- `visible` → `hidden` (when widget is closed)

## Validation Rules

### ChatMessage Validation
- `content` must be between 1 and 2000 characters
- `sender` must be either 'user' or 'bot'
- `timestamp` must be a valid date/time
- `status` is only applicable for user messages

### ConversationSession Validation
- `messages` array cannot exceed 100 messages (to prevent memory issues)
- Each message in `messages` must have a unique `id`
- `createdAt` must be before or equal to `lastActiveAt`

## Relationships

- A `ConversationSession` contains 0 or more `ChatMessage` objects
- Each `ChatMessage` belongs to exactly one `ConversationSession`