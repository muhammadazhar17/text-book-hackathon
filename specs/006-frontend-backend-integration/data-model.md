# Data Model: Frontend ↔ Backend Integration

## Chat Message Entity

**Definition**: Represents a single message in the conversation

**Fields**:
- `id`: string (unique identifier for the message)
- `sender`: enum ("user" | "agent") - indicates who sent the message
- `content`: string - the actual message content
- `timestamp`: datetime - when the message was created
- `status`: enum ("sent" | "delivered" | "error") - delivery status for user messages

**Validation Rules**:
- `content` must not be empty or exceed 2000 characters
- `sender` must be one of the allowed values
- `timestamp` must be in ISO 8601 format

## Chat Session Entity

**Definition**: Represents the current conversation context in the browser session

**Fields**:
- `sessionId`: string (unique identifier for the session)
- `messages`: array of ChatMessage - ordered list of messages in the conversation
- `createdAt`: datetime - when the session was started
- `lastActive`: datetime - when the last message was sent/received

**Validation Rules**:
- `messages` array must not exceed 50 messages (to prevent memory issues)
- `createdAt` and `lastActive` must be in ISO 8601 format

## API Request Entity

**Definition**: Structure for requests sent from frontend to backend

**Fields**:
- `message`: string - the user's query to send to the RAG agent
- `sessionId`: string (optional) - to maintain conversation context

**Validation Rules**:
- `message` must not be empty and should be between 1 and 2000 characters
- `sessionId` if provided, must be a valid session identifier

## API Response Entity

**Definition**: Structure for responses received from backend to frontend

**Fields**:
- `response`: string - the RAG agent's answer to the user's query
- `sources`: array of strings (optional) - list of source documents referenced
- `error`: string (optional) - error message if the request failed

**Validation Rules**:
- `response` must not be empty when no error is present
- `sources` array, if present, must contain valid document references
- `error` field is mutually exclusive with `response` field

## State Management Schema

**Definition**: Structure for local state management in the frontend component

**Fields**:
- `isOpen`: boolean - whether the chat panel is currently open
- `isLoading`: boolean - whether a response is currently being loaded
- `messages`: array of ChatMessage - current conversation messages
- `inputValue`: string - current value in the input field
- `error`: string (optional) - any error message to display

**Validation Rules**:
- `isLoading` and `isOpen` states must be consistent with UI behavior
- `messages` must follow the ChatMessage structure
- `inputValue` must be less than 2000 characters