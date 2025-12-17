---
id: 1
title: chatbot-integration-adr
stage: plan
date: 2025-12-10
surface: agent
model: Qwen
feature: chatbot-integration
branch: 003-chatbot-integration
user: cw
command: sp.adr
labels: ["architecture", "decision", "chatbot", "docusaurus", "api"]
links:
  spec: "C:/Users/cw/Desktop/hackta_one/physicalai-and-humanoids-robotics-book/specs/003-chatbot-integration/spec.md"
  ticket: null
  adr: "C:/Users/cw/Desktop/hackta_one/physicalai-and-humanoids-robotics-book/history/adr/001-docusaurus-chatbot-integration-approach.md"
  pr: null
files:
 - C:/Users/cw/Desktop/hackta_one/physicalai-and-humanoids-robotics-book/history/adr/001-docusaurus-chatbot-integration-approach.md
 - C:/Users/cw/Desktop/hackta_one/physicalai-and-humanoids-robotics-book/specs/003-chatbot-integration/plan.md
 - C:/Users/cw/Desktop/hackta_one/physicalai-and-humanoids-robotics-book/specs/003-chatbot-integration/research.md
 - C:/Users/cw/Desktop/hackta_one/physicalai-and-humanoids-robotics-book/specs/003-chatbot-integration/contracts/chat-api.yaml
tests:
 - C:/Users/cw/Desktop/hackta_one/physicalai-and-humanoids-robotics-book/history/adr/001-docusaurus-chatbot-integration-approach.md
---

## Prompt

Write an Architecture Decision Record (ADR) explaining:

Context

Backend chatbot already built and functional.

Frontend is a Docusaurus site.

Need seamless communication between both.

Decision

Use REST API POST requests for sending/receiving chat messages.

Implement chatbot as a floating UI widget included globally.

Use Axios/fetch for communication.

Alternatives Considered

WebSockets (not needed for simple chat)

Inline embedding (not flexible)

External chat service (no control)

Consequences

Simple to implement

Easy deployment

Works across all pages

Low maintenance

## Response snapshot

# ADR-001: Docusaurus Chatbot Integration Approach

## Status

Accepted

## Date

2025-12-10

## Context

We have a functional backend chatbot that needs to be integrated into a Docusaurus documentation site. The frontend is a Docusaurus-based documentation site built with React. The goal is to provide seamless communication between the frontend and backend while maintaining good user experience and following Docusaurus best practices.

The chatbot needs to be accessible from every page of the documentation site, with a persistent UI element that users can interact with to ask questions about physical AI and robotics.

## Decision

We will implement the chatbot integration using the following approach:

**API Communication**: Use REST API POST requests for sending and receiving chat messages between the Docusaurus frontend and the backend chatbot. This approach uses standard HTTP protocols and is well-supported across all environments.

**UI Implementation**: Implement the chatbot as a floating UI widget that appears on every page of the Docusaurus site. The widget will include:
- A floating button in the bottom-left corner of each page
- A collapsible chat panel that opens when the button is clicked
- Message history display
- User input field for sending messages

**Technology Stack**: Use Axios or the native fetch API for communication with the backend, following standard web practices for frontend-backend communication.

## Alternatives Considered

1. **WebSockets**: 
   - Pros: Real-time bidirectional communication, more efficient for frequent updates
   - Cons: More complex to implement, requires persistent connections, may face issues with proxies/load balancers, not needed for simple chat functionality

2. **Inline Embedding**:
   - Pros: Direct integration, potentially simpler DOM management
   - Cons: Less flexible positioning, harder to maintain persistent state across page changes, more intrusive to existing page layout

3. **External Chat Service (e.g., Chatwoot, Intercom)**:
   - Pros: Pre-built solution, additional features, analytics
   - Cons: Loss of control, potential privacy concerns, additional costs, dependency on third-party service, may not integrate well with domain-specific knowledge

4. **Server-Sent Events (SSE)**:
   - Pros: Good for one-way streaming from server to client
   - Cons: Not ideal for bidirectional chat interactions, less standard than REST APIs

## Consequences

**Positive**:
- Simple to implement and maintain
- Easy deployment and troubleshooting
- Works consistently across all pages in the Docusaurus site
- Familiar technology stack for developers
- Good compatibility with various hosting platforms (Vercel, Netlify, etc.)
- Low maintenance overhead
- Clear separation between frontend and backend responsibilities

**Negative**:
- May have slightly higher latency than real-time solutions like WebSockets
- Requires handling of state management on the client side
- Each message requires a separate API call, which could result in more network requests

**Risks**:
- Backend API availability directly impacts chat functionality
- Potential CORS issues when connecting to external backend
- Message history is not persisted across browser sessions unless implemented separately

## References

- Feature specification: `specs/003-chatbot-integration/spec.md`
- Implementation plan: `specs/003-chatbot-integration/plan.md`
- API contract: `specs/003-chatbot-integration/contracts/chat-api.yaml`
- Research findings: `specs/003-chatbot-integration/research.md`

## Outcome

- ✅ Impact: Created a comprehensive ADR documenting the architectural decision for chatbot integration approach
- 🧪 Tests: The ADR includes alternatives considered with pros and cons for each
- 📁 Files: Created ADR file in history/adr directory with proper structure and references
- 🔁 Next prompts: Ready for implementation planning with clear architectural direction
- 🧠 Reflection: The ADR properly documents the decision-making process and consequences

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Proceed with implementation tasks