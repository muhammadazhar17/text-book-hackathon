---
description: "Task list for Gemini RAG Agent implementation"
---

# Tasks: Gemini RAG Agent with Qdrant Cloud Integration

**Input**: Design documents from `/specs/1-gemini-rag-agent/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: Implementation will be in `/backend/main.py` as a single file
- Paths shown below follow the single file requirement

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure with backend directory
- [x] T002 [P] Create requirements.txt with fastapi, uvicorn, google-generativeai, cohere, qdrant-client, python-dotenv
- [x] T003 [P] Create .env.example file with API key placeholders
- [x] T004 Create backend directory structure

---
## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 Set up environment variable loading in main.py
- [x] T006 [P] Initialize Google Gemini client in main.py
- [x] T007 [P] Initialize Cohere client in main.py
- [x] T008 [P] Initialize Qdrant client in main.py
- [x] T009 Create data models for Query, Document Chunk, Retrieved Context, and Response in main.py
- [x] T010 Set up basic FastAPI app structure in main.py
- [x] T011 Implement error handling and logging infrastructure in main.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---
## Phase 3: User Story 1 - Query Processing via API (Priority: P1) 🎯 MVP

**Goal**: Enable users to send a query to the API and receive a response, implementing the core endpoint

**Independent Test**: Can be fully tested by sending a query to the API endpoint and verifying that the response contains relevant information from the retrieved context, delivering a complete answer to the user's question.

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T012 [P] [US1] Contract test for /chat endpoint in backend/test_contract.py
- [ ] T013 [P] [US1] Integration test for query-response journey in backend/test_integration.py

### Implementation for User Story 1

- [x] T014 [P] [US1] Create /chat endpoint in main.py
- [x] T015 [US1] Implement query validation and sanitization in main.py
- [x] T016 [US1] Implement basic response structure in main.py
- [x] T017 [US1] Add API request/response models in main.py
- [x] T018 [US1] Add health check endpoint in main.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---
## Phase 4: User Story 2 - Context Retrieval from Knowledge Base (Priority: P1)

**Goal**: Implement the ability to retrieve relevant document chunks from Qdrant Cloud based on user queries

**Independent Test**: Can be tested by submitting a query and verifying that the system retrieves appropriate document chunks from the knowledge base that are relevant to the query.

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [ ] T019 [P] [US2] Contract test for retrieval functionality in backend/test_retrieval.py
- [ ] T020 [P] [US2] Integration test for retrieval journey in backend/test_integration.py

### Implementation for User Story 2

- [x] T021 [P] [US2] Implement query embedding generation using Cohere in main.py
- [x] T022 [US2] Implement Qdrant Cloud similarity search in main.py
- [x] T023 [US2] Implement top-k retrieval logic in main.py
- [x] T024 [US2] Create function to aggregate retrieved chunks into context in main.py
- [x] T025 [US2] Preserve source metadata during retrieval in main.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---
## Phase 5: User Story 3 - Response Generation with Context (Priority: P2)

**Goal**: Combine retrieved context with user queries to generate coherent, accurate responses using Gemini

**Independent Test**: Can be tested by providing a query and retrieved context to the language model and verifying that the generated response is coherent, relevant, and incorporates information from the context.

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [ ] T026 [P] [US3] Contract test for response generation in backend/test_generation.py
- [ ] T027 [P] [US3] Integration test for full RAG journey in backend/test_integration.py

### Implementation for User Story 3

- [x] T028 [P] [US3] Create prompt construction function in main.py
- [x] T029 [US3] Implement Gemini response generation in main.py
- [x] T030 [US3] Configure generation parameters (temperature, max tokens) in main.py
- [x] T031 [US3] Parse Gemini output and attach source references in main.py
- [x] T032 [US3] Ensure responses incorporate retrieved context in main.py

**Checkpoint**: All user stories should now be independently functional

---
## Phase 6: User Story 4 - Query Understanding and Processing (Priority: P2)

**Goal**: Process user queries to enable semantic matching with the knowledge base for effective information retrieval

**Independent Test**: Can be tested by sending a query and verifying that the system processes it appropriately for similarity matching.

### Tests for User Story 4 (OPTIONAL - only if tests requested) ⚠️

- [ ] T033 [P] [US4] Contract test for query processing in backend/test_query_processing.py
- [ ] T034 [P] [US4] Integration test for query processing journey in backend/test_integration.py

### Implementation for User Story 4

- [x] T035 [P] [US4] Implement query preprocessing and validation in main.py
- [x] T036 [US4] Ensure embedding compatibility with stored vectors in main.py
- [x] T037 [US4] Add query length and format validation in main.py
- [x] T038 [US4] Implement fallback logic for unsupported query types in main.py

**Checkpoint**: All user stories should now be independently functional

---
## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T039 [P] Documentation updates in backend/README.md
- [x] T040 Code cleanup and refactoring of main.py
- [x] T041 Performance optimization across all stories in main.py
- [ ] T042 [P] Additional unit tests in backend/test_unit.py
- [x] T043 Security hardening (input validation, rate limiting) in main.py
- [x] T044 Error handling for edge cases (empty retrieval, LLM failures) in main.py
- [x] T045 Run quickstart.md validation
- [x] T046 Final integration testing of complete RAG pipeline

---
## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Depends on US2 (retrieval) but should be independently testable
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---
## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for /chat endpoint in backend/test_contract.py"
Task: "Integration test for query-response journey in backend/test_integration.py"

# Launch all components for User Story 1 together:
Task: "Create /chat endpoint in main.py"
Task: "Implement query validation and sanitization in main.py"
```

---
## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: User Story 4
3. Stories complete and integrate independently

---
## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence