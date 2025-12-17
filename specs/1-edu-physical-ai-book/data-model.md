# Data Model: Physical AI & Humanoid Robotics Course Book

## Overview
This document outlines the data models for the Physical AI & Humanoid Robotics Course Book application. These models represent the core entities identified in the feature specification and support the required functionality for student progress tracking, authentication, and content management.

## User
Represents a user of the course book (student, educator, or admin).

### Fields
- `id` (string, required): Unique identifier for the user
- `email` (string, required): User's email address for authentication
- `name` (string, required): User's full name
- `role` (string, required): User role (student, educator, admin)
- `locale` (string, optional): User's preferred language (default: "en")
- `created_at` (datetime, required): Timestamp of account creation
- `updated_at` (datetime, required): Timestamp of last update

### Validations
- Email must be unique and properly formatted
- Role must be one of: "student", "educator", "admin"
- Name must be between 2 and 100 characters

## Module
Represents a course module containing multiple lessons.

### Fields
- `id` (string, required): Unique identifier for the module
- `title` (string, required): Title of the module
- `description` (string, required): Detailed description of the module content
- `order` (integer, required): Order position of the module in the course
- `created_at` (datetime, required): Timestamp of module creation
- `updated_at` (datetime, required): Timestamp of last update

### Validations
- Title must be between 5 and 200 characters
- Order must be a positive integer
- Description must be between 10 and 5000 characters

## Lesson
Represents an individual lesson within a module.

### Fields
- `id` (string, required): Unique identifier for the lesson
- `module_id` (string, required): Reference to the parent module
- `title` (string, required): Title of the lesson
- `content_path` (string, required): Path to the lesson's content file
- `order` (integer, required): Order position of the lesson in the module
- `duration_estimate` (integer, optional): Estimated completion time in minutes
- `created_at` (datetime, required): Timestamp of lesson creation
- `updated_at` (datetime, required): Timestamp of last update

### Validations
- Title must be between 5 and 200 characters
- Module ID must reference an existing module
- Order must be a positive integer
- Content path must be a valid path format
- Duration estimate must be a positive integer if provided

## UserProgress
Tracks a user's progress through lessons and modules.

### Fields
- `id` (string, required): Unique identifier for the progress record
- `user_id` (string, required): Reference to the user
- `lesson_id` (string, required): Reference to the lesson being tracked
- `is_completed` (boolean, required): Whether the lesson is completed (default: false)
- `completion_date` (datetime, optional): Date when the lesson was completed
- `time_spent` (integer, optional): Time spent on the lesson in seconds
- `quiz_scores` (array of objects, optional): Scores from lesson quizzes
- `last_accessed` (datetime, required): Timestamp of last access to this lesson
- `created_at` (datetime, required): Timestamp of progress record creation
- `updated_at` (datetime, required): Timestamp of last update

### Validations
- User ID must reference an existing user
- Lesson ID must reference an existing lesson
- User-lesson combination must be unique
- Quiz scores must be an array of objects with 'quiz_id' and 'score' properties
- Score values must be between 0 and 100
- Time spent must be a positive integer if provided

## Quiz
Represents a quiz within a lesson.

### Fields
- `id` (string, required): Unique identifier for the quiz
- `lesson_id` (string, required): Reference to the lesson containing the quiz
- `title` (string, required): Title of the quiz
- `questions` (array of objects, required): Array of quiz questions
- `created_at` (datetime, required): Timestamp of quiz creation
- `updated_at` (datetime, required): Timestamp of last update

### Validations
- Lesson ID must reference an existing lesson
- Title must be between 5 and 200 characters
- Questions must be an array of objects with 'question', 'options', and 'correct_answer' properties
- Must have at least 1 question

## ContentReference
Stores metadata about content files and their relationships.

### Fields
- `id` (string, required): Unique identifier for the content reference
- `content_type` (string, required): Type of content ('module', 'lesson', 'resource')
- `content_id` (string, required): ID of the related content item
- `file_path` (string, required): Path to the content file
- `file_type` (string, required): Type of file ('md', 'mdx', 'pdf', 'image', 'video')
- `language` (string, required): Language code (default: 'en')
- `version` (string, optional): Content version
- `checksum` (string, optional): File integrity check
- `created_at` (datetime, required): Timestamp of creation
- `updated_at` (datetime, required): Timestamp of last update

### Validations
- Content type must be one of: 'module', 'lesson', 'resource'
- File path must be a valid path format
- File type must be one of: 'md', 'mdx', 'pdf', 'image', 'video', 'other'
- Language must be a valid ISO 639-1 code
- Version must follow semantic versioning format if provided

## Relationships
- User has many UserProgress records
- Module has many Lessons
- Lesson has many UserProgress records
- Lesson has many Quiz records
- UserProgress belongs to one User and one Lesson
- Quiz belongs to one Lesson
- ContentReference links to Modules, Lessons, or Resources