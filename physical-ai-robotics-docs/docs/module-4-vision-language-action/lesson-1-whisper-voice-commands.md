---
sidebar_position: 2
---

# Lesson 1: Whisper Voice Commands - Speech Recognition for Robotics

## Learning Objectives

By the end of this lesson, you will be able to:

1. Integrate OpenAI Whisper for real-time speech recognition in robotics
2. Implement noise reduction and preprocessing for robust voice command recognition
3. Design command parsing and validation systems for voice interfaces
4. Handle ambiguity and errors in voice command interpretation
5. Create multimodal feedback systems for voice command confirmation

## Introduction

Voice interfaces enable natural, intuitive interaction between humans and humanoid robots. OpenAI's Whisper model provides state-of-the-art speech recognition capabilities that can be deployed on robotic platforms to enable voice-controlled operation. This lesson explores how to implement robust voice command systems that work effectively in real-world environments with background noise, acoustic reflections, and varying speaker characteristics.

Effective voice command systems for humanoid robots must handle challenges not present in general-purpose speech recognition systems:

- **Acoustic environment**: Robot environments often have background noise from motors, fans, and other equipment
- **Real-time constraints**: Robot systems require quick response to user commands
- **Limited vocabulary**: Robots typically respond to specific commands, not general conversation
- **Feedback requirements**: Robots need to confirm understanding and execution of commands

## Whisper Model Overview

Whisper is a general-purpose speech recognition model developed by OpenAI that demonstrates strong performance across diverse datasets and languages. For robotics applications, Whisper offers several advantages:

- **Robustness**: Performs well with diverse accents and background noise
- **Multilingual support**: Can recognize speech in multiple languages
- **Open-source**: Available for deployment and customization
- **Accuracy**: High transcription accuracy even for technical language

### Whisper Architecture

Whisper uses a Transformer-based encoder-decoder architecture:

- **Encoder**: Processes audio spectrograms and extracts acoustic features
- **Decoder**: Generates text tokens based on encoder output and previous tokens
- **Multilingual capability**: Can handle speech in 100 different languages
- **Timestamp information**: Provides timing information for detected speech segments

### Model Variants

Whisper is available in several sizes with different performance and resource characteristics:

| Model | Size | Required VRAM | Relative Speed | Accuracy |
|-------|------|---------------|----------------|----------|
| tiny | 75MB | ~1GB | 32x | Lower |
| base | 142MB | ~1GB | 16x | Good |
| small | 465MB | ~2GB | 6x | Better |
| medium | 1.5GB | ~5GB | 2x | High |
| large | 3.0GB | ~10GB | 1x | Highest |

For robotics applications, the choice depends on the available hardware and required response time.

## Implementing Whisper in Robotics Systems

### Audio Input and Preprocessing

Robot systems require careful audio preprocessing to handle their unique acoustic environment:

```python
import pyaudio
import numpy as np
import torch
import whisper
from collections import deque
import webrtcvad
import time

class RobustVoiceCommandProcessor:
    def __init__(self, model_size="base.en", sample_rate=16000, vad_mode=2):
        # Initialize Whisper model
        self.model = whisper.load_model(model_size)
        self.sample_rate = sample_rate
        
        # Initialize audio interface
        self.audio = pyaudio.PyAudio()
        self.audio_buffer = deque(maxlen=sample_rate * 10)  # 10 seconds buffer
        self.chunk_size = 1024
        
        # Initialize Voice Activity Detection (VAD) to identify speech segments
        self.vad = webrtcvad.Vad(vad_mode)  # 0-3, higher = more aggressive VAD
        self.frame_duration = 30  # ms (multiple of 10)
        self.frame_size = int(sample_rate * self.frame_duration / 1000 * 2)  # 2 bytes per sample
        
        # Audio processing parameters
        self.noise_threshold = 0.01  # Minimum amplitude to consider signal
        self.silence_duration = 2.0  # Seconds of silence to trigger processing
        
        # Start audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # For silence detection
        self.last_speech_time = time.time()
        self.is_processing = False

    def capture_audio(self):
        """Capture audio from microphone with VAD preprocessing"""
        data = self.stream.read(self.chunk_size)
        audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        
        # Add to buffer
        self.audio_buffer.extend(audio_array)
        
        # Check if this audio chunk contains speech using VAD
        if len(audio_array) == int(self.sample_rate * self.frame_duration / 1000):
            # Convert to 16-bit integers for VAD
            audio_int16 = (audio_array * 32767).astype(np.int16)
            vad_decision = self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
            
            if vad_decision:
                self.last_speech_time = time.time()
        
        return audio_array

    def should_process_audio(self):
        """Determine if we should process the current audio buffer"""
        current_time = time.time()
        
        # Process if we have a significant amount of audio (>1 second) and there's been silence for a while
        if (len(self.audio_buffer) > self.sample_rate and 
            current_time - self.last_speech_time > self.silence_duration and 
            not self.is_processing):
            return True
        return False

    def process_voice_command(self):
        """Process buffered audio with Whisper for voice command recognition"""
        if not self.should_process_audio():
            return None

        self.is_processing = True
        try:
            # Convert buffer to format expected by Whisper
            if len(self.audio_buffer) < self.sample_rate:  # Need at least 1 second
                return None

            # Take the last N seconds of audio where N is the minimum of 
            # 10 seconds or the buffer length
            samples_to_take = min(int(self.sample_rate * 5), len(self.audio_buffer))
            audio_np = np.array(list(self.audio_buffer)[-samples_to_take:])
            
            # Ensure we have enough audio
            if len(audio_np) < self.sample_rate:
                return None

            # Run transcription using Whisper
            result = self.model.transcribe(audio_np)
            command_text = result["text"].strip()
            
            # Clear buffer after processing to avoid repetitive commands
            self.audio_buffer.clear()
            
            # Clean up the command text (remove punctuation, normalize)
            cleaned_command = self.clean_command_text(command_text)
            
            self.is_processing = False
            return cleaned_command
            
        except Exception as e:
            print(f"Error in voice command processing: {e}")
            self.is_processing = False
            return None

    def clean_command_text(self, text):
        """Clean up the transcribed text"""
        # Remove common punctuation that doesn't affect meaning
        import re
        cleaned = re.sub(r'[.!?]+', '', text)
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def continuous_listening_loop(self):
        """Continuous loop for voice command processing"""
        print("Starting voice command listening...")
        
        while True:
            # Capture audio
            self.capture_audio()
            
            # Check if we should process
            if self.should_process_audio():
                command = self.process_voice_command()
                if command:
                    print(f"Recognized command: {command}")
                    return command  # In practice, would be handled by callback system
            
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
```

### Optimizing Whisper for Robotics

For real-time robotics applications, several optimizations can improve performance:

```python
import whisper
from whisper.tokenizer import get_tokenizer
import torch

class OptimizedWhisperProcessor:
    def __init__(self, model_size="base.en"):
        # Load model with optimizations
        self.model = whisper.load_model(model_size).cuda().eval()
        
        # Initialize tokenizer for custom keyword recognition
        self.tokenizer = get_tokenizer(multilingual=False)  # Use English-specific tokenizer
        
        # Pre-define robot command tokens to improve accuracy
        self.robot_commands = [
            "go to the kitchen", "pick up the red ball", 
            "bring me coffee", "wave hello", "stop moving",
            "turn left", "turn right", "move forward", "move backward"
        ]
        
        # Initialize the task-specific decoder prefix
        self.enforced_decoder_prompt = self.tokenizer.encode(" " + " ".join([
            "go", "to", "the", "pick", "up", "bring", "me", "wave", 
            "hello", "stop", "turn", "left", "right", "move", "forward", 
            "backward", "kitchen", "living", "room", "coffee", "water"
        ]))

    def transcribe_with_options(self, audio):
        """Transcribe audio with robotics-specific options"""
        # Use specific options optimized for command recognition
        result = self.model.transcribe(
            audio,
            # Limit beam search to reduce computation
            beam_size=5,
            # Only generate tokens from our command vocabulary
            initial_prompt=" ".join(self.robot_commands[:5]),  # Use first 5 as prompt
            # Limit maximum tokens to prevent rambling
            max_new_tokens=20,
            # Set temperature to 0 for more deterministic output
            temperature=0.0,
            # Don't predict timestamps for command recognition
            word_timestamps=False
        )
        return result

    def batch_process_audio(self, audio_segments):
        """Process multiple audio segments efficiently"""
        # Batch multiple segments for more efficient processing
        # This is helpful when processing multiple possible commands
        results = []
        for segment in audio_segments:
            result = self.transcribe_with_options(segment)
            results.append(result)
        return results
```

## Command Parsing and Validation

After speech recognition, commands must be parsed and validated to determine the robot's intended action:

```python
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class ParsedCommand:
    action: str
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str

class CommandParser:
    def __init__(self):
        # Define command patterns with regex
        self.command_patterns = {
            'navigate': [
                r'go to the (kitchen|living room|bedroom|office|dining room|bathroom|garage)',
                r'go to (kitchen|living room|bedroom|office|dining room|bathroom|garage)',
                r'please navigate to the (kitchen|living room|bedroom|office|dining room|bathroom|garage)',
                r'head to the (kitchen|living room|bedroom|office|dining room|bathroom|garage)'
            ],
            'grasp': [
                r'pick up the (red|blue|green|small|large)?\s*(ball|cup|book|bottle|toy|phone|keys)',
                r'grab the (red|blue|green|small|large)?\s*(ball|cup|book|bottle|toy|phone|keys)',
                r'get the (red|blue|green|small|large)?\s*(ball|cup|book|bottle|toy|phone|keys)',
                r'take the (red|blue|green|small|large)?\s*(ball|cup|book|bottle|toy|phone|keys)'
            ],
            'bring': [
                r'bring me the (water|coffee|phone|book|keys)',
                r'bring the (water|coffee|phone|book|keys) to me',
                r'can you get me the (water|coffee|phone|book|keys)',
                r'could you bring me the (water|coffee|phone|book|keys)'
            ],
            'greet': [
                r'(wave hello|wave|say hello|greet me|hello)',
                r'(wave to me|wave hi)',
                r'(perform greeting|greeting)'
            ],
            'follow': [
                r'follow me',
                r'follow',
                r'come with me',
                r'come along',
                r'come after me'
            ],
            'stop': [
                r'(stop|halt|wait|pause|cease)',
                r'please stop',
                r'stop moving'
            ]
        }

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        """Parse natural language command into structured action"""
        text_lower = text.lower().strip()
        
        for action, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Extract named groups or positional groups
                    groups = match.groups()
                    if groups:
                        # Filter out None values and empty strings
                        params = [g for g in groups if g is not None and g != '']
                        if params:
                            # Create parameter dictionary based on context
                            parameters = self.create_parameters(action, params)
                        else:
                            parameters = {}
                    else:
                        parameters = {}

                    confidence = self.estimate_confidence(text, match)
                    
                    return ParsedCommand(
                        action=action,
                        parameters=parameters,
                        confidence=confidence,
                        raw_text=text
                    )
        
        return None

    def create_parameters(self, action: str, groups: List[str]) -> Dict[str, Any]:
        """Create parameter dictionary based on action and matched groups"""
        params = {}
        
        if action == 'navigate':
            params['destination'] = groups[0]
        elif action == 'grasp':
            # Handle optional color qualifier and required object
            if len(groups) == 2 and groups[0] is not None:
                params['color'] = groups[0].strip()
                params['object'] = groups[1].strip()
            elif len(groups) == 1:
                params['object'] = groups[0].strip()
        elif action == 'bring':
            params['item'] = groups[0].strip()
        
        return params

    def estimate_confidence(self, text: str, match) -> float:
        """Estimate confidence in command recognition"""
        if not match:
            return 0.0
            
        # Calculate confidence based on multiple factors:
        # 1. How much of the text matched
        match_length = len(match.group(0)) if match else 0
        text_length = len(text) if text else 1
        text_confidence = match_length / text_length
        
        # 2. How specific was the match (longer patterns tend to be more specific)
        pattern_specificity = min(len(match.group(0)) / 5.0, 1.0)  # Normalize to 0-1
        
        # Combine factors
        confidence = (text_confidence * 0.7 + pattern_specificity * 0.3)
        
        return min(confidence, 1.0)  # Clamp to 0-1 range

    def validate_command(self, parsed_command: ParsedCommand, robot_capabilities: Dict[str, Any]) -> bool:
        """Validate that parsed command is executable by the robot"""
        action = parsed_command.action
        params = parsed_command.parameters
        
        # Check if robot supports the action
        if action not in robot_capabilities.get('actions', []):
            return False
            
        # Check if required parameters are available and valid
        if action == 'navigate':
            destination = params.get('destination')
            if destination and destination in robot_capabilities.get('known_locations', []):
                return True
        elif action == 'grasp':
            obj = params.get('object')
            if obj and obj in robot_capabilities.get('graspable_objects', []):
                return True
        elif action == 'bring':
            item = params.get('item')
            if item and item in robot_capabilities.get('bringable_items', []):
                return True
        else:
            # For other actions, assume valid if in supported actions
            return True
            
        return False
```

## Error Handling and Disambiguation

Robust voice command systems must handle ambiguity and errors gracefully:

```python
import threading
import time
from queue import Queue

class VoiceCommandHandler:
    def __init__(self, robot_capabilities: Dict[str, Any]):
        self.processor = RobustVoiceCommandProcessor()
        self.parser = CommandParser()
        self.robot_capabilities = robot_capabilities
        
        # Command queue for processing
        self.command_queue = Queue()
        self.response_queue = Queue()
        
        # History for context
        self.command_history = deque(maxlen=10)
        
        # State management
        self.is_listening = True
        self.requires_confirmation = True  # Whether to ask for confirmation
        
        # Threading
        self.processing_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.processing_thread.start()

    def start_listening_loop(self):
        """Start the continuous listening loop"""
        print("Starting voice command system...")
        
        while self.is_listening:
            try:
                # Capture and process audio
                command_text = self.processor.process_voice_command()
                if command_text:
                    self.handle_command_text(command_text)
                    
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
            except KeyboardInterrupt:
                print("Stopping voice command system...")
                self.is_listening = False
            except Exception as e:
                print(f"Error in listening loop: {e}")
                time.sleep(1)  # Longer delay on error

    def handle_command_text(self, command_text: str):
        """Handle recognized command text"""
        print(f"Recognized: {command_text}")
        
        # Parse the command
        parsed_command = self.parser.parse_command(command_text)
        if not parsed_command:
            self.request_clarification(command_text)
            return

        print(f"Parsed command: {parsed_command.action} with params {parsed_command.parameters}")
        
        # Validate confidence
        if parsed_command.confidence < 0.6:  # Threshold for confidence
            if self.requires_confirmation:
                self.request_confirmation(command_text, parsed_command)
                return
            else:
                # If no confirmation required, set low confidence command as lower priority
                parsed_command.confidence = 0.5

        # Validate the command against robot capabilities
        if not self.parser.validate_command(parsed_command, self.robot_capabilities):
            self.request_clarification(command_text)
            return

        # Add to execution queue
        self.command_queue.put(parsed_command)
        self.command_history.append(parsed_command)

    def request_clarification(self, command_text: str):
        """Request user to clarify ambiguous or unrecognized commands"""
        print(f"Could not understand command: '{command_text}'. Please speak more clearly or use different words.")
        
        # In a real implementation, this would use text-to-speech to speak the message
        # self.tts_system.speak(f"Could not understand command: '{command_text}'. Please speak more clearly or use different words.")
        
        # Possible implementation of asking related questions:
        # "Did you mean to navigate, grasp, or perform another action?"
        pass

    def request_confirmation(self, command_text: str, parsed_command: ParsedCommand):
        """Request user confirmation for low-confidence commands"""
        print(f"Did you mean: '{parsed_command.action}' with parameters {parsed_command.parameters}? Please confirm.")
        
        # In a real implementation, would use TTS and wait for confirmation
        # self.tts_system.speak(f"Did you mean: '{parsed_command.action}'? Please confirm with yes or no.")
        pass

    def process_commands(self):
        """Process commands from the queue in a separate thread"""
        while True:
            try:
                # Get command from queue
                parsed_command = self.command_queue.get(timeout=1.0)
                
                # Execute the validated command
                success = self.execute_command(parsed_command)
                
                if success:
                    print(f"Command '{parsed_command.action}' executed successfully")
                    # Add to history
                    self.command_history.append(parsed_command)
                else:
                    print(f"Command '{parsed_command.action}' failed to execute")
                    # Could try alternative execution or ask for clarification
                    
                self.command_queue.task_done()
                
            except Exception as e:
                # Continue processing, just log the error
                print(f"Error processing command: {e}")
                time.sleep(0.1)

    def execute_command(self, command: ParsedCommand) -> bool:
        """Execute the parsed command on the robot"""
        # This would interface with the robot's action system
        # For now, we'll simulate execution
        
        print(f"Executing command: {command.action}")
        print(f"Parameters: {command.parameters}")
        print(f"Confidence: {command.confidence}")
        
        # Simulate execution time
        time.sleep(0.5)
        
        # Return success (in real implementation, this would be determined 
        # by whether the robot successfully completed the action)
        return True

    def get_robot_capabilities(self) -> Dict[str, Any]:
        """Get robot capabilities for validation"""
        return self.robot_capabilities

    def update_robot_capabilities(self, capabilities: Dict[str, Any]):
        """Update robot capabilities when they change"""
        self.robot_capabilities = capabilities

    def add_known_location(self, location: str):
        """Add a known location to robot capabilities"""
        known_locations = self.robot_capabilities.setdefault('known_locations', [])
        if location not in known_locations:
            known_locations.append(location)
```

## Multimodal Feedback and Interaction

Effective voice command systems provide multimodal feedback to confirm understanding and execution:

```python
from enum import Enum
import cv2

class FeedbackType(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    HAPTIC = "haptic"  # Not applicable for most humanoid robots but included for completeness
    LED = "led"

class VoiceCommandFeedbackSystem:
    def __init__(self):
        self.feedback_types = [FeedbackType.VISUAL, FeedbackType.AUDITORY, FeedbackType.LED]
        
        # Initialize feedback components
        self.speech_synthesizer = self.initialize_tts()
        self.led_controller = self.initialize_leds()
        self.display_system = self.initialize_display()
        
    def initialize_tts(self):
        """Initialize text-to-speech system"""
        # Would initialize a TTS engine like pyttsx3, gTTS, or similar
        pass
        
    def initialize_leds(self):
        """Initialize LED feedback system"""
        # Would interface with robot's LED system
        pass
        
    def initialize_display(self):
        """Initialize visual display system"""
        # Would interface with robot's screen or other visual output
        pass

    def provide_feedback(self, command: ParsedCommand, status: str = "received"):
        """Provide multimodal feedback for command"""
        # Visual feedback - change LED pattern or display status
        self.visual_feedback(command, status)
        
        # Auditory feedback - speak the understood command
        if status == "received":
            self.auditory_feedback(f"I heard {command.action}, with parameters {command.parameters}")
        elif status == "executing":
            self.auditory_feedback(f"Now executing {command.action}")
        elif status == "completed":
            self.auditory_feedback(f"Completed {command.action}")

    def visual_feedback(self, command: ParsedCommand, status: str):
        """Provide visual feedback through LEDs or display"""
        if status == "received":
            # Flash or change LED color to indicate command received
            self.led_controller.set_pattern("pulsing_blue")
        elif status == "validating":
            self.led_controller.set_pattern("flashing_yellow")  # Processing
        elif status == "confirmed":
            self.led_controller.set_pattern("solid_green")  # Ready to execute
        elif status == "executing":
            self.led_controller.set_pattern("moving_green")  # Execution in progress
        elif status == "completed":
            self.led_controller.set_pattern("success_sequence")  # Success animation
        elif status == "failed":
            self.led_controller.set_pattern("error_red")  # Error indication

    def auditory_feedback(self, message: str):
        """Provide auditory feedback through speech synthesis"""
        # In real implementation, would call TTS system
        print(f"Robot says: {message}")
        # self.speech_synthesizer.speak(message)

    def display_feedback(self, message: str, command: ParsedCommand = None):
        """Provide feedback on robot's display"""
        # Create a visual representation of the command
        if command:
            display_text = f"Command: {command.action}\nParams: {command.parameters}"
        else:
            display_text = message
            
        # In real implementation, would update robot's display
        # self.display_system.show_text(display_text)
        print(f"Displaying: {display_text}")


class IntegratedVoiceCommandSystem:
    def __init__(self):
        self.robot_capabilities = {
            'actions': ['navigate', 'grasp', 'bring', 'greet', 'follow', 'stop'],
            'known_locations': ['kitchen', 'living room', 'bedroom', 'office'],
            'graspable_objects': ['ball', 'cup', 'book', 'bottle', 'phone', 'keys'],
            'bringable_items': ['water', 'coffee', 'phone', 'book', 'keys']
        }
        
        self.command_handler = VoiceCommandHandler(self.robot_capabilities)
        self.feedback_system = VoiceCommandFeedbackSystem()
        
        self.current_command = None

    def start_system(self):
        """Start the complete voice command system"""
        print("Starting integrated voice command system")
        
        # Start listening loop in a separate thread
        listening_thread = threading.Thread(target=self.command_handler.start_listening_loop, daemon=True)
        listening_thread.start()
        
        print("Voice command system ready to receive commands")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down voice command system")
```

## Privacy and Security Considerations

Voice interfaces raise important privacy and security concerns that must be addressed in humanoid robot implementations:

### 1. Local Processing

To preserve privacy, sensitive audio processing should occur locally on the robot:

```python
class PrivacyPreservingProcessor:
    def __init__(self):
        # Ensure all processing occurs locally
        self.requires_cloud_processing = False
        
        # Use local-only models
        self.whisper_model = whisper.load_model("base.en", download_root="./models")
        
        # Secure audio buffer handling
        self.secure_buffer = True
        
    def process_audio_locally(self, audio_data):
        """Process audio without sending to cloud services"""
        # Perform all processing on-device
        result = self.whisper_model.transcribe(audio_data)
        return result

    def secure_audio_buffer(self):
        """Securely handle audio buffers"""
        # Implement secure buffer clearing
        # Ensure audio data doesn't persist beyond necessary processing
        pass
```

### 2. User Authentication

For sensitive commands, voice interfaces should include authentication:

```python
import hashlib
import hmac

class VoiceAuthenticator:
    def __init__(self):
        self.user_voices = {}  # Store voice profiles
        self.authenticated_users = set()
        
    def enroll_user_voice(self, user_id: str, audio_samples: list):
        """Enroll a user's voice for authentication"""
        # This is a simplified example - real implementation would use
        # speaker recognition technology
        voice_signature = self.create_voice_signature(audio_samples)
        self.user_voices[user_id] = voice_signature
        
    def create_voice_signature(self, audio_samples):
        """Create a signature from audio samples"""
        # In practice, would use speaker recognition algorithms
        # Simplified implementation for demonstration
        combined = b''.join(audio_samples)
        return hashlib.sha256(combined).hexdigest()
        
    def authenticate_user(self, audio_sample):
        """Authenticate user based on voice"""
        # Compare audio sample to stored signatures
        sample_signature = self.create_voice_signature([audio_sample])
        
        for user_id, stored_signature in self.user_voices.items():
            if stored_signature == sample_signature:
                return user_id
                
        return None
```

## Performance Optimization

For efficient operation on robotics hardware, performance optimization is crucial:

### 1. Model Quantization

```python
import torch

def quantize_model(model):
    """Apply quantization to reduce model size and improve inference speed"""
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model
```

### 2. Audio Processing Optimization

```python
import numpy as np
from scipy import signal

class OptimizedAudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.buffer_size = 4096  # Optimized buffer size
        
        # Pre-compute audio processing filters
        self.noise_reduction_filter = self.design_noise_filter()
        
    def design_noise_filter(self):
        """Design a noise reduction filter"""
        # Design a low-pass filter to remove high-frequency noise
        nyquist = self.sample_rate / 2
        cutoff = 7000 / nyquist  # Cutoff at 7kHz
        b, a = signal.butter(6, cutoff, btype='low', analog=False)
        return b, a

    def preprocess_audio(self, raw_audio):
        """Optimized audio preprocessing"""
        # Apply noise reduction filter
        filtered_audio = signal.filtfilt(*self.noise_reduction_filter, raw_audio)
        
        # Normalize audio
        max_val = np.max(np.abs(filtered_audio))
        if max_val > 0:
            normalized_audio = filtered_audio / max_val
        else:
            normalized_audio = filtered_audio
            
        return normalized_audio
```

## Testing and Evaluation

### 1. Performance Metrics

```python
class VoiceCommandEvaluator:
    def __init__(self):
        self.recognition_accuracy = 0
        self.response_time = 0
        self.command_success_rate = 0
        self.user_satisfaction = 0
        
    def evaluate_recognition_accuracy(self, test_commands, expected_results):
        """Evaluate voice recognition accuracy"""
        correct = 0
        total = len(test_commands)
        
        for i, command in enumerate(test_commands):
            # Simulate processing the command
            recognized = self.simulate_recognition(command)
            if recognized == expected_results[i]:
                correct += 1
                
        self.recognition_accuracy = correct / total if total > 0 else 0
        return self.recognition_accuracy
        
    def evaluate_response_time(self, commands):
        """Evaluate average response time"""
        times = []
        
        for command in commands:
            start_time = time.time()
            self.simulate_processing(command)
            end_time = time.time()
            times.append(end_time - start_time)
            
        self.response_time = sum(times) / len(times) if times else 0
        return self.response_time

    def simulate_recognition(self, audio_command):
        """Simulate recognition for testing purposes"""
        # This would normally interface with actual recognition system
        return "simulated_recognition_result"

    def simulate_processing(self, command):
        """Simulate processing for testing purposes"""
        time.sleep(0.1)  # Simulate processing delay
```

## Summary

This lesson covered the implementation of Whisper-based voice command systems for humanoid robots, including:

1. Setting up Whisper for robotics applications with proper audio preprocessing
2. Command parsing and validation to convert speech to executable actions
3. Error handling and disambiguation for robust operation
4. Multimodal feedback systems for effective human-robot interaction
5. Privacy and security considerations for voice interfaces
6. Performance optimization techniques for real-time operation

The integration of Whisper with robotics systems enables natural, intuitive interaction between humans and robots. By properly implementing these components, we can create voice interfaces that are accurate, responsive, and robust in real-world environments.

## Exercises

1. Implement a custom keyword spotting system to wake up the voice command system
2. Add speaker recognition to authenticate users before executing sensitive commands
3. Implement an alternative to Whisper using a different speech recognition engine
4. Create a multimodal interface that combines voice and gesture commands
5. Develop a privacy-preserving voice command system that operates entirely offline

## Further Reading

- OpenAI Whisper GitHub Repository and Documentation
- NVIDIA Riva Speech SDK for robotics applications
- Papers on voice interfaces for robotics applications
- Privacy-preserving speech recognition techniques