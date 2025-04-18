#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import threading
import requests
import queue
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Add the drone controller code path
sys.path.append('./drone_controller')

# Import our drone controller components
from drone_controller import DroneEKF, PointCloudProcessor, DroneController, DroneVisualization, DroneSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLM-Drone")

class LLMDroneInterface:
    """
    Interface between LLM and Drone Controller using models from Awesome-LLM-Robotics
    """
    
    def __init__(self, 
                 model_name: str = "gpt2",  # Default lightweight model for testing
                 use_api: bool = False,     # Whether to use local models or API
                 api_endpoint: str = None,  # API endpoint if using remote model
                 api_key: str = None,       # API key if needed
                 drone_system: Any = None,  # DroneSystem instance
                 memory_length: int = 10,   # Number of exchanges to remember
                 verbose: bool = True):     # Print detailed logs
        
        self.verbose = verbose
        self.memory_length = memory_length
        self.conversation_history = []
        self.use_api = use_api
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        
        # Connect to drone system if provided, otherwise will be set later
        self.drone_system = drone_system
        
        # Load the tokenizer and model from Hugging Face
        if not use_api:
            logger.info(f"Loading local model: {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Set up generation pipeline
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=256
                )
                logger.info("Local model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.warning("Switching to API mode")
                self.use_api = True
        
        # Prepare drone command templates
        self.command_templates = self._load_command_templates()
        
        # Set up message queue for async processing
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_messages)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("LLM Drone Interface initialized")
    
    def _load_command_templates(self) -> Dict:
        """Load command templates for structuring LLM outputs"""
        # These templates help the LLM generate consistent, parseable responses
        return {
            "move_to_position": {
                "pattern": "I'll move to position ({x}, {y}, {z})",
                "response": "Moving to coordinates ({x}, {y}, {z})"
            },
            "move_relative": {
                "pattern": "I'll move {direction} by {distance} meters",
                "response": "Moving {direction} by {distance} meters"
            },
            "hover": {
                "pattern": "I'll hover in place",
                "response": "Hovering at current position"
            },
            "land": {
                "pattern": "I'll land now",
                "response": "Initiating landing sequence"
            },
            "unknown": {
                "pattern": "I don't understand that command",
                "response": "Sorry, I couldn't understand that command"
            }
        }
    
    def set_drone_system(self, drone_system: Any) -> None:
        """Set the drone system after initialization"""
        self.drone_system = drone_system
        logger.info("Drone system connected to LLM interface")
    
    def _process_messages(self) -> None:
        """Background thread for processing messages"""
        while self.running:
            try:
                # Get the next message from the queue
                if not self.message_queue.empty():
                    user_input = self.message_queue.get()
                    
                    # Process the message
                    llm_response = self._generate_llm_response(user_input)
                    command_result = self._extract_and_execute_command(llm_response)
                    
                    # Package the full response
                    full_response = {
                        "llm_response": llm_response,
                        "command_result": command_result,
                        "drone_status": self._get_drone_status()
                    }
                    
                    # Add to response queue
                    self.response_queue.put(full_response)
                    self.message_queue.task_done()
                
                # Prevent tight loop
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                time.sleep(1)  # Pause on error
    
    def _get_drone_status(self) -> Dict:
        """Get the current status of the drone"""
        if self.drone_system and hasattr(self.drone_system, "controller"):
            # Get basic status from controller
            status = self.drone_system.controller.command_status.copy()
            
            # Add position information
            if hasattr(self.drone_system, "ekf"):
                position = self.drone_system.ekf.get_position()
                status["position"] = [float(p) for p in position]
            
            return status
        return {"status": "unknown", "message": "Drone system not connected"}
    
    def _generate_llm_response(self, user_input: str) -> str:
        """Generate response from LLM based on user input"""
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Trim history to memory length
        if len(self.conversation_history) > self.memory_length * 2:  # *2 because each exchange is 2 entries
            self.conversation_history = self.conversation_history[-self.memory_length * 2:]
        
        if self.use_api:
            return self._call_llm_api(user_input)
        else:
            return self._generate_local_response(user_input)
    
    def _generate_local_response(self, user_input: str) -> str:
        """Generate response using local model"""
        try:
            # Format the conversation history for the model
            formatted_history = self._format_conversation_history()
            prompt = f"{formatted_history}\nUser: {user_input}\nDrone Assistant:"
            
            # Generate response
            result = self.generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
            response_text = result[0]['generated_text'].split("Drone Assistant:")[-1].strip()
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            if self.verbose:
                logger.info(f"LLM Response: {response_text}")
            
            return response_text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble processing that request."
    
    def _call_llm_api(self, user_input: str) -> str:
        """Call external API to generate response"""
        try:
            # Format conversation for API
            messages = []
            for entry in self.conversation_history:
                messages.append({"role": entry["role"], "content": entry["content"]})
            
            # Prepare system message with context about drone control
            system_message = {
                "role": "system", 
                "content": (
                    "You are an AI assistant that controls a drone. "
                    "You can move the drone to specific coordinates, move it relative to its current position, "
                    "make it hover, or land. Respond conversationally but always include explicit control commands "
                    "when appropriate. For movement, use formats like 'I'll move to position (x, y, z)' or "
                    "'I'll move forward by 2 meters'. For hovering say 'I'll hover in place'. "
                    "For landing say 'I'll land now'."
                )
            }
            
            # Add system message at beginning
            messages.insert(0, system_message)
            
            # Call API
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key if provided
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": "gpt-3.5-turbo",  # Default model, replace with parameter if needed
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                
                # Add to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                if self.verbose:
                    logger.info(f"API Response: {response_text}")
                
                return response_text
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return "Sorry, I encountered an error while processing your request."
        
        except Exception as e:
            logger.error(f"Error calling API: {e}")
            return "I'm having trouble connecting to my language processing service."
    
    def _format_conversation_history(self) -> str:
        """Format the conversation history for local model prompt"""
        formatted = []
        for entry in self.conversation_history:
            role = "User" if entry["role"] == "user" else "Drone Assistant"
            formatted.append(f"{role}: {entry['content']}")
        return "\n".join(formatted)
    
    def _extract_and_execute_command(self, llm_response: str) -> Dict:
        """Extract and execute drone commands from LLM response"""
        import re
        
        # Initialize result
        result = {
            "command_type": "unknown",
            "parameters": {},
            "success": False,
            "message": "No command detected"
        }
        
        if not self.drone_system or not hasattr(self.drone_system, "controller"):
            result["message"] = "Drone system not connected"
            return result
        
        # Try to extract commands using regex patterns
        
        # Move to position pattern
        move_to_match = re.search(r"(?:move|go|fly) to position\s*\(?\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)?", llm_response, re.IGNORECASE)
        if move_to_match:
            x, y, z = map(float, move_to_match.groups())
            result["command_type"] = "move_to_position"
            result["parameters"] = {"position": [x, y, z]}
            
            # Execute command
            command = {
                "type": "move_to_position",
                "parameters": {"position": np.array([x, y, z])},
                "text": f"move to position ({x}, {y}, {z})"
            }
            status = self.drone_system.controller.process_command_text(command["text"])
            result["success"] = True
            result["message"] = status
            return result
        
        # Move relative patterns
        directions = {
            "forward": [1, 0, 0],
            "backward": [-1, 0, 0],
            "left": [0, 1, 0],
            "right": [0, -1, 0],
            "up": [0, 0, 1],
            "down": [0, 0, -1]
        }
        
        for direction, vector in directions.items():
            pattern = rf"(?:move|go|fly) {direction}(?: by)?\s+(-?\d+\.?\d*)\s*(?:m|meters|meter)??"
            rel_match = re.search(pattern, llm_response, re.IGNORECASE)
            if rel_match:
                distance = float(rel_match.group(1))
                movement = np.array(vector) * distance
                
                result["command_type"] = "move_relative"
                result["parameters"] = {
                    "direction": direction,
                    "distance": distance,
                    "movement": [float(m) for m in movement]
                }
                
                # Execute command
                command_text = f"move {direction} by {distance}"
                status = self.drone_system.controller.process_command_text(command_text)
                result["success"] = True
                result["message"] = status
                return result
        
        # Hover pattern
        if re.search(r"(?:hover|stay|hold position)", llm_response, re.IGNORECASE):
            result["command_type"] = "hover"
            
            # Execute command
            status = self.drone_system.controller.process_command_text("hover")
            result["success"] = True
            result["message"] = status
            return result
        
        # Land pattern
        if re.search(r"(?:land|touch down|come down)", llm_response, re.IGNORECASE):
            result["command_type"] = "land"
            
            # Execute command
            status = self.drone_system.controller.process_command_text("land")
            result["success"] = True
            result["message"] = status
            return result
        
        # No command detected
        return result
    
    def process_message(self, user_input: str) -> Dict:
        """
        Process a user message synchronously (for direct API use)
        
        Returns a dictionary with:
        - llm_response: The raw text response from the LLM
        - command_result: Details about the extracted command and execution result
        - drone_status: Current drone status information
        """
        llm_response = self._generate_llm_response(user_input)
        command_result = self._extract_and_execute_command(llm_response)
        
        return {
            "llm_response": llm_response,
            "command_result": command_result,
            "drone_status": self._get_drone_status()
        }
    
    def send_message(self, user_input: str) -> None:
        """
        Send a message for asynchronous processing
        The response will be available in the response_queue
        """
        self.message_queue.put(user_input)
    
    def get_response(self, timeout: float = 5.0) -> Optional[Dict]:
        """Get the next response from the queue"""
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def shutdown(self) -> None:
        """Shut down the interface"""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        logger.info("LLM Drone Interface shut down")


class EnhancedDroneSystem(DroneSystem):
    """Enhanced version of DroneSystem with LLM integration"""
    
    def __init__(self, args=None, llm_interface=None):
        # Initialize base drone system
        super().__init__(args)
        
        # Initialize or connect LLM interface
        if llm_interface is None:
            # Initialize with default local model (if available)
            self.llm_interface = LLMDroneInterface(drone_system=self)
        else:
            # Use provided interface and connect to this system
            self.llm_interface = llm_interface
            self.llm_interface.set_drone_system(self)
        
        # Add LLM integration to visualization
        self.setup_llm_visualization()
    
    def setup_llm_visualization(self):
        """Extend visualization with LLM interaction elements"""
        # This would be implemented by extending the visualization class
        # For simplicity, we're just adding a method to process LLM input
        self.visualization.process_llm_command = self.process_llm_command
    
    def process_llm_command(self, command_text):
        """Process a command through the LLM interface"""
        # Send command to LLM interface
        response = self.llm_interface.process_message(command_text)
        
        # Display the response in the visualization
        if hasattr(self.visualization, 'update_llm_response'):
            self.visualization.update_llm_response(response)
        
        return response


def setup_model_from_awesome_llm_robotics():
    """
    Set up a model from the Awesome-LLM-Robotics repository
    https://github.com/GT-RIPL/Awesome-LLM-Robotics
    """
    # Check if repository is cloned
    repo_path = "./Awesome-LLM-Robotics"
    if not os.path.exists(repo_path):
        logger.info("Cloning Awesome-LLM-Robotics repository...")
        os.system("git clone https://github.com/GT-RIPL/Awesome-LLM-Robotics.git")
    
    # Add repository to path
    sys.path.append(repo_path)
    
    # Look for specific implementations we can use
    # This is a placeholder - the actual implementation would depend on the repository structure
    logger.info("Checking for available models in Awesome-LLM-Robotics...")
    
    # For now, return a configuration for a default model
    # In a real implementation, this would select an appropriate model from the repository
    return {
        "model_name": "vicuna-7b-v1.5",  # Example model
        "use_api": False,
        "api_endpoint": None,
        "api_key": None
    }


def main():
    """Main function to run the LLM-enhanced drone system"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LLM-Enhanced Drone Control System")
    
    # Add standard drone system arguments
    parser.add_argument('--update-interval', type=int, default=30,
                       help='Visualization update interval in milliseconds')
    parser.add_argument('--process-noise', type=float, default=0.1,
                       help='EKF process noise (higher = faster response, less accuracy)')
    parser.add_argument('--measurement-noise', type=float, default=0.01,
                       help='EKF measurement noise (lower = trust sensors more)')
    parser.add_argument('--safety-threshold', type=float, default=0.5,
                       help='Minimum safe distance to obstacles in meters')
    parser.add_argument('--max-velocity', type=float, default=2.0,
                       help='Maximum drone velocity in m/s')
    
    # Add LLM-specific arguments
    parser.add_argument('--model', type=str, default=None,
                       help='Model name for LLM (or "api" to use API)')
    parser.add_argument('--api-endpoint', type=str, default=None,
                       help='API endpoint URL for LLM if using API mode')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key for LLM if using API mode')
    parser.add_argument('--awesome-llm', action='store_true',
                       help='Use models from Awesome-LLM-Robotics repository')
    
    args = parser.parse_args()
    
    try:
        # Configure LLM interface
        if args.awesome_llm:
            # Set up with models from the Awesome-LLM-Robotics repository
            llm_config = setup_model_from_awesome_llm_robotics()
            llm_interface = LLMDroneInterface(**llm_config)
        else:
            # Set up with specified or default model
            use_api = args.model == "api" or args.api_endpoint is not None
            model_name = None if use_api else (args.model or "gpt2")
            
            llm_interface = LLMDroneInterface(
                model_name=model_name,
                use_api=use_api,
                api_endpoint=args.api_endpoint,
                api_key=args.api_key
            )
        
        # Create enhanced drone system with LLM interface
        system = EnhancedDroneSystem(args=args, llm_interface=llm_interface)
        
        # Run the system
        logger.info("Starting LLM-Enhanced Drone Control System...")
        system.run()
        
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
