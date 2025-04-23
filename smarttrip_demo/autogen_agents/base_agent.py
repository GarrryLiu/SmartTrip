#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base AutoGen agent implementation for the SmartTrip Multi-Agent System.
Provides common functionality for message handling and agent coordination.
"""

import autogen
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI
from config.settings import API_KEYS, AUTOGEN_SETTINGS, AGENT_SETTINGS, SYSTEM_SETTINGS

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(SYSTEM_SETTINGS["LOG_LEVEL"])

# Add file handler if not already added
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('travel_mas.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class BaseAutoGenAgent:
    """
    Base class for AutoGen-based agents in the SmartTrip system.
    Provides common functionality and configuration management.
    """
    
    def __init__(
        self,
        agent_type: str,
        name: Optional[str] = None,
        system_message: Optional[str] = None,
        llm_config: Optional[Dict] = None,
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: int = 3
    ):
        """
        Initialize the base AutoGen agent.
        
        Args:
            agent_type: Type of agent (e.g., "TRAVEL_ASSISTANT", "BOOKING_ASSISTANT")
            name: Optional custom name for the agent
            system_message: Optional custom system message
            llm_config: Optional custom LLM configuration
            human_input_mode: Mode for handling human input
            max_consecutive_auto_reply: Maximum number of consecutive auto-replies
        """
        # Initialize collaboration-related attributes
        self.collaborators = {}  # Store references to other agents
        self.task_state = {}    # Store task state
        self.pending_requests = []  # Store pending requests from other agents
        self.agent_type = agent_type
        self.config = AUTOGEN_SETTINGS.get(agent_type, {})
        
        # Set up agent configuration
        self.name = name or self.config.get("name", f"agent_{agent_type.lower()}")
        self.system_message = system_message or self.config.get("system_message", "")
        self.llm_config = llm_config or self.config.get("llm_config", {})
        
        # Set up default LLM config if not provided
        if not self.llm_config:
            self.llm_config = {
                "config_list": [{
                    "model": "gpt-3.5-turbo-1106",
                    "api_key": API_KEYS["OPENAI_API_KEY"]
                }],
                "functions": [],
                "timeout": 60
            }
            
        # Initialize OpenAI client for direct LLM calls
        self.client = OpenAI(api_key=API_KEYS["OPENAI_API_KEY"])
        self.model = AGENT_SETTINGS["BOOKING"]["MODEL"]
        
        # Initialize the appropriate AutoGen agent type
        if agent_type == "USER_PROXY":
            self.agent = self._init_user_proxy(human_input_mode, max_consecutive_auto_reply)
        else:
            self.agent = self._init_assistant()
    
    def _init_assistant(self) -> autogen.AssistantAgent:
        """Initialize an AutoGen AssistantAgent."""
        
        # Create the assistant agent first
        agent = autogen.AssistantAgent(
            name=self.name,
            system_message=self.system_message,
            llm_config=self.llm_config,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            max_consecutive_auto_reply=3
        )
        
        # Define trigger function with proper signature
        def _trigger(messages: List[Dict], sender: Any, config: Dict) -> bool:
            """Custom trigger function that checks message validity."""
            if not messages:
                return False
            return True
        
        # Define reply function with proper signature
        def _reply_func(messages: List[Dict], sender: Any, config: Dict) -> Optional[str]:
            """Custom reply function that handles message processing and response generation."""
            if not messages:
                return None
                
            last_message = messages[-1]
            
            try:
                # Generate and format response
                response = self.generate_response(last_message)
                
                # Format the response
                if isinstance(response, str):
                    return response
                    
                # Handle dictionary responses
                if isinstance(response, dict):
                    if "error" in response:
                        return f"Error: {response['error']}"
                        
                    if "recommendations" in response:
                        recommendations = response["recommendations"]
                        if isinstance(recommendations, list):
                            return "\n".join([
                                f"Recommendation {i+1}:\n{json.dumps(rec, indent=2)}"
                                for i, rec in enumerate(recommendations)
                            ])
                        return json.dumps(recommendations, indent=2)
                        
                    return json.dumps(response, indent=2)
                    
                return str(response)
                
            except Exception as e:
                logger.error(f"{self.agent_type} - Error in reply function: {e}")
                return f"Error: {str(e)}"
        
        # Register reply function with proper trigger
        agent.register_reply(
            trigger=_trigger,  # Use the properly defined trigger function
            reply_func=_reply_func  # Use the properly defined reply function
        )
        
        return agent
    def _init_user_proxy(
        self,
        human_input_mode: str,
        max_consecutive_auto_reply: int
    ) -> autogen.UserProxyAgent:
        """
        Initialize an AutoGen UserProxyAgent.
        """
        return autogen.UserProxyAgent(
            name=self.name,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config=self.config.get("code_execution_config", {})
        )
    
    def initiate_chat(
        self,
        recipient: Union[autogen.AssistantAgent, autogen.UserProxyAgent],
        message: str,
        clear_history: bool = True
    ) -> None:
        """
        Initiate a chat with another agent.
        
        Args:
            recipient: The agent to chat with
            message: The initial message to send
            clear_history: Whether to clear chat history before starting
        """
        if clear_history:
            self.agent.reset()
            recipient.reset()
        
        self.agent.initiate_chat(
            recipient,
            message=message
        )
    
    def get_chat_history(self) -> List[Dict]:
        """
        Get the chat history for this agent.
        
        Returns:
            List of chat messages with their metadata
        """
        return self.agent.chat_messages
    
    def reset(self) -> None:
        """
        Reset the agent's chat history and state.
        """
        self.agent.reset()
    
    def check_message_relevance(self, message: Union[str, Dict]) -> Dict:
        """
        Use LLM to determine if the message is relevant to this agent.
        
        Args:
            message: The message to check, can be string or structured data
            
        Returns:
            Dict: Contains decision and reasoning
        """
        try:
            # Convert message to string
            if isinstance(message, dict):
                message_str = json.dumps(message)
            else:
                message_str = str(message)
                
            # Construct prompt
            prompt = f"""As a message relevance expert, analyze if this travel request requires the {self.agent_type}'s expertise.

Travel Request:
{message_str}

{self.agent_type} Core Responsibilities:
{self._get_agent_responsibility()}

Based on the request content and the agent's responsibilities, determine if this agent should process this request.

Key Analysis Points:
1. Does the request contain elements that match this agent's expertise?
2. Are there specific requirements that align with this agent's capabilities?
3. Would this agent's involvement improve the response quality?

Return a JSON object with:
{{
  "should_reply": true/false,
  "reason": "Detailed explanation of the decision"
}}
"""
            
            # Call LLM with debug logging
            logger.debug(f"{self.agent_type} checking message relevance...")
            
            response = self.safe_llm_call([
                {"role": "system", "content": "You are a specialized message relevance analyzer for a travel planning system."},
                {"role": "user", "content": prompt}
            ])
            
            logger.debug(f"{self.agent_type} LLM response: {response[:100]}...")
            
            # Parse response
            try:
                result = json.loads(response)
                should_reply = result.get("should_reply", False)
                reason = result.get("reason", "No reason provided")
            except json.JSONDecodeError:
                # Fall back to simple true/false check
                should_reply = response.strip().lower() == "true"
                logger.debug(f"{self.agent_type} - Simple relevance check: {should_reply}")
                reason = "Fallback to simple true/false check"
            
            logger.debug(f"{self.agent_type} relevance check:")
            logger.debug(f"  - Should reply: {should_reply}")
            logger.debug(f"  - Reason: {reason}")
            
            return {
                "should_reply": bool(should_reply),  # Ensure boolean type
                "reason": str(reason)  # Ensure string type
            }
            
        except Exception as e:
            logger.error(f"{self.agent_type} - Error in relevance check: {e}")
            # Fall back to keyword matching
            keyword_match = self._check_keywords(message_str)
            return {
                "should_reply": bool(keyword_match),  # Ensure boolean type
                "reason": f"Fallback to keyword matching due to error: {str(e)}"
            }
    
    def _get_agent_responsibility(self) -> str:
        """Return the responsibility description for this agent type"""
        responsibilities = {
            "FLIGHT_ASSISTANT": """
            - Handle flight search and booking requests
            - Analyze departure and destination information
            - Process flight date and time requirements
            - Handle flight budget constraints
            - Process airline preferences
            - Handle cabin class selections
            """,
            "HOTEL_ASSISTANT": """
            - Handle hotel search and booking requests
            - Analyze location and area preferences
            - Process check-in and check-out dates
            - Handle room type requirements
            - Process hotel budget constraints
            - Handle amenity and service preferences
            """,
            "ITINERARY_ASSISTANT": """
            - Handle itinerary planning requests
            - Analyze trip type (business/leisure/family)
            - Process activity and attraction recommendations
            - Arrange daily schedule timelines
            - Consider meal and rest times
            - Balance itinerary pace and intensity
            """
        }
        return responsibilities.get(self.agent_type, "")
    
    def _check_keywords(self, message_str: str) -> bool:
        """Use keyword matching as a fallback method"""
        keywords = []
        if hasattr(self, 'get_relevant_keywords'):
            keywords = self.get_relevant_keywords()
        return any(keyword in message_str.lower() for keyword in keywords)
    
    def register_collaborator(self, agent_type: str, agent: 'BaseAutoGenAgent') -> None:
        """
        Register a collaborating agent.
        
        Args:
            agent_type: Agent type identifier
            agent: Agent instance
        """
        self.collaborators[agent_type] = agent
        logger.debug(f"{self.agent_type} registered collaborator: {agent_type}")

    def send_request(self, to_agent: str, request_type: str, content: Dict) -> str:
        """
        Send request to another agent.
        
        Args:
            to_agent: Target agent type identifier
            request_type: Request type
            content: Request content
            
        Returns:
            str: Request response
        """
        if to_agent not in self.collaborators:
            return f"Error: {to_agent} not registered"
            
        request_id = f"{self.agent_type}_{to_agent}_{int(time.time())}"
        message = {
            "request_id": request_id,
            "from_agent": self.agent_type,
            "request_type": request_type,
            "content": content
        }
        
        logger.debug(f"{self.agent_type} sending request to {to_agent}: {request_type}")
        return self.collaborators[to_agent].handle_request(message)

    def handle_request(self, message: Dict) -> str:
        """
        Handle requests from other agents. Subclasses must implement this method.
        
        Args:
            message: Request message
            
        Returns:
            str: Response content
        """
        raise NotImplementedError("Agents must implement handle_request")

    def handle_message(self, message: Union[str, Dict]) -> Optional[Dict]:
        """
        Process incoming message and generate response.
        
        Args:
            message: Message to process
            
        Returns:
            Optional[Dict]: Response if message is relevant, None otherwise
        """
        # First check message relevance
        relevance = self.check_message_relevance(message)
        if not relevance["should_reply"]:
            return None
            
        try:
            # Analyze if other agents' assistance is needed
            analysis_result = self.analyze_request(message)
            
            # If other agents' assistance is needed
            if analysis_result.get("needed_agents"):
                for agent_type in analysis_result["needed_agents"]:
                    if agent_type in self.collaborators:
                        response = self.send_request(
                            agent_type,
                            "get_information",
                            {"request": message}
                        )
                        # Update message content
                        if isinstance(message, dict):
                            message.update(json.loads(response))
                        
            # Generate response
            response = self.generate_response(message)
            return {
                "agent_type": self.agent_type,
                "has_response": True,
                "response": response
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "agent_type": self.agent_type,
                "has_response": False,
                "error": str(e)
            }
    
    def safe_llm_call(self, messages: List[Dict]) -> str:
        """
        Safe LLM call wrapper that returns natural language response.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: Natural language response from LLM
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2
            )
            
            if not response or not hasattr(response, "choices") or not response.choices:
                return "Error: No response from LLM"
                
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            return f"Error: {str(e)}"
    
    def generate_reply(self, messages: List[Dict], sender: Any, config: Dict) -> Optional[str]:
        """
        Legacy method for compatibility. Use _init_assistant's _reply_func instead.
        
        Args:
            messages: Message history
            sender: Message sender
            config: Configuration information
        
        Returns:
            Optional[str]: Reply content, None if no reply needed
        """
        logger.debug(f"{self.agent_type} - generate_reply called directly")
        if not messages:
            return None
            
        try:
            # Get the latest message
            last_message = messages[-1]
            
            # Check message relevance
            decision = self.check_message_relevance(last_message)
            logger.debug(f"{self.agent_type} - Message relevance decision:")
            logger.debug(f"  - Should reply: {decision['should_reply']}")
            logger.debug(f"  - Reason: {decision['reason']}")
            
            if not decision["should_reply"]:
                logger.debug(f"{self.agent_type} - Message not relevant")
                return None
                
            # Generate and format response
            response = self.generate_response(last_message)
            if isinstance(response, str):
                return response
            return json.dumps(response, indent=2)
            
        except Exception as e:
            logger.error(f"{self.agent_type} - Error in generate_reply: {e}")
            return f"Error: {str(e)}"
    
    def generate_response(self, message: Union[str, Dict]) -> Dict:
        """
        Generate a response to a message.
        Should be implemented by each specific agent type.
        
        Args:
            message: The message to respond to
            
        Returns:
            Dict: The response data
        """
        raise NotImplementedError("Specific agents must implement generate_response")
    
    def analyze_request(self, message: Union[str, Dict]) -> Dict:
        """
        Analyze request to determine if other agents' assistance is needed.
        Subclasses should override this method.
        
        Args:
            message: Message to analyze
            
        Returns:
            Dict: Analysis result containing missing_info and needed_agents
        """
        return {
            "missing_info": [],
            "needed_agents": [],
            "can_process": True
        }

    def process_feedback(self, feedback: Union[str, Dict]) -> Dict:
        """
        Process user feedback and adjust response accordingly.
        
        Args:
            feedback: User feedback
            
        Returns:
            Dict: Updated response based on feedback
        """
        try:
            # Convert feedback to structured format
            if isinstance(feedback, str):
                feedback_data = {"text": feedback}
            else:
                feedback_data = feedback
                
            # Analyze if other agents' assistance is needed
            analysis_result = self.analyze_request(feedback_data)
            
            # If other agents' assistance is needed
            if analysis_result.get("needed_agents"):
                for agent_type in analysis_result["needed_agents"]:
                    if agent_type in self.collaborators:
                        response = self.send_request(
                            agent_type,
                            "process_feedback",
                            {"feedback": feedback_data}
                        )
                        feedback_data.update(json.loads(response))
            
            # Generate updated response
            updated_response = self.generate_response(feedback_data)
            
            return {
                "agent_type": self.agent_type,
                "has_response": True,
                "response": updated_response,
                "feedback_processed": True
            }
        except Exception as e:
            return {
                "agent_type": self.agent_type,
                "has_response": False,
                "error": f"Failed to process feedback: {str(e)}"
            }
