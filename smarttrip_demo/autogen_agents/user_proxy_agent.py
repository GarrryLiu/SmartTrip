"""
User Proxy Agent implementation using AutoGen.
This agent serves as the main interface between the user and other AutoGen agents.
Implements flexible GroupChat coordination for parallel agent processing.
"""

from typing import Dict, List, Optional, Any, Union
from .base_agent import BaseAutoGenAgent
import json
import time
import logging
import autogen
from openai import OpenAI
from config.settings import AUTOGEN_SETTINGS, AGENT_SETTINGS, API_KEYS, SYSTEM_SETTINGS

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(SYSTEM_SETTINGS["LOG_LEVEL"])

# Add file handler if not already added
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('travel_mas.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class UserProxyAgent(BaseAutoGenAgent):
    """
    User Proxy Agent that manages user interactions and coordinates with other AutoGen agents.
    Implements flexible message-based coordination through GroupChat.
    """ 
    
    def __init__(
        self,
        name: Optional[str] = None,
        system_message: Optional[str] = None,
        human_input_mode: str = "ALWAYS",
        max_consecutive_auto_reply: int = 1,
        agents: Optional[Dict[str, BaseAutoGenAgent]] = None
    ):
        """
        Initialize the User Proxy Agent with GroupChat capabilities.
        
        Args:
            name: Optional custom name for the agent
            system_message: Optional custom system message
            human_input_mode: Mode for handling human input
            max_consecutive_auto_reply: Maximum number of consecutive auto-replies
            agents: Dictionary of agent instances to coordinate with
        """
        super().__init__(
            agent_type="USER_PROXY",
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply
        )
        
        # Initialize OpenAI client for message parsing
        self.client = OpenAI(api_key=API_KEYS["OPENAI_API_KEY"])
        self.model = AUTOGEN_SETTINGS["FLIGHT_ASSISTANT"]["llm_config"]["config_list"][0]["model"]
        
        # Initialize GroupChat components
        self.agents = agents or {}
        self.conversation_history = []
        self.travel_data = {}
        self.previous_results = {}  # Store previous results
        self.termination_word = "done"  # Termination word to exit the system
        # Enhanced cache for responses and context
        self.cached_responses = {
            'flight_assistant': {
                'response': None,
                'original_request': None,
                'preferences': None
            },
            'hotel_assistant': {
                'response': None,
                'original_request': None,
                'preferences': None
            },
            'itinerary_assistant': {
                'response': None,
                'original_request': None,
                'preferences': None
            }
        }  # Cache for each assistant's context and response
        
        # Initialize GroupChat
        self.groupchat = None
        self.manager = None
        if agents:
            self._setup_groupchat()
    
    def _setup_groupchat(self) -> None:
        """Set up GroupChat with all available agents."""
        try:
            # Collect all agents
            all_agents = []
            logger.debug("Initializing agents for GroupChat...")
            for agent_name, agent in self.agents.items():
                logger.debug(f"Adding agent: {agent_name}")
                if hasattr(agent, 'agent'):
                    all_agents.append(agent.agent)
                else:
                    all_agents.append(agent)
            
            logger.debug(f"Total agents collected: {len(all_agents)}")
            
            # Initialize shared messages list
            shared_messages = []
            
            # Create GroupChat with shared messages
            self.groupchat = autogen.GroupChat(
                agents=all_agents,
                messages=shared_messages,
                max_round=12,
                speaker_selection_method="round_robin"  # Ensure each agent gets a turn
            )
            
            logger.debug("GroupChat initialized")
            
            # Create GroupChat manager with the same messages list
            self.manager = autogen.GroupChatManager(
                groupchat=self.groupchat,  # Use the same GroupChat instance
                llm_config={
                    "config_list": [{
                        "model": "gpt-4-1106-preview",
                        "api_key": API_KEYS["OPENAI_API_KEY"]
                    }],
                    "functions": [],
                    "timeout": 60
                }
            )
            
            logger.debug("GroupChat manager initialized")
            logger.debug(f"Available agents: {[agent.name for agent in all_agents]}")
        except Exception as e:
            logger.error(f"Error setting up GroupChat: {e}")
            raise
    def start_conversation(self, agents: Dict[str, BaseAutoGenAgent]) -> None:
        """
        Start an interactive conversation with the user.
        
        Args:
            agents: Dictionary of agent instances to coordinate with
        """
        # Initialize agents for coordination
        self.agents = agents
        self._setup_groupchat()
        
        if not self.groupchat or not self.manager:
            logger.error("GroupChat initialization failed.")
            return
            
        logger.info("Starting new conversation with user")
        print("\nWelcome to SmartTrip! I can help you plan your journey.")
        print("You can tell me about your travel plans in natural language or provide structured details.")
        print("\nFor example:")
        print("1. Natural: 'I want to fly from New York to London next month for a week-long vacation'")
        print("2. Structured: 'Flight: NYC to London; Hotel: Central London; Activities: Museums and Shopping'")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # Check termination word first
                if user_input.lower() == self.termination_word:
                    logger.info("User triggered termination word")
                    print("\nTerminating the entire agent system.")
                    import sys
                    sys.exit(0)
                elif user_input.lower() in ['exit', 'quit', 'bye']:
                    logger.info("User ended conversation")
                    print("\nThank you for using SmartTrip! Have a great journey!")
                    break
                
                if user_input.lower() == 'restart':
                    logger.info("User requested conversation restart")
                    self._handle_restart()
                    continue
                
                # Process user input
                logger.debug(f"Processing user input: {user_input}")
                self._process_input(user_input)
                
            except Exception as e:
                logger.error(f"Error processing user request: {e}")
                print("\nSorry, I encountered an error. Please try again or type 'restart' to start over.")
    
    def _process_input(self, user_input: str) -> None:
        """Process user input and coordinate with agents."""
        try:
            # Parse input into structured format
            parsed_input = self._parse_input(user_input)
            
            # Create message for agents
            message = {
                "role": "user",
                "content": f"""Process this travel request and provide recommendations:
                
Request Details:
{json.dumps(parsed_input, indent=2)}

Please work together to create a complete travel plan:
1. Flight Assistant: Search and recommend suitable flights
2. Hotel Assistant: Find and suggest accommodations
3. Itinerary Assistant: Create an activity plan

Coordinate your responses to create a cohesive travel experience."""
            }
            
            logger.debug("Broadcasting message to agents")
            logger.debug(f"Message content: {message}")
            
            # Validate message format
            logger.debug("Validating message format")
            validation_result = self._validate_message(message)
            logger.debug(f"Message validation result: {validation_result}")
            
            if not validation_result:
                raise ValueError("Invalid message format")
            
            # Broadcast message to all agents
            logger.debug("Adding message to GroupChat")
            self.groupchat.messages.append(message)
            
            logger.debug(f"Current agents in group chat: {[agent.name for agent in self.groupchat.agents]}")
            logger.debug(f"GroupChat messages before run: {len(self.groupchat.messages)}")
            logger.debug("First message content: {content}".format(
                content=self.groupchat.messages[0]["content"][:100] if self.groupchat.messages else "None"
            ))
            
            # Record agent decisions and run group chat
            logger.debug("Starting agent decision process")
            agent_decisions = {}
            
            # Check each agent's decision using original agent instances
            for agent_name, agent in self.agents.items():
                if agent_name != "user_proxy":
                    logger.debug(f"Checking {agent_name} decision...")
                    
                    # Check message relevance using original agent
                    decision = agent.check_message_relevance(message)
                    
                    # Ensure decision is in correct format
                    if not isinstance(decision, dict):
                        logger.debug(f"Converting legacy decision format from {agent_name}")
                        decision = {
                            "should_reply": bool(decision),
                            "reason": "Legacy boolean response"
                        }
                    
                    # Ensure required fields are present and correct types
                    should_reply = bool(decision.get("should_reply", False))
                    reason = str(decision.get("reason", "No reason provided"))
                    
                    logger.debug(f"{agent_name} decision:")
                    logger.debug(f"Should reply: {should_reply}")
                    logger.debug(f"Reason: {reason}")
                    
                    agent_decisions[agent_name] = {
                        "should_reply": should_reply,
                        "reason": reason,
                        "response": None
                    }
                    
                    if decision["should_reply"]:
                        logger.debug(f"{agent_name} decided to reply. Generating response...")
                        try:
                            # Generate response using the agent's generate_response method
                            response = agent.generate_response(message)
                            if response:
                                # Format the response
                                formatted_response = (
                                    response if isinstance(response, str)
                                    else json.dumps(response, indent=2)
                                )
                                
                                # Add response to group chat
                                self.groupchat.messages.append({
                                    "role": "assistant",
                                    "name": agent_name,
                                    "content": formatted_response
                                })
                                
                                logger.debug(f"Got response from {agent_name}")
                                agent_decisions[agent_name]["response"] = "Generated response successfully"
                            else:
                                logger.debug(f"{agent_name} generated empty response")
                                agent_decisions[agent_name]["response"] = "Generated empty response"
                                
                        except Exception as e:
                            logger.error(f"Error generating response from {agent_name}: {e}")
                            agent_decisions[agent_name]["response"] = f"Error generating response: {str(e)}"
            
            # Log decision summary
            logger.debug("Agent Decision Summary:")
            for agent_name, decision in agent_decisions.items():
                logger.debug(f"{agent_name}:")
                logger.debug(f"Should reply: {decision['should_reply']}")
                logger.debug(f"Reason: {decision['reason']}")
                logger.debug(f"Response status: {decision['response'] or 'No response needed'}")
            
            # Check final message count
            logger.debug(f"Final message count: {len(self.groupchat.messages)}")
            logger.debug("Last few messages:")
            for msg in self.groupchat.messages[-3:]:
                logger.debug(f"{msg.get('role')}: {msg.get('content')[:100]}...")
            
            # Process and display responses
            self._process_messages()
            
            # Ask for feedback with more flexible input
            print("\nHow does this look? Feel free to tell me if you'd like to adjust anything.")
            feedback = input("Your feedback (or press Enter if satisfied): ").strip()
            
            # If no feedback provided, assume user is satisfied
            if not feedback:
                logger.info("User satisfied with recommendations")
                print("\nGreat! Your travel plan is ready.")
                return
            
            # Check termination word before processing feedback
            if feedback.lower() == self.termination_word:
                logger.info("User triggered termination word")
                print("\nTerminating the entire agent system.")
                import sys
                sys.exit(0)
            
            # Process the feedback directly
            logger.debug(f"Processing user feedback: {feedback}")
            self._handle_feedback(feedback)
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            print("\nSorry, I encountered an error. Please try again with your travel details.")
    
    def _parse_input(self, text: str) -> Dict:
        """Parse user input into structured format using OpenAI."""
        try:
            prompt = f"""Parse this travel request and extract all relevant details.
            
            Input: {text}
            
            Return a JSON object with any of these fields that are mentioned:
            - flight: {{
                origin: departure city,
                destination: arrival city,
                dates: travel dates,
                budget: price limit,
                preferences: any specific requirements
            }}
            - hotel: {{
                location: area/district,
                dates: check-in/out dates,
                room_type: type of room,
                preferences: any specific requirements
            }}
            - activities: {{
                trip_type: type of trip (business/leisure/family),
                interests: list of interests/activities,
                pace: preferred pace,
                preferences: any specific requirements
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a travel request parser."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            parsed = json.loads(response.choices[0].message.content)
            
            # Store in travel data
            self.travel_data.update(parsed)
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing input: {e}")
            return {}
    
    def _display_responses(self) -> None:
        """Display agent responses in a structured format."""
        try:
            # Get the latest assistant messages
            assistant_messages = [
                msg for msg in self.groupchat.messages[-5:]  # Last 5 messages
                if msg.get("role") == "assistant"
            ]
            
            if not assistant_messages:
                # Only print this message if we're not in a feedback loop
                if not any(msg.get("content", {}).get("type") == "feedback" 
                         for msg in self.groupchat.messages[-5:]):
                    print("\nI'm working on your request...")
                return
                
            # Display each assistant's response
            for msg in assistant_messages:
                print(f"\nResponse from {msg.get('name', 'Assistant')}:")
                content = msg.get('content', '')
                
                # Try to parse as JSON first
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        if "error" in data:
                            print(f"Error: {data['error']}")
                        elif "recommendations" in data:
                            print("\nRecommendations:")
                            print(json.dumps(data["recommendations"], indent=2))
                            
                            # Save flight assistant results
                            if msg.get('name') == 'flight_assistant':
                                if 'raw_response' in data:
                                    self.previous_results = {
                                        'raw_response': data['raw_response'],
                                        'preferences': self.travel_data.get('flight', {}).get('preferences', {})
                                    }
                                    
                        elif "flight_recommendations" in data:
                            print("\nFlight Options:")
                            for i, flight in enumerate(data["flight_recommendations"], 1):
                                print(f"\nOption {i}:")
                                print(f"Airline: {flight.get('airline')}")
                                print(f"Flight: {flight.get('flight_number')}")
                                print(f"Departure: {flight.get('departure', {}).get('time')} from {flight.get('departure', {}).get('airport')}")
                                print(f"Price: ${flight.get('price', {}).get('amount')} {flight.get('price', {}).get('currency')}")
                        elif "hotel_recommendations" in data:
                            print("\nHotel Options:")
                            for i, hotel in enumerate(data["hotel_recommendations"], 1):
                                print(f"\nOption {i}:")
                                print(f"Name: {hotel.get('name')}")
                                print(f"Location: {hotel.get('address')}")
                                print(f"Price: ${hotel.get('price', {}).get('amount')} per night")
                                print(f"Rating: {hotel.get('rating', {}).get('score')}/5")
                        elif "itinerary" in data:
                            print("\nProposed Itinerary:")
                            for day in data["itinerary"]:
                                print(f"\nDay {day.get('day')} - {day.get('date')}:")
                                for activity in day.get('activities', []):
                                    print(f"{activity.get('time')}: {activity.get('activity')}")
                        else:
                            # Default to printing the formatted JSON
                            print(json.dumps(data, indent=2))
                except json.JSONDecodeError:
                    # If not JSON, print as plain text
                    print(content)
            
        except Exception as e:
            logger.error(f"Error displaying responses: {e}")
    
    def _handle_feedback(self, feedback_text: str) -> None:
        """
        Handle user feedback and request adjustments.
        
        Args:
            feedback_text: User's feedback text
        """
        try:
            logger.debug("Processing feedback...")
            
            # Check if feedback is hotel-related
            is_hotel_feedback = self._is_hotel_feedback(feedback_text)
            
            # Create feedback message with complete context
            feedback_message = {
                "role": "user",
                "content": {
                    "type": "feedback",
                    "feedback": feedback_text,
                    "original_request": self.travel_data,
                    "previous_results": self.previous_results,
                    "current_preferences": {
                        'flight': self.travel_data.get('flight', {}).get('preferences', {}),
                        'hotel': self.travel_data.get('hotel', {}).get('preferences', {}),
                        'activities': self.travel_data.get('activities', {}).get('preferences', {})
                    },
                    "feedback_history": self.groupchat.messages[-5:],  # Include recent context
                    "target_assistant": "hotel_assistant" if is_hotel_feedback else None
                }
            }
            
            # Store the current message count
            initial_message_count = len(self.groupchat.messages)
            
            # Process feedback
            self.groupchat.messages.append(feedback_message)
            
            # Run the manager with feedback and wait for responses
            logger.debug("Running manager with feedback...")
            print("Looking for better options...")
            self.manager.run(messages=[feedback_message])
            
            # Wait for new responses (maximum 10 attempts)
            max_attempts = 10
            attempt = 0
            has_new_responses = False
            
            while attempt < max_attempts:
                # Get new messages since feedback
                new_messages = self.groupchat.messages[initial_message_count:]
                new_assistant_messages = [
                    msg for msg in new_messages
                    if msg.get("role") == "assistant"
                ]
                
                if new_assistant_messages:
                    # Process and display responses, using cache for non-hotel assistants
                    self._display_specific_responses(new_assistant_messages, use_cache=is_hotel_feedback)
                    has_new_responses = True
                    break
                    
                attempt += 1
                time.sleep(1)  # Wait 1 second before checking again
            
            if not has_new_responses:
                logger.debug("No new responses received")
                print("\nI couldn't find new options. Could you provide more specific feedback?")
                # If no new responses but using cache, display cached responses
                if is_hotel_feedback:
                    logger.debug("Showing cached responses")
                    print("\nHere are the previous recommendations:")
                    cached_messages = [
                        msg['response'] for name, msg in self.cached_responses.items()
                        if msg['response'] is not None and name != 'hotel_assistant'
                    ]
                    if cached_messages:
                        self._display_specific_responses(cached_messages)
                return
            
            # Ask for additional feedback
            print("\nHow do these new options look?")
            additional_feedback = input("Your feedback (or press Enter if satisfied): ").strip()
            if additional_feedback.lower() == self.termination_word:
                logger.info("User triggered termination word")
                print("\nTerminating the entire agent system.")
                import sys
                sys.exit(0)
            elif additional_feedback:
                self._handle_feedback(additional_feedback)
            else:
                print("\nGreat! Your updated travel plan is ready.")
            
        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
    
    def _validate_message(self, message: Dict) -> bool:
        """
        Validate message format.
        
        Args:
            message: Message to validate
            
        Returns:
            bool: True if message format is valid
        """
        try:
            # Check required fields
            if not isinstance(message, dict):
                logger.debug("Message is not a dictionary")
                return False
                
            if 'role' not in message or 'content' not in message:
                logger.debug("Message missing required fields")
                return False
                
            if not isinstance(message['content'], (str, dict)):  # Allow content to be string or dict
                logger.debug("Message content is not a string or dict")
                return False
                
            if message['role'] not in ['user', 'assistant', 'system']:
                logger.debug("Invalid message role")
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Message validation error: {e}")
            return False
    
    def _process_messages(self) -> None:
        """Process messages after GroupChat run."""
        try:
            logger.debug("Processing messages...")
            
            # Get the latest messages
            latest_messages = self.groupchat.messages[-5:]  # Last 5 messages
            
            # Filter out assistant messages
            assistant_messages = [
                msg for msg in latest_messages 
                if msg.get("role") == "assistant"
            ]
            
            logger.debug(f"GroupChat messages: {len(self.groupchat.messages)}")
            logger.debug(f"Latest messages: {len(latest_messages)}")
            logger.debug(f"Assistant messages: {len(assistant_messages)}")
            
            if not assistant_messages:
                logger.debug("Waiting for assistant responses...")
                print("\nI'm gathering information...")
                return
                
            logger.debug(f"Found {len(assistant_messages)} assistant messages")
            logger.debug("Assistant responses:")
            for msg in assistant_messages:
                logger.debug(f"{msg.get('name', 'Unknown')}: {msg.get('content')[:100]}...")
                
            # Process and display responses
            self._display_responses()
            
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
    
    def _is_hotel_feedback(self, feedback_text: str) -> bool:
        """
        Check if the feedback is related to hotel.
        
        Args:
            feedback_text: The feedback text to analyze
            
        Returns:
            bool: True if feedback is hotel-related
        """
        hotel_keywords = [
            'hotel', 'room', 'accommodation', 'stay', 'lodge', 'resort',
            'suite', 'booking', 'check-in', 'checkout', 'location'
        ]
        feedback_lower = feedback_text.lower()
        return any(keyword in feedback_lower for keyword in hotel_keywords)

    def _cache_response(self, message: Dict) -> None:
        """
        Cache an assistant's response and context.
        
        Args:
            message: The response message to cache
        """
        assistant_name = message.get('name', '').lower()
        if assistant_name in self.cached_responses:
            # Parse content if it's a string
            content = message.get('content', '')
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {"text": content}
            
            # Update cache with response and context
            self.cached_responses[assistant_name].update({
                'response': message,
                'original_request': self.travel_data.copy(),
                'preferences': {
                    'flight': self.travel_data.get('flight', {}).get('preferences', {}),
                    'hotel': self.travel_data.get('hotel', {}).get('preferences', {}),
                    'activities': self.travel_data.get('activities', {}).get('preferences', {})
                }
            })
            logger.debug(f"Updated cache for {assistant_name}")

    def _display_specific_responses(self, messages: List[Dict], use_cache: bool = False) -> None:
        """
        Display specific agent responses.
        
        Args:
            messages: List of messages to display
        """
        try:
            displayed_assistants = set()
            
            # Display new messages first
            for msg in messages:
                assistant_name = msg.get('name', '').lower()
                displayed_assistants.add(assistant_name)
                
                print(f"\nResponse from {msg.get('name', 'Assistant')}:")
                content = msg.get('content', '')
                
                # Cache the response
                self._cache_response(msg)
                
                # Try to parse as JSON first
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        if "error" in data:
                            print(f"Error: {data['error']}")
                        elif "recommendations" in data:
                            print("\nRecommendations:")
                            print(json.dumps(data["recommendations"], indent=2))
                            
                            # Save flight assistant results
                            if msg.get('name') == 'flight_assistant':
                                if 'raw_response' in data:
                                    self.previous_results = {
                                        'raw_response': data['raw_response'],
                                        'preferences': self.travel_data.get('flight', {}).get('preferences', {})
                                    }
                        else:
                            # Default to printing the formatted JSON
                            print(json.dumps(data, indent=2))
                except json.JSONDecodeError:
                    # If not JSON, print as plain text
                    print(content)
                    
            # If using cache, display cached responses for assistants that didn't provide new messages
            if use_cache:
                for assistant_name, cached_msg in self.cached_responses.items():
                    if assistant_name not in displayed_assistants and cached_msg is not None:
                        print(f"\nPrevious response from {cached_msg.get('name', 'Assistant')}:")
                        content = cached_msg.get('content', '')
                        try:
                            data = json.loads(content)
                            print(json.dumps(data, indent=2))
                        except json.JSONDecodeError:
                            print(content)
                    
        except Exception as e:
            logger.error(f"Error displaying specific responses: {e}")
    
    def _handle_restart(self) -> None:
        """Reset the conversation state."""
        try:
            # Clear conversation history
            if self.groupchat:
                self.groupchat.messages = []
            
            # Reset travel data and previous results
            self.travel_data = {}
            self.previous_results = {}
            
            print("\nStarting over with a new travel plan.")
            print("Please tell me about your travel plans.")
            
        except Exception as e:
            logger.error(f"Error restarting conversation: {e}")
