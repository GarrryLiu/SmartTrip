#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SmartTrip - Travel Multi-Agent System
Main entry point for the system that coordinates AutoGen-based agents.
"""

import os
import sys
import logging
from config.settings import (
    API_KEYS,
    SYSTEM_SETTINGS,
    AGENT_SETTINGS,
    AUTOGEN_SETTINGS
)
from autogen_agents.user_proxy_agent import UserProxyAgent
from autogen_agents.flight_booking_assistant import FlightBookingAssistant
from autogen_agents.hotel_booking_assistant import HotelBookingAssistant
from autogen_agents.itinerary_assistant import ItineraryAssistant

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=SYSTEM_SETTINGS["LOG_LEVEL"],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_api_keys():
    """Validate that all required API keys are present."""
    required_keys = [
        "OPENAI_API_KEY",
        "GOOGLE_MAPS_API_KEY",
        "SKYSCANNER_API_KEY",
        "BOOKING_COM_API_KEY"
    ]
    missing_keys = [key for key in required_keys if not API_KEYS.get(key)]
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

def initialize_agents():
    """Initialize all agent instances and set up the collaboration network."""
    # Initialize specialized agents
    flight_assistant = FlightBookingAssistant(
        name=AUTOGEN_SETTINGS["FLIGHT_ASSISTANT"]["name"],
        system_message="""You are a flight booking assistant in a collaborative travel planning system.
Your responsibilities:
1. Search and recommend flights based on user requirements
2. Coordinate with hotel assistant on dates and locations
3. Share flight details with itinerary assistant for planning
4. Handle user feedback and adjust recommendations
5. Proactively communicate with other agents when needed""",
        llm_config=AUTOGEN_SETTINGS["FLIGHT_ASSISTANT"]["llm_config"]
    )
    
    hotel_assistant = HotelBookingAssistant(
        name=AUTOGEN_SETTINGS["HOTEL_ASSISTANT"]["name"],
        system_message="""You are a hotel booking assistant in a collaborative travel planning system.
Your responsibilities:
1. Search and recommend hotels based on user requirements
2. Use flight information to align check-in/out dates
3. Share hotel location with itinerary assistant for planning
4. Handle user feedback and adjust recommendations
5. Consider proximity to planned activities""",
        llm_config=AUTOGEN_SETTINGS["HOTEL_ASSISTANT"]["llm_config"]
    )
    
    itinerary_assistant = ItineraryAssistant(
        name=AUTOGEN_SETTINGS["ITINERARY_ASSISTANT"]["name"],
        system_message="""You are an itinerary assistant in a collaborative travel planning system.
Your responsibilities:
1. Create personalized travel itineraries
2. Use flight times to plan arrival/departure day activities
3. Consider hotel location when planning daily activities
4. Optimize routes and timing for efficiency
5. Adjust plans based on user feedback and preferences""",
        llm_config=AUTOGEN_SETTINGS["ITINERARY_ASSISTANT"]["llm_config"]
    )
    
    # Set up collaboration relationships
    flight_assistant.register_collaborator("HOTEL_ASSISTANT", hotel_assistant)
    flight_assistant.register_collaborator("ITINERARY_ASSISTANT", itinerary_assistant)
    
    hotel_assistant.register_collaborator("FLIGHT_ASSISTANT", flight_assistant)
    hotel_assistant.register_collaborator("ITINERARY_ASSISTANT", itinerary_assistant)
    
    itinerary_assistant.register_collaborator("FLIGHT_ASSISTANT", flight_assistant)
    itinerary_assistant.register_collaborator("HOTEL_ASSISTANT", hotel_assistant)
    
    # Initialize user proxy agent
    user_proxy = UserProxyAgent(
        name=AUTOGEN_SETTINGS["USER_PROXY"]["name"],
        system_message="""You are a travel planning coordinator in a collaborative system.
Your responsibilities:
1. Understand user travel requirements
2. Coordinate between flight, hotel, and itinerary assistants
3. Ensure all components work together seamlessly
4. Handle user feedback and direct it to appropriate assistants
5. Present final travel plans in a clear format""",
        human_input_mode=AUTOGEN_SETTINGS["USER_PROXY"]["human_input_mode"],
        max_consecutive_auto_reply=AUTOGEN_SETTINGS["USER_PROXY"]["max_consecutive_auto_reply"]
    )
    
    # Return all agents
    return {
        'user_proxy': user_proxy,
        'flight_assistant': flight_assistant,
        'hotel_assistant': hotel_assistant,
        'itinerary_assistant': itinerary_assistant
    }

def main():
    """
    Initialize and run the AutoGen-based multi-agent system.
    """
    logger = setup_logging()
    logger.info("Starting SmartTrip Multi-Agent System")
    
    print("\nWelcome to SmartTrip! Let's plan your perfect journey.\n")
    print("I can help you plan your entire trip, including:")
    print("1. Finding the best flights")
    print("2. Recommending suitable hotels")
    print("3. Creating personalized daily itineraries")
    print("\nAll components will work together to create your perfect travel plan!")
    
    try:
        # Validate API keys
        validate_api_keys()
        logger.info("API keys validated successfully")
        
        # Initialize all agents
        agents = initialize_agents()
        logger.info("All agents initialized successfully")
        
        # Create agent dictionary for coordination
        agent_dict = {
            'flight_assistant': agents['flight_assistant'],
            'hotel_assistant': agents['hotel_assistant'],
            'itinerary_assistant': agents['itinerary_assistant']
        }
        
        # 启动对话，由user proxy agent协调
        agents['user_proxy'].start_conversation(agent_dict)
        
    except ValueError as ve:
        print(f"\nConfiguration Error: {ve}")
        logger.error(f"Configuration error: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError initializing the system: {e}")
        print("Please check the configuration and try again.")
        logger.error(f"System initialization error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
