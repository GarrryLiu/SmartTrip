"""
Configuration settings for the SmartTrip Multi-Agent System.
Contains API keys, system settings, and other configuration parameters.
"""

# Load environment variables
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# API Keys and configuration loaded from environment variables
API_KEYS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),  # OpenAI API key from environment
    "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY"),  # Google Maps API key
    "SKYSCANNER_API_KEY": os.getenv("SKYSCANNER_API_KEY"),  # Skyscanner API key
    "SKYSCANNER_API_HOST": os.getenv("SKYSCANNER_API_HOST", "skyscanner89.p.rapidapi.com"),
    "SKYSCANNER_API_BASE_URL": os.getenv("SKYSCANNER_API_BASE_URL", "https://skyscanner89.p.rapidapi.com"),
    "BOOKING_COM_API_KEY": os.getenv("BOOKING_COM_API_KEY"),  # Booking.com API key
    "BOOKING_COM_API_HOST": os.getenv("BOOKING_COM_API_HOST", "booking-com.p.rapidapi.com")
}

# Validate required API keys
required_keys = ["OPENAI_API_KEY", "GOOGLE_MAPS_API_KEY", "SKYSCANNER_API_KEY", "BOOKING_COM_API_KEY"]
missing_keys = [key for key in required_keys if not API_KEYS[key]]
if missing_keys:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")

# Google Maps API configuration
GOOGLE_MAPS_CONFIG = {
    "PLACES_API_BASE_URL": "https://maps.googleapis.com/maps/api/place",
    "DETAILS_FIELDS": "name,rating,formatted_address,geometry,photo,type,opening_hours",
    "PHOTO_MAX_WIDTH": 400
}

# System settings
SYSTEM_SETTINGS = {
    "DEFAULT_LANGUAGE": "en",  # Default language for the system
    "DEBUG_MODE": False,  # Enable/disable debug mode
    "LOG_LEVEL": "INFO",  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    "MAX_RETRIES": 3,  # Maximum number of retries for API calls
    "TIMEOUT": 30,  # Timeout for API calls in seconds
}

# Agent settings
AGENT_SETTINGS = {
    "USER_PROXY": {
        "SYSTEM_MESSAGE": """You are a step-by-step travel planning assistant.
Guide users through their travel planning process in three stages:
1. Transportation details (flights)
2. Accommodation preferences (hotels)
3. Trip purpose and interests

Provide clear feedback and options at each step, and allow users to refine their choices.""",
        "MODEL": "gpt-4-1106-preview",
        "TEMPERATURE": 0.7,
    },
    "BOOKING": {
        "SYSTEM_MESSAGE": """You are a specialized agent for finding and booking travel arrangements.
Handle transportation and accommodation requests separately.
Consider user feedback and preferences to refine search results.
Provide clear options with pricing, timing, and relevant details.
Support iterative refinement based on user feedback.""",
        "MODEL": "gpt-4-1106-preview",
        "TEMPERATURE": 0.2,
    },
    "ITINERARY": {
        "SYSTEM_MESSAGE": """You are a specialized agent for creating personalized travel itineraries.
        
IMPORTANT INSTRUCTIONS:
1. Handle messages in a specific format:
   - User messages will be provided as input
   - Chat history will be a list of message objects
   - Agent scratchpad will be a list for intermediate work

2. Create itineraries based on trip type:
   - Business: Focus on efficiency and professional activities
   - Leisure: Balance sightseeing and relaxation
   - Family: Include family-friendly activities and flexible timing

3. Consider user preferences:
   - Adjust pace based on trip type
   - Include relevant activities and dining options
   - Allow for schedule flexibility

4. Provide detailed information:
   - Specific timing for activities
   - Transportation between locations
   - Cost estimates where applicable
   - Local tips and recommendations""",
        "MODEL": "gpt-4-1106-preview",
        "TEMPERATURE": 0.7,
    },
    "REVIEW": {
        "SYSTEM_MESSAGE": "You are a specialized agent for analyzing and summarizing travel reviews.",
        "MODEL": "gpt-4-1106-preview",
        "TEMPERATURE": 0.3,
    },
}

# AutoGen settings
AUTOGEN_SETTINGS = {
    "FLIGHT_ASSISTANT": {
        "name": "flight_assistant",
        "llm_config": {
            "config_list": [{
                "model": "gpt-4-1106-preview",
                "api_key": API_KEYS["OPENAI_API_KEY"]
            }]
        },
        "system_message": """You are a specialized flight booking assistant.
Focus on finding and recommending the best flight options:
- Understand complex travel requirements
- Search and filter flight options
- Consider price, timing, and comfort
- Provide clear comparisons and recommendations
- Handle special requests and preferences"""
    },
    "HOTEL_ASSISTANT": {
        "name": "hotel_assistant",
        "llm_config": {
            "config_list": [{
                "model": "gpt-4-1106-preview",
                "api_key": API_KEYS["OPENAI_API_KEY"]
            }]
        },
        "system_message": """You are a specialized hotel booking assistant.
Focus on finding and recommending the best accommodation options:
- Understand accommodation preferences
- Search and filter hotel options
- Consider location, amenities, and price
- Evaluate reviews and ratings
- Handle special requirements and requests"""
    },
    "ITINERARY_ASSISTANT": {
        "name": "itinerary_assistant",
        "llm_config": {
            "config_list": [{
                "model": "gpt-4-1106-preview",
                "api_key": API_KEYS["OPENAI_API_KEY"]
            }]
        },
        "system_message": """You are a specialized travel itinerary planner.
Focus on creating personalized travel experiences:
- Plan daily activities based on interests
- Optimize timing and logistics
- Include local insights and tips
- Balance activities and rest
- Consider weather and seasonal factors
- Adapt to different travel styles"""
    },
    "USER_PROXY": {
        "name": "user_proxy",
        "human_input_mode": "NEVER",
        "max_consecutive_auto_reply": 3,
        "code_execution_config": {
            "work_dir": "workspace",
            "use_docker": False
        }
    }
}
