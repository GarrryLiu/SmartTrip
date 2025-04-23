#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flight Booking Assistant Agent implementation using AutoGen.
Handles flight search and booking using Skyscanner API.
Implements message-based interaction for the multi-agent system.
"""

from typing import Dict, List, Optional, Union, Any
from .base_agent import BaseAutoGenAgent
import requests
import urllib.parse
import time
import json
import http.client
import logging
from datetime import datetime
from openai import OpenAI
from config.settings import API_KEYS, AGENT_SETTINGS, SYSTEM_SETTINGS

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(SYSTEM_SETTINGS["LOG_LEVEL"])

# Add file handler if not already added
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('travel_mas.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Maximum number of retries for API calls
MAX_RETRIES = 3
# Delay between retries (in seconds)
RETRY_DELAY = 2

class FlightBookingAssistant(BaseAutoGenAgent):
    """
    Flight Booking Assistant that handles flight search and recommendations
    using both OpenAI for understanding and Skyscanner for actual flight data.
    Implements message-based interaction for the multi-agent system.
    """
    
    def check_message_relevance(self, message: Union[str, Dict]) -> bool:
        """
        Check if the message is relevant to flight booking using both LLM and keywords.
        
        Args:
            message: The message to check
            
        Returns:
            bool: True if the message is relevant
        """
        try:
            # First check if message only contains hotel data
            if isinstance(message, dict):
                content = message.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except:
                        pass
                
                if isinstance(content, dict):
                    request_details = content
                    # If message only contains hotel data, return False
                    if "hotel" in request_details and "flight" not in request_details:
                        logger.debug(f"{self.agent_type} - Message only contains hotel data")
                        return False
            
            # Check if message is feedback type
            if isinstance(message, dict):
                content = message.get("content", {})
                if isinstance(content, dict) and content.get("type") == "feedback":
                    # Analyze if feedback is flight-related
                    feedback_text = content.get("feedback", "").lower()
                    flight_keywords = ["flight", "plane", "cheaper", "expensive", "price", "time", "direct"]
                    is_relevant = any(keyword in feedback_text for keyword in flight_keywords)
                    logger.debug(f"{self.agent_type} - Feedback relevance: {is_relevant}")
                    return is_relevant
            
            # Use LLM for more accurate relevance check
            prompt = """
            Determine if this message is specifically about flight booking.
            Return false if the message is only about hotels or other travel aspects.
            
            Message: {message}
            
            Return JSON: {{"is_flight_related": boolean, "reason": string}}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a message relevance analyzer."},
                    {"role": "user", "content": prompt.format(message=str(message))}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            is_relevant = result.get("is_flight_related", False)
            
            if is_relevant:
                logger.debug(f"{self.agent_type} - LLM determined message is relevant")
                return True
            
            # If LLM returns False, use stricter keyword matching as fallback
            if isinstance(message, dict):
                message_str = json.dumps(message)
            else:
                message_str = str(message)
                
            keywords = self.get_relevant_keywords()
            keyword_match = any(kw in message_str.lower() for kw in keywords)
            logger.debug(f"{self.agent_type} - Keyword match result: {keyword_match}")
            return keyword_match
            
        except Exception as e:
            logger.error(f"{self.agent_type} - Error in check_message_relevance: {e}")
            return False
    
    def get_relevant_keywords(self) -> List[str]:
        """Return keywords that indicate message relevance for flight booking."""
        return [
            "flight",
            "fly",
            "departure",
            "airline",
            "airport",
            "plane"
        ]
    
    def analyze_request(self, message: Union[str, Dict]) -> Dict:
        """
        Analyze request to determine if other agents' assistance is needed.
        
        Args:
            message: The message to analyze
            
        Returns:
            Dict: Analysis result containing missing_info and needed_agents
        """
        try:
            # Parse message content
            if isinstance(message, dict):
                content = message.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except:
                        content = {"text": content}
            else:
                content = {"text": str(message)}
            
            # Analyze missing information
            missing_info = []
            needed_agents = []
            
            # Check date information
            if "return_date" not in content and "round_trip" in content.get("text", "").lower():
                missing_info.append("return_date")
                needed_agents.append("ITINERARY_ASSISTANT")
            
            # Check destination information
            if "destination" in content:
                # If destination exists, might need hotel information
                needed_agents.append("HOTEL_ASSISTANT")
            
            # Check itinerary-related information
            if any(keyword in str(content).lower() for keyword in ["schedule", "itinerary", "plan"]):
                needed_agents.append("ITINERARY_ASSISTANT")
            
            return {
                "missing_info": missing_info,
                "needed_agents": list(set(needed_agents)),  # Remove duplicates
                "can_process": len(missing_info) == 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            return {
                "missing_info": [],
                "needed_agents": [],
                "can_process": True
            }

    def handle_request(self, message: Dict) -> str:
        """
        Handle requests from other agents.
        
        Args:
            message: Request message
            
        Returns:
            str: Response content
        """
        try:
            request_type = message["request_type"]
            content = message["content"]
            
            if request_type == "get_flight_times":
                # Return flight time information
                flight_id = content.get("flight_id")
                if not flight_id and "selected" in self.task_state:
                    flight_info = self.task_state["selected"]
                    logger.debug(f"Selected flight info: {json.dumps(flight_info, indent=2)}")
                    
                    # Handle both round-trip and one-way formats
                    if isinstance(flight_info, dict):
                        if "outbound" in flight_info:
                            # Round-trip format
                            outbound = flight_info["outbound"]
                            logger.debug("Using round-trip outbound flight data")
                            return json.dumps({
                                "departure_date": outbound.get("schedule", {}).get("departure", "").split("T")[0],
                                "return_date": flight_info.get("return", {}).get("schedule", {}).get("departure", "").split("T")[0],
                                "departure_time": outbound.get("schedule", {}).get("departure"),
                                "arrival_time": outbound.get("schedule", {}).get("arrival"),
                                "is_direct": outbound.get("is_direct", False),
                                "destination": outbound.get("schedule", {}).get("destination")
                            })
                        else:
                            # One-way format
                            logger.debug("Using one-way flight data")
                            departure_time = flight_info.get("schedule", {}).get("departure", "")
                            return json.dumps({
                                "departure_date": departure_time.split("T")[0] if departure_time else "",
                                "departure_time": departure_time,
                                "arrival_time": flight_info.get("schedule", {}).get("arrival"),
                                "is_direct": flight_info.get("is_direct", False),
                                "destination": flight_info.get("schedule", {}).get("destination")
                            })
                    else:
                        logger.warning(f"Invalid flight info type: {type(flight_info)}")
                
                logger.warning("No flight information available")
                return json.dumps({"error": "No flight information available"})
                
            elif request_type == "check_availability":
                # Check flight availability for specific dates
                search_result = self._get_flight_options(content)
                return json.dumps({
                    "available": bool(search_result.get("raw_response", {}).get("outbound")),
                    "options_count": len(search_result.get("raw_response", {}).get("outbound", []))
                })
                
            elif request_type == "get_information":
                # Handle general information request
                if isinstance(content, dict) and "request" in content:
                    result = self.process_request(json.dumps(content["request"]))
                    return json.dumps(result)
                
            return json.dumps({"error": f"Unknown request type: {request_type}"})
            
        except Exception as e:
            return json.dumps({"error": f"Error handling request: {str(e)}"})

    def generate_response(self, message: Union[str, Dict]) -> Dict:
        """
        Generate response to flight-related messages.
        
        Args:
            message: Message to respond to
            
        Returns:
            Dict: Flight recommendations and details
        """
        try:
            logger.debug("Generating flight response...")
            
            # First analyze if other agents' assistance is needed
            analysis_result = self.analyze_request(message)
            
            # If other agents' assistance is needed
            if analysis_result.get("needed_agents"):
                for agent_type in analysis_result["needed_agents"]:
                    if agent_type in self.collaborators:
                        if agent_type == "ITINERARY_ASSISTANT" and "return_date" in analysis_result["missing_info"]:
                            # Get trip duration information
                            response = self.send_request(
                                agent_type,
                                "get_trip_duration",
                                {"request": message}
                            )
                            trip_info = json.loads(response)
                            if isinstance(message, dict):
                                message["return_date"] = trip_info.get("end_date")
            
            # Handle feedback type messages
            if isinstance(message, dict):
                content = message.get("content", {})
                if isinstance(content, dict) and content.get("type") == "feedback":
                    logger.debug("Processing feedback message...")
                    # Re-analyze and recommend from raw response
                    previous_results = content.get("previous_results", {})
                    raw_response = previous_results.get("raw_response", {})
                    
                    if not raw_response or not raw_response.get("outbound"):
                        return {"error": "No previous flight data available for feedback"}
                    
                    # Update preferences based on feedback
                    preferences = self._update_preferences_from_feedback(
                        content.get("feedback"),
                        previous_results.get("preferences", {})
                    )
                    
                    # Re-analyze data
                    recommendations = self._analyze_with_openai(
                        raw_response=raw_response,
                        preferences=preferences
                    )
                    
                    if not recommendations or not recommendations.get("recommendations"):
                        return {"error": "Could not generate recommendations from feedback"}
                    
                    # Save selected flight information
                    self.task_state["selected"] = recommendations["recommendations"][0]
                    
                    return {
                        "status": "success",
                        "recommendations": recommendations,
                        "raw_flights": raw_response,
                        "message_type": "feedback_response"
                    }
                
                # Handle regular feedback
                if "feedback" in message:
                    return self._handle_feedback(message)
                user_input = json.dumps(message)
            else:
                user_input = str(message)
            
            logger.debug("Processing flight request...")
            # Process request using existing logic
            result = self.process_request(user_input)
            
            if "error" in result:
                logger.error(f"Process request error: {result['error']}")
                return result
            
            # Validate flight data
            raw_flights = result.get("raw_flights", {})
            if not raw_flights.get("outbound"):
                logger.error("No outbound flight data found")
                return {
                    "error": "No flight options found",
                    "debug_info": {
                        "stage": "data_validation",
                        "raw_response": bool(raw_flights),
                        "has_outbound": bool(raw_flights.get("outbound")),
                        "has_return": bool(raw_flights.get("return"))
                    }
                }
            
            # Validate recommendations
            recommendations = result.get("recommendations")
            if not recommendations:
                logger.error("No recommendations object in result")
                return {
                    "error": "Could not generate flight recommendations",
                    "debug_info": {
                        "stage": "recommendation_validation",
                        "has_raw_data": bool(raw_flights.get("outbound")),
                        "has_recommendations": bool(recommendations),
                        "raw_flight_count": len(raw_flights.get("outbound", {}).get("quotes", []))
                    }
                }
            
            if not recommendations.get("recommendations"):
                logger.error("Empty recommendations list")
                return {
                    "error": "No suitable flights found",
                    "debug_info": recommendations.get("debug_info", {})
                }
            
            # Process successful recommendations
            if len(recommendations["recommendations"]) > 0:
                # Save selected flight information
                selected_flight = recommendations["recommendations"][0]
                self.task_state["selected"] = selected_flight
                
                # Add success metrics
                success_info = {
                    "found_options": len(recommendations["recommendations"]),
                    "best_price": selected_flight.get("price") if isinstance(selected_flight, dict) else 
                                selected_flight.get("outbound", {}).get("price"),
                    "is_direct": selected_flight.get("is_direct") if isinstance(selected_flight, dict) else 
                                selected_flight.get("outbound", {}).get("is_direct")
                }
                
                # Notify hotel assistant about selected flight
                if "HOTEL_ASSISTANT" in self.collaborators:
                    logger.debug("Notifying hotel assistant about selected flight...")
                    self.send_request(
                        "HOTEL_ASSISTANT",
                        "notify_flight_selected",
                        {"flight": self.task_state["selected"]}
                    )
                
                logger.debug(f"Successfully generated {len(recommendations['recommendations'])} recommendations")
                return {
                    "status": "success",
                    "recommendations": recommendations,
                    "raw_flights": result.get("raw_flights", {}),
                    "message_type": "flight_recommendations",
                    "success_info": success_info
                }
            else:
                logger.error("No valid recommendations generated")
                return {
                    "error": "No valid flight recommendations",
                    "debug_info": recommendations.get("debug_info", {})
                }
            
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return {"error": f"Failed to generate response: {str(e)}"}
    
    def _handle_feedback(self, feedback_data: Dict) -> Dict:
        """
        Handle user feedback and adjust flight recommendations.
        
        Args:
            feedback_data: Dictionary containing feedback and original request
            
        Returns:
            Dict: Updated flight recommendations
        """
        try:
            feedback_text = feedback_data.get("feedback", "")
            original_request = feedback_data.get("original_request", {})
            
            # Use OpenAI to understand feedback and modify request
            prompt = f"""
            Original flight request:
            {json.dumps(original_request, indent=2)}
            
            User feedback:
            {feedback_text}
            
            Modify the original request parameters based on this feedback.
            Return the modified parameters in the same format.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a flight request modifier."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json"}
            )
            
            modified_request = json.loads(response.choices[0].message.content)
            
            # Process the modified request
            result = self.process_request(json.dumps(modified_request))
            
            return {
                "status": "success",
                "recommendations": result.get("recommendations", []),
                "raw_flights": result.get("raw_flights", {}),
                "message_type": "feedback_response",
                "modified_request": modified_request,
                "is_round_trip": bool(result.get("raw_flights", {}).get("return"))
            }
            
        except Exception as e:
            return {"error": f"Failed to process feedback: {str(e)}"}
    
    def __init__(
        self,
        name: Optional[str] = None,
        system_message: Optional[str] = None,
        llm_config: Optional[Dict] = None
    ):
        """Initialize the Flight Booking Assistant."""
        super().__init__(
            agent_type="FLIGHT_ASSISTANT",
            name=name,
            system_message=system_message,
            llm_config=llm_config
        )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=API_KEYS["OPENAI_API_KEY"])
        self.model = AGENT_SETTINGS["BOOKING"]["MODEL"]
        self.temperature = AGENT_SETTINGS["BOOKING"]["TEMPERATURE"]
        
        # Initialize Skyscanner API headers
        self.skyscanner_headers = {
            'x-rapidapi-key': API_KEYS["SKYSCANNER_API_KEY"],
            'x-rapidapi-host': API_KEYS["SKYSCANNER_API_HOST"]
        }
    
    def _extract_flight_data(self, response_data: Dict) -> Dict:
        """
        Extract and return flight-specific data from API response.
        
        Args:
            response_data: Raw API response data
            
        Returns:
            Dict: Filtered data containing only flight information
        """
        try:
            itineraries = response_data.get("data", {}).get("itineraries", {})
            return {
                "status": response_data.get("status"),
                "data": {
                    "itineraries": {
                        "buckets": itineraries.get("buckets", []),
                        "context": itineraries.get("context", {}),
                        "filterStats": itineraries.get("filterStats", {})
                    }
                }
            }
        except Exception as e:
            logger.error(f"Failed to extract flight data: {e}")
            return response_data
    
    def _make_api_request(self, endpoint: str, query_params: Dict = None, error_msg: str = "") -> Optional[Dict]:
        """
        Make API request with retry mechanism using http.client.
        
        Args:
            endpoint: API endpoint path (e.g. "/flights/auto-complete")
            query_params: Optional query parameters
            error_msg: Error message prefix for logging
            
        Returns:
            Optional[Dict]: Response JSON if successful, None otherwise
        """
        for attempt in range(MAX_RETRIES):
            conn = None
            try:
                # Construct query string
                query_string = ""
                if query_params:
                    query_parts = []
                    for key, value in query_params.items():
                        if value is not None:
                            query_parts.append(f"{key}={urllib.parse.quote(str(value))}")
                    if query_parts:
                        query_string = "?" + "&".join(query_parts)
                
                # Create full path
                full_path = endpoint + query_string
                logger.debug(f"Request path: {full_path}")
                
                # Make request
                conn = http.client.HTTPSConnection(API_KEYS["SKYSCANNER_API_HOST"])
                conn.request("GET", full_path, headers=self.skyscanner_headers)
                
                response = conn.getresponse()
                logger.debug(f"Response status: {response.status}")
                
                if response.status == 200:
                    data = response.read()
                    response_data = json.loads(data.decode("utf-8"))
                    # Extract flight data
                    flight_data = self._extract_flight_data(response_data)
                    logger.debug("API Response Structure:")
                    logger.debug(f"Response type: {type(response_data)}")
                    logger.debug(f"Top level keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
                    if isinstance(response_data, dict):
                        for key, value in response_data.items():
                            logger.debug(f"{key}: {type(value)}")
                            if isinstance(value, (list, dict)):
                                logger.debug(f"Length/Keys: {len(value) if isinstance(value, list) else list(value.keys())}")
                    return response_data
                elif response.status == 429:  # Rate limit
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Rate limit hit, retrying in {RETRY_DELAY} seconds...")
                        time.sleep(RETRY_DELAY)
                        continue
                    return {"error": "API rate limit exceeded"}
                else:
                    logger.error(f"{error_msg}: {response.status} - {response.reason}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    return None
                    
            except Exception as e:
                logger.error(f"{error_msg}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None
            finally:
                if conn:
                    conn.close()
        
        return None
    
    def _get_location_metadata(self, city: str) -> Optional[Dict]:
        """
        Get location code and ID from Skyscanner auto-complete API.
        
        Args:
            city: City name to search for
            
        Returns:
            Dictionary containing code and id if found, None otherwise
        """
        try:
            logger.debug(f"Getting location metadata for: {city}")
            
            # Make API request
            conn = http.client.HTTPSConnection(API_KEYS["SKYSCANNER_API_HOST"])
            
            # Construct query string
            encoded_city = urllib.parse.quote(city)
            endpoint = f"/flights/auto-complete?query={encoded_city}"
            
            logger.debug(f"Request endpoint: {endpoint}")
            
            # Make request
            conn.request("GET", endpoint, headers=self.skyscanner_headers)
            response = conn.getresponse()
            logger.debug(f"Response status: {response.status}")
            
            if response.status != 200:
                logger.error(f"API request failed: {response.status}")
                return None
                
            # Parse response
            data = json.loads(response.read().decode("utf-8"))
            # print(f"[DEBUG] Raw response type: {type(data)}")
            # print(f"[DEBUG] Raw response: {json.dumps(data, indent=2)}")
            
            # Parse response data based on type
            if isinstance(data, dict) and "inputSuggest" in data:
                suggestions = data["inputSuggest"]
                if suggestions:
                    # First try to find a city match
                    for suggestion in suggestions:
                        if suggestion.get("type") == "CITY":
                            flight_params = suggestion.get("navigation", {}).get("relevantFlightParams", {})
                            result = {
                                "code": flight_params.get("skyId"),
                                "id": flight_params.get("entityId")
                            }
                            logger.debug(f"Found city: {result}")
                            return result
                    
                    # If no city found, use first suggestion
                    top_result = suggestions[0]
                    flight_params = top_result.get("navigation", {}).get("relevantFlightParams", {})
                    result = {
                        "code": flight_params.get("skyId"),
                        "id": flight_params.get("entityId")
                    }
                    logger.debug(f"Using first suggestion: {result}")
                    return result
            elif isinstance(data, list):
                # Handle list response format
                for item in data:
                    if item.get("type") == "CITY":
                        result = {
                            "code": item.get("skyId"),
                            "id": item.get("entityId")
                        }
                        logger.debug(f"Found city: {result}")
                        return result
                
                # If no city found but we have results, use first one
                if data:
                    first_item = data[0]
                    result = {
                        "code": first_item.get("skyId"),
                        "id": first_item.get("entityId")
                    }
                    logger.debug(f"Using first suggestion: {result}")
                    return result
                    
            logger.error(f"No valid location info found for: {city}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting location metadata: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def process_request(self, user_input: str) -> Dict:
        """
        Process a flight booking request from start to finish.
        
        Args:
            user_input: User's raw input text
            
        Returns:
            Dictionary containing flight recommendations
        """
        try:
            # 1. Parse user input
            logger.debug("Parsing user request...")
            parsed_data = self._parse_with_openai(user_input)
            if "error" in parsed_data:
                return parsed_data
                
            logger.debug(f"Parsed request: {json.dumps(parsed_data, indent=2)}")
            
            # 2. Get flight options
            logger.debug("Searching for flights...")
            search_result = self._get_flight_options(parsed_data)
            if "error" in search_result:
                return search_result
            
            # 3. Analyze using raw response
            logger.debug("Analyzing options...")
            recommendations = self._analyze_with_openai(
                raw_response=search_result["raw_response"],
                preferences=parsed_data
            )
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "raw_flights": search_result["raw_response"]
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": f"Failed to process request: {str(e)}"}
    
    def _parse_with_openai(self, text: str) -> Dict:
        """Use OpenAI to parse user input into structured data."""
        try:
            prompt = f"""Please analyze this travel request and extract the following information:
            - Departure city
            - Destination city
            - Departure date
            - Return date (if mentioned)
            - Budget limit
            - Other preferences (airline, cabin class, etc.)
            
            User input: {text}
            
            Please respond in natural language with the following format:
            Departure: XXX
            Destination: XXX
            Departure Date: XXX
            Return Date: XXX (if mentioned)
            Budget: XXX
            Preferences: XXX"""
            
            response = self.safe_llm_call([
                {"role": "system", "content": "You are a travel request analysis expert."},
                {"role": "user", "content": prompt}
            ])
            
            return self._parse_natural_response(response)
            
        except Exception as e:
            logger.error(f"Error parsing input: {e}")
            return {"error": "Failed to parse request"}
    
    def _parse_natural_response(self, response: str) -> Dict:
        """Convert natural language response to structured data."""
        result = {}
        
        try:
            # Split response into lines and parse each line
            for line in response.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'departure':
                        result['origin'] = value
                    elif key == 'destination':
                        result['destination'] = value
                    elif key == 'departure date':
                        # Try to extract and format date
                        import re
                        date_match = re.search(r'\d{4}-\d{2}-\d{2}', value)
                        if date_match:
                            result['date'] = date_match.group(0)
                        else:
                            result['date'] = value
                    elif key == 'return date':
                        if value.lower() not in ['none', 'not mentioned', 'n/a']:
                            # Try to extract and format date
                            date_match = re.search(r'\d{4}-\d{2}-\d{2}', value)
                            if date_match:
                                result['return_date'] = date_match.group(0)
                            else:
                                result['return_date'] = value
                    elif key == 'budget':
                        # Extract numeric value
                        numbers = re.findall(r'\d+', value)
                        if numbers:
                            result['budget'] = int(numbers[0])
                    elif key == 'preferences':
                        result['preferences'] = {
                            "time": "any",
                            "class": "economy",
                            "airline": None,
                            "other": value
                        }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing natural response: {e}")
            return {"error": "Failed to parse natural language response"}
    
    def _get_flight_options(self, details: Dict) -> Dict:
        """Get flight options based on parsed details."""
        try:
            logger.debug("Getting flight options...")
            logger.debug(f"Search details: {json.dumps(details, indent=2)}")
            
            # Get location metadata for outbound flight
            origin_data = self._get_location_metadata(details["origin"])
            if not origin_data:
                return {"error": f"Could not find location data for {details['origin']}"}
            
            dest_data = self._get_location_metadata(details["destination"])
            if not dest_data:
                return {"error": f"Could not find location data for {details['destination']}"}
            
            # Construct base search parameters
            base_params = {
                "adults": "1",
                "currency": "USD",
                "market": "en-US",
                "countryCode": "US"
            }
            
            # Get outbound flight
            outbound_params = {
                **base_params,
                "origin": origin_data["code"],
                "originId": origin_data["id"],
                "destination": dest_data["code"],
                "destinationId": dest_data["id"],
                "date": details["date"]
            }
            
            # Add cabin class if specified
            if "cabin_class" in details:
                outbound_params["cabinClass"] = details["cabin_class"]
            
            # Get outbound flight results
            outbound_endpoint = (
                f"/flights/one-way/list"
                f"?origin={origin_data['code']}"
                f"&originId={origin_data['id']}"
                f"&destination={dest_data['code']}"
                f"&destinationId={dest_data['id']}"
                f"&date={urllib.parse.quote(details['date'])}"  # Add date parameter
            )
            
            logger.debug(f"Outbound request endpoint: {outbound_endpoint}")
            outbound_results = self._make_api_request(outbound_endpoint)
            
            logger.debug("Outbound Results Details:")
            logger.debug(f"Has data: {bool(outbound_results)}")
            logger.debug(f"Is error: {'error' in outbound_results if isinstance(outbound_results, dict) else False}")
            if isinstance(outbound_results, dict):
                logger.debug(f"Available keys: {list(outbound_results.keys())}")
                if 'quotes' in outbound_results:
                    logger.debug(f"Quotes count: {len(outbound_results['quotes'])}")
                    if outbound_results['quotes']:
                        logger.debug(f"First quote keys: {list(outbound_results['quotes'][0].keys())}")
                elif 'results' in outbound_results:
                    logger.debug(f"Results count: {len(outbound_results['results'])}")
                    if outbound_results['results']:
                        logger.debug(f"First result keys: {list(outbound_results['results'][0].keys())}")
            
            if not outbound_results or "error" in outbound_results:
                error_msg = outbound_results.get("error", "Failed to get outbound flight options") if isinstance(outbound_results, dict) else "No data returned"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Initialize return flight variables
            return_results = None
            return_params = None
            
            # Get return flight if return date is provided
            if details.get("return_date"):
                return_params = {
                    **base_params,
                    "origin": dest_data["code"],
                    "originId": dest_data["id"],
                    "destination": origin_data["code"],
                    "destinationId": origin_data["id"],
                    "date": details["return_date"]
                }
                
                if "cabin_class" in details:
                    return_params["cabinClass"] = details["cabin_class"]
                
                return_endpoint = (
                    f"/flights/one-way/list"
                    f"?origin={dest_data['code']}"
                    f"&originId={dest_data['id']}"
                    f"&destination={origin_data['code']}"
                    f"&destinationId={origin_data['id']}"
                    f"&date={urllib.parse.quote(details['return_date'])}"  # Add return date parameter
                )
                
                logger.debug(f"Return request endpoint: {return_endpoint}")
                return_results = self._make_api_request(return_endpoint)
                
                logger.debug("Return Results Details:")
                logger.debug(f"Has data: {bool(return_results)}")
                logger.debug(f"Is error: {'error' in return_results if isinstance(return_results, dict) else False}")
                if isinstance(return_results, dict):
                    logger.debug(f"Available keys: {list(return_results.keys())}")
                    if 'quotes' in return_results:
                        logger.debug(f"Quotes count: {len(return_results['quotes'])}")
                        if return_results['quotes']:
                            logger.debug(f"First quote keys: {list(return_results['quotes'][0].keys())}")
                    elif 'results' in return_results:
                        logger.debug(f"Results count: {len(return_results['results'])}")
                        if return_results['results']:
                            logger.debug(f"First result keys: {list(return_results['results'][0].keys())}")
                
                if not return_results or "error" in return_results:
                    error_msg = return_results.get("error", "Failed to get return flight options") if isinstance(return_results, dict) else "No data returned"
                    logger.error(error_msg)
                    return {"error": error_msg}
            
            # Return combined results
            return {
                "raw_response": {
                    "outbound": outbound_results,
                    "return": return_results
                },
                "metadata": {
                    "outbound": {
                        "origin": origin_data,
                        "destination": dest_data,
                        "search_params": outbound_params
                    },
                    "return": {
                        "origin": dest_data,
                        "destination": origin_data,
                        "search_params": return_params
                    } if return_params else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting flight options: {e}")
            return {"error": f"Failed to get flight options: {str(e)}"}
    
    def _filter_flights(self, flights: List[Dict]) -> List[Dict]:
        """
        Filter and sort flights by price, returning only top 5 cheapest options.
        Handles various price formats and data structures.
        
        Args:
            flights: List of raw flight data
            
        Returns:
            List[Dict]: Top 5 cheapest flights
        """
        try:
            if not flights:
                logger.debug("No flights to filter")
                return []
                
            logger.debug(f"Filtering {len(flights)} flights")
            
            def get_price(flight: Dict) -> float:
                """Extract price from various possible structures."""
                try:
                    # Try different price paths
                    if 'price' in flight and isinstance(flight['price'], dict):
                        return float(flight['price'].get('raw', float('inf')))
                    elif 'content' in flight and isinstance(flight['content'], dict):
                        return float(flight['content'].get('rawPrice', float('inf')))
                    elif isinstance(flight.get('rawPrice'), (int, float)):
                        return float(flight['rawPrice'])
                    else:
                        logger.warning(f"Could not find price in flight data: {flight.keys()}")
                        return float('inf')
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error parsing flight price: {e}")
                    return float('inf')
            
            # Sort flights by price with detailed error handling
            valid_flights = []
            invalid_flights = []
            
            for flight in flights:
                if not isinstance(flight, dict):
                    logger.warning(f"Invalid flight data type: {type(flight)}")
                    invalid_flights.append(flight)
                    continue
                    
                price = get_price(flight)
                if price != float('inf'):
                    valid_flights.append(flight)
                else:
                    invalid_flights.append(flight)
            
            logger.debug(f"Found {len(valid_flights)} valid flights, {len(invalid_flights)} invalid flights")
            
            sorted_flights = sorted(valid_flights, key=get_price)
            
            # Take top 5 flights
            result = sorted_flights[:5]
            logger.debug(f"Returning top {len(result)} flights")
            
            return result
            
        except Exception as e:
            logger.error(f"Error filtering flights: {str(e)}")
            logger.debug("Attempting fallback sorting...")
            try:
                # Simple fallback sorting
                return sorted(
                    [f for f in flights if isinstance(f, dict)],
                    key=lambda x: float(x.get('price', {}).get('raw', float('inf')))
                )[:5]
            except Exception as fallback_error:
                logger.error(f"Fallback sorting failed: {str(fallback_error)}")
                return flights[:5] if flights else []

    def _preprocess_flight_data(self, raw_data: Dict) -> Dict:
        """
        Extract essential flight information from raw data.
        Handles the Skyscanner API response format with legs and segments.
        
        Args:
            raw_data: Raw flight data from API
            
        Returns:
            Dict: Processed flight data with essential information
        """
        try:
            # Get the first leg which contains the main flight information
            legs = raw_data.get('legs', [{}])[0]
            
            # Get carrier information from the first segment
            segments = legs.get('segments', [{}])
            carrier = segments[0].get('marketingCarrier', {}) if segments else {}
            
            return {
                'flight_id': raw_data.get('id'),
                'basic_info': {
                    'price': raw_data.get('price', {}).get('formatted'),
                    'raw_price': raw_data.get('price', {}).get('raw'),
                    'is_direct': legs.get('stopCount', 1) == 0,
                    'duration': legs.get('durationInMinutes')
                },
                'airline': {
                    'name': carrier.get('name'),
                    'code': carrier.get('alternateId')
                },
                'schedule': {
                    'departure': legs.get('departure'),
                    'arrival': legs.get('arrival'),
                    'origin': legs.get('origin', {}).get('name'),
                    'destination': legs.get('destination', {}).get('name')
                }
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing flight data: {e}")
            return {
                'flight_id': 'unknown',
                'basic_info': {
                    'price': 'N/A',
                    'raw_price': None,
                    'is_direct': False,
                    'duration': None
                },
                'airline': {
                    'name': None,
                    'code': None
                },
                'schedule': {
                    'departure': None,
                    'arrival': None,
                    'origin': None,
                    'destination': None
                }
            }

    def safe_get_itineraries(self, flight_data: Dict) -> List[Dict]:
        """
        Safely extract itineraries from flight data with error handling.
        Handles the nested bucket structure from Skyscanner API.
        
        Args:
            flight_data: Raw flight data from API response
            
        Returns:
            List[Dict]: List of itineraries, empty list if extraction fails
        """
        try:
            logger.debug("Extracting itineraries from flight data")
            itineraries = []
            buckets = flight_data.get("data", {}).get("itineraries", {}).get("buckets", [])
            
            logger.debug(f"Found {len(buckets)} buckets")
            for bucket in buckets:
                items = bucket.get("items", [])
                logger.debug(f"Found {len(items)} items in bucket {bucket.get('id', 'unknown')}")
                itineraries.extend(items)
            
            logger.debug(f"Total extracted itineraries: {len(itineraries)}")
            return itineraries
            
        except Exception as e:
            logger.error(f"Failed to extract itineraries: {e}")
            logger.debug(f"Flight data type: {type(flight_data)}")
            if isinstance(flight_data, dict):
                logger.debug(f"Available keys: {list(flight_data.keys())}")
            return []
    
    def _analyze_with_openai(self, raw_response: Dict, preferences: Dict) -> Dict:
        """
        Use OpenAI to analyze flight search results and make recommendations.
        Returns a dictionary containing recommendations and summary.
        """
        try:
            logger.debug("Starting flight analysis...")
            
            # Check if this is a round-trip flight
            is_round_trip = raw_response.get("return") is not None
            logger.debug(f"Flight type: {'round-trip' if is_round_trip else 'one-way'}")
            
            # Analyze raw response structure
            logger.debug("Raw Response Analysis:")
            logger.debug(f"Response type: {type(raw_response)}")
            if isinstance(raw_response, dict):
                logger.debug(f"Top level keys: {list(raw_response.keys())}")
                logger.debug(f"Outbound type: {type(raw_response.get('outbound'))}")
                if isinstance(raw_response.get('outbound'), dict):
                    logger.debug(f"Outbound keys: {list(raw_response['outbound'].keys())}")
                    logger.debug(f"Has quotes: {'quotes' in raw_response['outbound']}")
                    logger.debug(f"Has results: {'results' in raw_response['outbound']}")
            
            # Safely extract flight data
            outbound_flights = self.safe_get_itineraries(raw_response.get("outbound", {}))
            
            logger.debug("Extracting flight data:")
            logger.debug(f"Found outbound itineraries: {len(outbound_flights)}")
            if outbound_flights:
                logger.debug(f"Sample itinerary keys: {list(outbound_flights[0].keys()) if outbound_flights else []}")
            
            # Validate outbound flights
            if not outbound_flights:
                logger.error("No outbound flights found")
                return {
                    "recommendations": [],
                    "summary": "No outbound flights available",
                    "debug_info": {
                        "raw_response_keys": list(raw_response.keys()) if isinstance(raw_response, dict) else None,
                        "outbound_data": raw_response.get("outbound")
                    }
                }
                
            # Process outbound flights with detailed logging
            logger.debug("Processing outbound flights:")
            logger.debug(f"Raw flights count: {len(outbound_flights)}")
            
            try:
                filtered_outbound = self._filter_flights(outbound_flights)
                logger.debug(f"Filtered quotes count: {len(filtered_outbound)}")
                
                processed_outbound = []
                for flight in filtered_outbound:
                    try:
                        processed = self._preprocess_flight_data(flight)
                        if processed.get('flight_id') != 'unknown':
                            processed_outbound.append(processed)
                        else:
                            logger.debug(f"Skipping flight due to missing data: {flight.get('id', 'No ID')}")
                    except Exception as e:
                        logger.error(f"Error processing flight: {str(e)}")
                        continue
                
                logger.debug(f"Successfully processed {len(processed_outbound)} flights")
                if processed_outbound:
                    logger.debug("Sample processed flight:")
                    logger.debug(json.dumps(processed_outbound[0], indent=2))
                
            except Exception as e:
                logger.error(f"Failed to process outbound flights: {str(e)}")
                return {
                    "recommendations": [],
                    "summary": "Error processing flight data",
                    "debug_info": {
                        "error": str(e),
                        "stage": "outbound_processing",
                        "raw_flights": len(outbound_flights)
                    }
                }
            
            # Process return flights if round trip
            processed_return = None
            if is_round_trip:
                logger.debug("Processing return flights:")
                # Safely extract return flight data
                return_flights = self.safe_get_itineraries(raw_response.get("return", {}))
                logger.debug(f"Found return itineraries: {len(return_flights)}")
                if return_flights:
                    logger.debug(f"Sample return itinerary keys: {list(return_flights[0].keys()) if return_flights else []}")
                
                if not return_flights:
                    logger.error("No return flights found")
                    return {
                        "recommendations": [],
                        "summary": "No return flights available",
                        "debug_info": {
                            "raw_response_keys": list(raw_response.keys()) if isinstance(raw_response, dict) else None,
                            "return_data": raw_response.get("return")
                        }
                    }
                
                try:
                    filtered_return = self._filter_flights(return_flights)
                    logger.debug(f"Filtered quotes count: {len(filtered_return)}")
                    
                    processed_return = []
                    for flight in filtered_return:
                        try:
                            processed = self._preprocess_flight_data(flight)
                            if processed.get('flight_id') != 'unknown':
                                processed_return.append(processed)
                            else:
                                logger.debug(f"Skipping return flight due to missing data: {flight.get('id', 'No ID')}")
                        except Exception as e:
                            logger.error(f"Error processing return flight: {str(e)}")
                            continue
                    
                    logger.debug(f"Successfully processed {len(processed_return)} return flights")
                    if processed_return:
                        logger.debug("Sample processed return flight:")
                        logger.debug(json.dumps(processed_return[0], indent=2))
                    
                except Exception as e:
                    logger.error(f"Failed to process return flights: {str(e)}")
                    return {
                        "recommendations": [],
                        "summary": "Error processing return flight data",
                        "debug_info": {
                            "error": str(e),
                            "stage": "return_processing",
                            "raw_quotes": len(return_quotes)
                        }
                    }
            
            # Create optimized prompt with preprocessed data
            logger.debug("Creating analysis prompt...")
            prompt = f"""Analyze and recommend the best flight options based on the following criteria:

            Flight Type: {is_round_trip and "Round-trip" or "One-way"}
            Dates:
            - Outbound: {preferences.get('date')}
            {f"- Return: {preferences.get('return_date')}" if is_round_trip else ""}

            Available Options:
            1. Outbound Flights (Top {len(processed_outbound)}):
            {json.dumps(processed_outbound, indent=2)}

            {f"2. Return Flights (Top {len(processed_return)}):" if is_round_trip else ""}
            {json.dumps(processed_return, indent=2) if is_round_trip else ""}

            User Preferences:
            {json.dumps(preferences, indent=2)}

            Selection Criteria (in order of importance):
            1. Price optimization
            2. Direct flight availability
            3. Schedule convenience
            4. Airline reputation and service quality

            Required Response Format:
            {{
                "recommendations": [
                    {{
                        "flight_id": "unique identifier",
                        "airline": "carrier name",
                        "schedule": {{
                            "departure": "YYYY-MM-DDTHH:mm:ss",
                            "arrival": "YYYY-MM-DDTHH:mm:ss",
                            "origin": "origin city/airport",
                            "destination": "destination city/airport"
                        }},
                        "price": "formatted price string",
                        "is_direct": true/false,
                        "benefits": ["benefit 1", "benefit 2"],
                        "recommendation_reason": "detailed explanation"
                    }}
                ],
                "summary": "overall analysis and recommendation summary"
            }}

            Additional Requirements:
            - Provide exactly 3 recommendations
            - Each recommendation must include all specified fields
            - Focus on concrete benefits (e.g., "shortest duration", "best value")
            - Include specific reasons for each recommendation
            
            Return a JSON response with this structure:
            {{
              "recommendations": [
                {{"outbound": {{
                    "flight_id": string,
                    "airline": string,
                        "schedule": {{
                        "departure": string,
                        "arrival": string,
                        "origin": string,
                        "destination": string
                    }},
                    "price": string,
                    "is_direct": boolean
                }},
                "return": {{
                    // Same structure as outbound
                }},
                "total_cost": string,
                "benefits": [string],
                "recommendation_reason": string
                }} if is_round_trip else {{
                    "flight_id": string,
                    "airline": string,
                    "schedule": {{
                        "departure": string,
                        "arrival": string
                    }},
                    "price": string,
                    "is_direct": boolean,
                    "benefits": [string],
                    "recommendation_reason": string
                }}
              ],
              "summary": string
            }}
            """
            
            logger.debug("Sending analysis request to OpenAI...")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "You are a flight recommendation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                # Parse OpenAI response
                recommendations = json.loads(response.choices[0].message.content)
                logger.debug(f"Generated {len(recommendations.get('recommendations', []))} recommendations")
                
                # Validate response structure
                if not isinstance(recommendations, dict):
                    raise ValueError("OpenAI response is not a dictionary")
                
                if "recommendations" not in recommendations:
                    raise ValueError("OpenAI response missing 'recommendations' field")
                
                if not isinstance(recommendations["recommendations"], list):
                    raise ValueError("'recommendations' field is not a list")
                
                # Validate each recommendation
                for i, rec in enumerate(recommendations["recommendations"]):
                    if is_round_trip:
                        required_fields = ["outbound", "return", "total_cost", "benefits", "recommendation_reason"]
                        if not all(field in rec for field in required_fields):
                            raise ValueError(f"Recommendation {i} missing required fields for round-trip")
                        
                        flight_fields = ["flight_id", "airline", "schedule", "price", "is_direct"]
                        schedule_fields = ["departure", "arrival", "origin", "destination"]
                        for flight_type in ["outbound", "return"]:
                            if not all(field in rec[flight_type] for field in flight_fields):
                                raise ValueError(f"Recommendation {i} {flight_type} missing required flight fields")
                            if not all(field in rec[flight_type]["schedule"] for field in schedule_fields):
                                logger.debug(f"Copying schedule fields from processed data for {flight_type}")
                                # Find matching flight from processed data
                                flight_id = rec[flight_type]["flight_id"]
                                flights = processed_outbound if flight_type == "outbound" else processed_return
                                for flight in flights:
                                    if flight["flight_id"] == flight_id:
                                        rec[flight_type]["schedule"] = flight["schedule"]
                                        break
                    else:
                        required_fields = ["flight_id", "airline", "schedule", "price", "is_direct", "benefits", "recommendation_reason"]
                        schedule_fields = ["departure", "arrival", "origin", "destination"]
                        if not all(field in rec for field in required_fields):
                            raise ValueError(f"Recommendation {i} missing required fields for one-way")
                        if not all(field in rec["schedule"] for field in schedule_fields):
                            logger.debug(f"Copying schedule fields from processed data for one-way flight")
                            # Find matching flight from processed data
                            flight_id = rec["flight_id"]
                            for flight in processed_outbound:
                                if flight["flight_id"] == flight_id:
                                    rec["schedule"] = flight["schedule"]
                                    break
                
                logger.debug("OpenAI response validation successful")
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to parse OpenAI response as JSON: {json_error}")
                return {"recommendations": [], "summary": "Error: Invalid JSON response from OpenAI"}
            except ValueError as val_error:
                logger.error(f"OpenAI response validation failed: {val_error}")
                return {"recommendations": [], "summary": f"Error: Invalid response format - {str(val_error)}"}
            except Exception as api_error:
                logger.error(f"OpenAI API error: {api_error}")
                return {"recommendations": [], "summary": f"Error generating recommendations: {str(api_error)}"}
            
            # Initialize recommendations list if not present
            if "recommendations" not in recommendations:
                recommendations["recommendations"] = []
            
            # Add debug information
            recommendations["debug_info"] = {
                "outbound_flights": len(processed_outbound),
                "return_flights": len(processed_return) if processed_return else 0,
                "processed_count": len(recommendations["recommendations"])
            }
            
            # Update summary if empty
            if not recommendations.get("summary"):
                if len(recommendations["recommendations"]) > 0:
                    recommendations["summary"] = f"Found {len(recommendations['recommendations'])} flight options"
                else:
                    recommendations["summary"] = "No suitable flights found matching criteria"
            
            logger.debug(f"Generated {len(recommendations['recommendations'])} recommendations")
            return recommendations
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in flight analysis: {error_msg}")
            return {
                "recommendations": [],
                "summary": f"Error analyzing flights: {error_msg}",
                "debug_info": {
                    "error": error_msg,
                    "outbound_flights": len(processed_outbound) if 'processed_outbound' in locals() else 0,
                    "return_flights": len(processed_return) if 'processed_return' in locals() else 0,
                    "stage": "flight_analysis"
                }
            }
    
    def _update_preferences_from_feedback(self, feedback: str, original_preferences: Dict) -> Dict:
        """
        Update preferences based on user feedback.
        Optimized to focus on key preference aspects.
        
        Args:
            feedback: User feedback text
            original_preferences: Original preference settings
            
        Returns:
            Dict: Updated preferences
        """
        prompt = """
        Analyze this feedback and update flight preferences.
        Focus on concrete changes in:
        1. Price sensitivity (e.g., 'too expensive', 'looking for cheaper')
        2. Schedule preferences (e.g., 'prefer morning', 'too late')
        3. Airline choice (e.g., 'prefer United', 'avoid American')
        4. Direct flight preference (e.g., 'no layovers', 'direct only')
        
        Original preferences: {original}
        User feedback: {feedback}
        
        Return a focused JSON response with only modified preferences:
        {{
            "price_range": {"max": number, "flexibility": string},
            "schedule": {"preferred_time": string, "flexibility": string},
            "airline": {{"preferred": [string], "avoided": [string]}},
            "routing": {{"direct_only": boolean, "max_stops": number}}
        }}
        """.format(
            original=json.dumps(original_preferences, indent=2),
            feedback=feedback
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a flight preference analyzer focused on extracting specific changes from feedback."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Get updated preferences
            updates = json.loads(response.choices[0].message.content)
            
            # Merge updates with original preferences
            merged_preferences = original_preferences.copy()
            
            # Update only changed preferences
            if "price_range" in updates:
                merged_preferences["budget"] = updates["price_range"]["max"]
            if "schedule" in updates:
                merged_preferences["time"] = updates["schedule"]["preferred_time"]
            if "airline" in updates:
                merged_preferences["airline"] = updates["airline"]["preferred"][0] if updates["airline"]["preferred"] else None
            if "routing" in updates:
                merged_preferences["direct_only"] = updates["routing"]["direct_only"]
            
            return merged_preferences
            
        except Exception as e:
            logger.error(f"Error updating preferences: {e}")
            return original_preferences
        
    def _matches_time_preference(self, flight_time: str, preferred_time: str) -> bool:
        """Check if flight time matches user's preference."""
        # Convert times to comparable format and check if within preferred range
        # This is a simplified version - could be expanded based on needs
        time_ranges = {
            "morning": (6, 12),
            "afternoon": (12, 18),
            "evening": (18, 23),
            "night": (0, 6)
        }
        
        try:
            flight_hour = int(flight_time.split(":")[0])
            preferred_range = time_ranges.get(preferred_time.lower())
            
            if preferred_range:
                start_hour, end_hour = preferred_range
                if start_hour <= flight_hour < end_hour:
                    return True
        except (ValueError, IndexError):
            pass
        
        return False
