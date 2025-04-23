"""
Hotel Booking Assistant Agent implementation using AutoGen.
Handles hotel search and booking using Booking.com API.
Implements message-based interaction for the multi-agent system.
"""

from typing import Dict, List, Optional, Union, Any
from .base_agent import BaseAutoGenAgent
from datetime import datetime
import requests
import urllib.parse
import time
import json
import http.client
import re
import logging
from openai import OpenAI
from config.settings import API_KEYS, AGENT_SETTINGS

logger = logging.getLogger(__name__)

class HotelBookingAssistant(BaseAutoGenAgent):
    """
    Hotel Booking Assistant that handles hotel search and recommendations
    using both OpenAI for understanding and Booking.com for actual hotel data.
    Implements message-based interaction for the multi-agent system.
    """
    
    def check_message_relevance(self, message: Union[str, Dict]) -> bool:
        """
        Check if the message is relevant to hotel booking using both LLM and keywords.
        
        Args:
            message: The message to check
            
        Returns:
            bool: True if the message is relevant
        """
        try:
            # First check if message only contains flight data
            if isinstance(message, dict):
                content = message.get("content", {})
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except:
                        pass
                
                if isinstance(content, dict):
                    request_details = content
                    # If message only contains flight data, return False
                    if "flight" in request_details and "hotel" not in request_details:
                        return False
            
            # Check if message is feedback type
            if isinstance(message, dict):
                content = message.get("content", {})
                if isinstance(content, dict) and content.get("type") == "feedback":
                    # Analyze if feedback is hotel-related
                    feedback_text = content.get("feedback", "").lower()
                    hotel_keywords = ["hotel", "room", "stay", "accommodation", "cheaper", "expensive", "location"]
                    is_relevant = any(keyword in feedback_text for keyword in hotel_keywords)
                    return is_relevant
            
            # Use LLM for more accurate relevance check
            prompt = """
            Determine if this message is specifically about hotel booking.
            Return false if the message is only about flights or other travel aspects.
            
            Message: {message}
            
            Return as a json object: {{"is_hotel_related": boolean, "reason": string}}
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
            is_relevant = result.get("is_hotel_related", False)
            
            if is_relevant:
                logger.debug(f"[DEBUG] {self.agent_type} - LLM determined message is relevant")
                return True
            
            # If LLM returns False, use stricter keyword matching as fallback
            if isinstance(message, dict):
                message_str = json.dumps(message)
            else:
                message_str = str(message)
                
            keywords = self.get_relevant_keywords()
            keyword_match = any(kw in message_str.lower() for kw in keywords)
            logger.debug(f"[DEBUG] {self.agent_type} - Keyword match result: {keyword_match}")
            return keyword_match
            
        except Exception as e:
            logger.debug(f"[DEBUG] {self.agent_type} - Error in check_message_relevance: {e}")
            return False
    
    def get_relevant_keywords(self) -> List[str]:
        """Return keywords that indicate message relevance for hotel booking."""
        return [
            "hotel",
            "accommodation",
            "room",
            "stay",
            "lodging",
            "check-in",
            "check-out"
        ]
    
    def analyze_request(self, message: Union[str, Dict]) -> Dict:
        """
        Analyze request to determine if other agents' assistance is needed.
        
        Args:
            message: Message to analyze
            
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
            if "check_in" not in content or "check_out" not in content:
                missing_info.extend(["check_in", "check_out"])
                needed_agents.append("FLIGHT_ASSISTANT")
            
            # Check location information
            if "location" not in content and "destination" not in content:
                missing_info.append("location")
                needed_agents.append("FLIGHT_ASSISTANT")
            
            # Check itinerary-related information
            if any(keyword in str(content).lower() for keyword in ["activity", "itinerary", "plan"]):
                needed_agents.append("ITINERARY_ASSISTANT")
            
            return {
                "missing_info": missing_info,
                "needed_agents": list(set(needed_agents)),  # Remove duplicates
                "can_process": len(missing_info) == 0
            }
            
        except Exception as e:
            print(f"Error analyzing request: {e}")
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
            
            if request_type == "notify_flight_selected":
                # Save flight information for hotel search
                flight_info = content.get("flight", {})
                self.task_state["flight_info"] = flight_info
                
                # Validate and clean flight information
                if not isinstance(flight_info, dict):
                    logger.warning("Invalid flight info format")
                    return json.dumps({"error": "Invalid flight information format"})
                
                # Extract destination based on flight format
                destination = None
                if "outbound" in flight_info:
                    # Round-trip format
                    outbound = flight_info["outbound"]
                    destination = outbound.get("schedule", {}).get("destination")
                    departure_date = outbound.get("schedule", {}).get("departure", "").split("T")[0]
                    return_date = flight_info.get("return", {}).get("schedule", {}).get("departure", "").split("T")[0]
                else:
                    # One-way format
                    destination = flight_info.get("schedule", {}).get("destination")
                    departure_time = flight_info.get("schedule", {}).get("departure", "")
                    departure_date = departure_time.split("T")[0] if departure_time else None
                    return_date = None
                
                logger.debug(f"[DEBUG] Raw destination from flight: {destination}")
                
                # Validate destination
                if not destination:
                    logger.warning("No destination in flight info")
                    return json.dumps({"error": "No destination in flight information"})
                
                # Clean destination
                cleaned_destination = self._clean_location_name(destination)
                if not cleaned_destination:
                    logger.warning("Invalid destination format")
                    return json.dumps({"error": f"Invalid destination format: {destination}"})
                
                logger.debug(f"[DEBUG] Flight destination '{destination}' cleaned to '{cleaned_destination}'")
                
                # Validate dates
                if not departure_date:
                    logger.warning("No departure date in flight info")
                    return json.dumps({"error": "No departure date in flight information"})
                
                # Update search parameters
                self.task_state["search_params"] = {
                    "check_in": departure_date,
                    "check_out": return_date,
                    "location": cleaned_destination
                }
                
                logger.debug(f"[DEBUG] Updated search params: {json.dumps(self.task_state['search_params'], indent=2)}")
                
                return json.dumps({
                    "status": "success",
                    "message": "Flight information received and saved"
                })
                
            elif request_type == "get_hotel_details":
                # Return selected hotel information
                if "selected" in self.task_state:
                    return json.dumps(self.task_state["selected"])
                return json.dumps({"error": "No hotel selected yet"})
                
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
        Generate response to hotel-related messages.
        
        Args:
            message: Message to respond to
            
        Returns:
            Dict: Hotel recommendations and details
        """
        try:
            logger.debug(f"[DEBUG] Received message: {json.dumps(message, indent=2)}")
            
            # First analyze if other agents' assistance is needed
            analysis_result = self.analyze_request(message)
            
            # If other agents' assistance is needed
            if analysis_result.get("needed_agents"):
                for agent_type in analysis_result["needed_agents"]:
                    if agent_type in self.collaborators:
                        if agent_type == "FLIGHT_ASSISTANT":
                            # Get flight information
                            response = self.send_request(
                                agent_type,
                                "get_flight_times",
                                {"request": message}
                            )
                            flight_info = json.loads(response)
                            
                            # Validate flight information
                            destination = flight_info.get("destination", "")
                            if not destination:
                                logger.warning("Received empty destination from flight assistant")
                                return {"error": "No destination received from flight assistant"}
                            
                            # Clean and validate destination
                            cleaned_destination = self._clean_location_name(destination)
                            if not cleaned_destination:
                                logger.warning("Cleaned destination is empty")
                                return {"error": f"Invalid destination format: {destination}"}
                            
                            logger.debug(f"[DEBUG] Flight destination from times '{destination}' cleaned to '{cleaned_destination}'")
                            
                            # Update search parameters
                            if isinstance(message, dict):
                                departure_date = flight_info.get("departure_date")
                                return_date = flight_info.get("return_date")
                                
                                if not departure_date:
                                    logger.warning("No departure date received")
                                    return {"error": "No departure date received from flight assistant"}
                                
                                message.update({
                                    "check_in": departure_date,
                                    "check_out": return_date,
                                    "location": cleaned_destination
                                })
            
            # Check if there is saved flight information
            if self.task_state.get("flight_info"):
                if isinstance(message, dict):
                    message.update(self.task_state["search_params"])
            
            # Handle feedback type messages
            if isinstance(message, dict):
                content = message.get("content", {})
                if isinstance(content, dict) and content.get("type") == "feedback":
                    logger.debug("[DEBUG] Processing feedback message")
                    return self._handle_feedback(content)
                
                # Handle regular feedback
                if "feedback" in message:
                    logger.debug("[DEBUG] Processing feedback request")
                    return self._handle_feedback(message)
                user_input = json.dumps(message)
            else:
                user_input = str(message)
            
            # Process request using existing logic
            logger.debug("[DEBUG] Processing request using process_request")
            result = self.process_request(user_input)
            
            if "error" in result:
                return {"error": result["error"]}
            
            # Get recommendations list, handling both old and new formats
            recs = result.get("recommendations", [])
            if isinstance(recs, dict):  # New format: {"recommendations": [...], "summary": "..."}
                logger.debug("[DEBUG] Found new recommendations format (dict)")
                rec_list = recs.get("recommendations", [])
            else:  # Old format: direct list
                logger.debug("[DEBUG] Found old recommendations format (list)")
                rec_list = recs
            
            # Save selected hotel information
            if rec_list:  # Only save if we have recommendations
                logger.debug(f"[DEBUG] Saving first recommendation from list of {len(rec_list)}")
                self.task_state["selected"] = rec_list[0]
            else:
                logger.debug("[DEBUG] No recommendations to save")
            
            # Notify itinerary assistant about selected hotel
            if "ITINERARY_ASSISTANT" in self.collaborators and self.task_state.get("selected"):
                self.send_request(
                    "ITINERARY_ASSISTANT",
                    "update_hotel_location",
                    {"hotel": self.task_state["selected"]}
                )
            
            return {
                "status": "success",
                "recommendations": recs,  # Keep the original format in response
                "raw_hotels": result.get("raw_hotels", {}),
                "message_type": "hotel_recommendations"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {"error": f"Failed to generate response: {str(e)}"}
    
    def process_request(self, user_input: str) -> Dict:
        """
        Process a hotel booking request from start to finish.
        
        Args:
            user_input: User's raw input text
            
        Returns:
            Dictionary containing hotel recommendations
        """
        try:
            # 1. Parse user input using OpenAI
            logger.debug("Parsing user request...")
            parsed_data = self._parse_with_openai(user_input)
            if "error" in parsed_data:
                return parsed_data
                
            logger.debug(f"Parsed request: {json.dumps(parsed_data, indent=2)}")
            
            # 2. Search for hotel options
            logger.debug("Searching for hotels...")
            search_result = self.search_hotels(parsed_data)
            if not search_result:
                return {"error": "No hotels found"}
            
            # 3. Analyze results using raw response
            logger.debug("Analyzing options...")
            recommendations = self.analyze_options(search_result, parsed_data)
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "raw_hotels": search_result
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": f"Failed to process request: {str(e)}"}
    
    def _parse_with_openai(self, text: str) -> Dict:
        """Use OpenAI to parse user input into structured data."""
        try:
            prompt = f"""Please analyze this hotel booking request and extract the following information:
            - City/Area (just the city name without any annotations)
            - Check-in date (in YYYY-MM-DD format)
            - Check-out date (in YYYY-MM-DD format)
            - Number of guests (as a number)
            - Room type
            - Maximum price per night (as a number)
            - Required amenities (comma-separated list)
            - Location preference (downtown/beach etc.)
            - Minimum star rating (as a number)
            - Other special requirements
            
            User input: {text}
            
            Please respond in natural language with the following format:
            Location: [city name only]
            Check-in: [YYYY-MM-DD]
            Check-out: [YYYY-MM-DD]
            Guests: [number]
            Room Type: [type]
            Price Limit: [number]
            Amenities: [comma-separated list]
            Area Preference: [preference]
            Star Rating: [number]
            Special Requests: [text]
            
            Example:
            Location: Boston
            Check-in: 2025-05-01
            Check-out: 2025-05-05
            Guests: 2
            Room Type: Double
            Price Limit: 300
            Amenities: wifi, pool, breakfast
            Area Preference: downtown
            Star Rating: 4
            Special Requests: quiet room away from elevator"""
            
            response = self.safe_llm_call([
                {"role": "system", "content": "You are a hotel booking request analysis expert."},
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
                    
                    if key == 'location':
                        result['location'] = value
                    elif key == 'check-in':
                        # Clean and format date
                        cleaned_date = self._clean_date(value)
                        result['check_in'] = cleaned_date
                    elif key == 'check-out':
                        # Clean and format date
                        cleaned_date = self._clean_date(value)
                        result['check_out'] = cleaned_date
                    elif key == 'guests':
                        numbers = re.findall(r'\d+', value)
                        if numbers:
                            result['guests'] = int(numbers[0])
                    elif key == 'room type':
                        result['room_type'] = value
                    elif key == 'price limit':
                        numbers = re.findall(r'\d+', value)
                        if numbers:
                            result['max_price'] = int(numbers[0])
                    elif key == 'amenities':
                        result['amenities'] = [a.strip() for a in value.split(',')]
                    elif key == 'area preference':
                        result['preferences'] = {
                            "location_type": value,
                            "star_rating": None,
                            "other": None
                        }
                    elif key == 'star rating':
                        if result.get('preferences'):
                            result['preferences']['star_rating'] = value
                    elif key == 'special requests':
                        if result.get('preferences'):
                            result['preferences']['other'] = value
            
            return result
            
        except Exception as e:
            print(f"Error parsing natural response: {e}")
            return {"error": "Failed to parse natural language response"}
    
    def _handle_feedback(self, feedback_data: Dict) -> Dict:
        """
        Handle user feedback and adjust hotel recommendations.
        
        Args:
            feedback_data: Dictionary containing feedback and original request
            
        Returns:
            Dict: Updated hotel recommendations
        """
        try:
            feedback_text = feedback_data.get("feedback", "")
            original_request = feedback_data.get("original_request", {})
            
            # Use OpenAI to understand feedback and modify request
            prompt = f"""
            Original hotel request:
            {json.dumps(original_request, indent=2)}
            
            User feedback:
            {feedback_text}
            
            Modify the original request parameters based on this feedback and return as a json object.
            Important format requirements:
            - Location should be just the city name without annotations
            - Dates must be in YYYY-MM-DD format
            - Guest count and price limits must be numbers
            - Amenities should be a comma-separated list
            
            Example format:
            {{
                "location": "Boston",
                "check_in": "2025-05-01",
                "check_out": "2025-05-05",
                "guests": 2,
                "room_type": "Double",
                "max_price": 300,
                "amenities": ["wifi", "pool", "breakfast"],
                "preferences": {{
                    "location_type": "downtown",
                    "star_rating": 4,
                    "other": "quiet room away from elevator"
                }}
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a hotel request modifier."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json"}
            )
            
            modified_request = json.loads(response.choices[0].message.content)
            
            # Clean location and dates in modified request
            if isinstance(modified_request, dict):
                if "location" in modified_request:
                    modified_request["location"] = self._clean_location_name(modified_request["location"])
                    print(f"[DEBUG] Cleaned feedback location to: {modified_request['location']}")
                
                if "check_in" in modified_request:
                    modified_request["check_in"] = self._clean_date(modified_request["check_in"])
                    print(f"[DEBUG] Cleaned feedback check-in to: {modified_request['check_in']}")
                
                if "check_out" in modified_request:
                    modified_request["check_out"] = self._clean_date(modified_request["check_out"])
                    print(f"[DEBUG] Cleaned feedback check-out to: {modified_request['check_out']}")
            
            # Process the cleaned request
            result = self.process_request(json.dumps(modified_request))
            
            return {
                "status": "success",
                "recommendations": result.get("recommendations", []),
                "raw_hotels": result.get("raw_hotels", {}),
                "message_type": "feedback_response",
                "modified_request": modified_request
            }
            
        except Exception as e:
            return {"error": f"Failed to process feedback: {str(e)}"}
    
    def __init__(
        self,
        name: Optional[str] = None,
        system_message: Optional[str] = None,
        llm_config: Optional[Dict] = None
    ):
        """初始化Hotel Booking Assistant。"""
        super().__init__(
            agent_type="HOTEL_ASSISTANT",
            name=name,
            system_message=system_message,
            llm_config=llm_config
        )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=API_KEYS["OPENAI_API_KEY"])
        self.model = AGENT_SETTINGS["BOOKING"]["MODEL"]
        
        # Initialize Booking.com API client
        self.booking_headers = {
            'x-rapidapi-key': API_KEYS["BOOKING_COM_API_KEY"],
            'x-rapidapi-host': "booking-com15.p.rapidapi.com"
        }
    
    def _clean_date(self, date_str: str) -> str:
        """Clean date string by removing annotations and formatting consistently."""
        if not date_str:
            return ""
            
        # Remove annotations in parentheses
        cleaned = re.sub(r"\s*\([^)]*\)", "", date_str)
        
        # Try to extract date in YYYY-MM-DD format
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', cleaned)
        if date_match:
            return date_match.group(1)
        
        # Try other common formats
        try:

            # Try various date formats
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
                try:
                    parsed_date = datetime.strptime(cleaned.strip(), fmt)
                    return parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Error parsing date '{cleaned}': {e}")
        
        # If all parsing attempts fail, return cleaned string
        return cleaned.strip()
    
    def _clean_location_name(self, location: str) -> str:
        """
        Clean location name by removing annotations and extra whitespace.
        
        Args:
            location: Raw location string that may contain annotations
            
        Returns:
            str: Cleaned location name
        """
        if not location:
            return ""
            
        # Remove text in parentheses and clean up whitespace
        cleaned = re.sub(r"\s*\([^)]*\)", "", location)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()
        
        logger.debug(f"Cleaned location '{location}' to '{cleaned}'")
        return cleaned
    
    def _get_city_metadata(self, city: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Get city metadata using searchDestination API.
        
        Args:
            city: City name to search for
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict with dest_id and search_type if found, None otherwise
        """
        for attempt in range(max_retries):
            try:
                # Clean city name before searching
                cleaned_city = self._clean_location_name(city)
                if not cleaned_city:
                    logger.debug("No valid city name after cleaning")
                    return None
                    
                logger.debug(f"Getting metadata for city: {cleaned_city} (attempt {attempt + 1})")
                
                # First get dest_id using searchDestination
                conn = http.client.HTTPSConnection("booking-com15.p.rapidapi.com")
                
                # Construct query string
                encoded_city = urllib.parse.quote(cleaned_city)
                endpoint = f"/api/v1/hotels/searchDestination?query={encoded_city}"
                
                logger.debug(f"Request endpoint: {endpoint}")
                
                # Make request
                conn.request("GET", endpoint, headers=self.booking_headers)
                response = conn.getresponse()
                
                logger.debug(f"Response status: {response.status}")
                
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    logger.debug(f"Location data: {json.dumps(data, indent=2)}")
                    
                    # Try different response formats
                    if "data" in data:
                        # New API format
                        for item in data.get("data", []):
                            if item.get("dest_type") == "city":
                                result = {
                                    "dest_id": item.get("dest_id"),
                                    "search_type": "CITY"
                                }
                                logger.debug(f"Found city (new format): {result}")
                                return result
                    elif isinstance(data, list):
                        # Old API format
                        for item in data:
                            if item.get("type") == "CITY":
                                result = {
                                    "dest_id": item.get("id"),
                                    "search_type": "CITY"
                                }
                                logger.debug(f"Found city (old format): {result}")
                                return result
                    
                    logger.debug(f"Response format: {list(data.keys()) if isinstance(data, dict) else 'list'}")
                    logger.debug("No city found in response")
                            
                elif response.status == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        logger.debug(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    
                logger.debug(f"No city metadata found for: {city}")
                return None
                
            except Exception as e:
                logger.error(f"Error getting city metadata: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            finally:
                if 'conn' in locals():
                    conn.close()
        
        return None
    
    def search_hotels(self, request: Dict, max_retries: int = 3) -> Dict:
        """
        Search for hotels using Booking.com API.
        
        Args:
            request: Dictionary containing search parameters
                - location: City or area
                - check_in: Check-in date (mm/dd/yyyy)
                - check_out: Check-out date (mm/dd/yyyy)
                - guests: Number of guests
                - room_type: Type of room
                - max_price: Maximum price per night
                - amenities: List of required amenities
                - min_rating: Minimum guest rating
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict containing API response with status and data/error message
        """
        
        for attempt in range(max_retries):
            try:
                # Get city metadata
                city = request.get("location")
                if not city:
                    logger.debug("No location provided in request")
                    return {"status": False, "message": "No location provided"}
                    
                metadata = self._get_city_metadata(city)
                if not metadata:
                    logger.debug(f"Could not get metadata for city: {city}")
                    return {"status": False, "message": f"Could not find city: {city}"}
                    
                # Format search parameters
                search_params = {
                    "dest_id": metadata["dest_id"],
                    "search_type": "CITY",
                    "adults": str(request.get("guests", 1)),
                    "room_qty": "1",
                    "page_number": "1",
                    "units": "metric",
                    "temperature_unit": "c",
                    "languagecode": "en-us",
                    "currency_code": "USD",
                    "location": "US"
                }
                
                # Validate and clean dates
                if "check_in" not in request:
                    logger.debug("No check-in date provided")
                    return {"status": False, "message": "No check-in date provided"}
                
                check_in = self._clean_date(request["check_in"])
                if not check_in:
                    logger.debug("Invalid check-in date format")
                    return {"status": False, "message": f"Invalid check-in date format: {request['check_in']}"}
                
                search_params["arrival_date"] = check_in
                
                if "check_out" in request:
                    check_out = self._clean_date(request["check_out"])
                    if check_out:
                        search_params["departure_date"] = check_out
                    else:
                        logger.warning(f"Invalid check-out date format: {request['check_out']}")
                
                # Add children parameters if specified
                if "children" in request:
                    children = request["children"]
                    if isinstance(children, list):
                        search_params["children_age"] = ",".join(map(str, children))
                else:
                    search_params["children_age"] = "0,17"  # Default children ages
                
                # Add price filters if specified
                if "max_price" in request:
                    search_params["price_max"] = str(request["max_price"])
                if "min_price" in request:
                    search_params["price_min"] = str(request["min_price"])
                
                logger.debug(f"Search parameters: {json.dumps(search_params, indent=2)}")
                
                # Construct query string
                query_string = "&".join([f"{k}={urllib.parse.quote(str(v))}" for k, v in search_params.items()])
                
                # Make API request
                conn = http.client.HTTPSConnection("booking-com15.p.rapidapi.com")
                endpoint = f"/api/v1/hotels/searchHotels?{query_string}"
                
                logger.debug(f"Request endpoint: {endpoint}")
                
                conn.request("GET", endpoint, headers=self.booking_headers)
                response = conn.getresponse()
                
                logger.debug(f"Response status: {response.status}")
                
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    
                    # Analyze data structure
                    logger.debug("API Response Analysis:")
                    logger.debug(f"Response type: {type(data)}")
                    logger.debug(f"Top level keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
                    
                    if isinstance(data, dict):
                        for key, value in data.items():
                            logger.debug(f"Analyzing '{key}':")
                            logger.debug(f"Type: {type(value)}")
                            if isinstance(value, dict):
                                logger.debug(f"Keys: {list(value.keys())}")
                                # Analyze first nested object structure
                                for subkey, subvalue in value.items():
                                    logger.debug(f"{subkey}: {type(subvalue)}")
                                    if isinstance(subvalue, (dict, list)) and subvalue:
                                        if isinstance(subvalue, dict):
                                            logger.debug(f"Nested keys: {list(subvalue.keys())}")
                                        else:
                                            logger.debug(f"List length: {len(subvalue)}")
                                            if isinstance(subvalue[0], dict):
                                                logger.debug(f"First item keys: {list(subvalue[0].keys())}")
                            elif isinstance(value, list) and value:
                                logger.debug(f"List length: {len(value)}")
                                if value:
                                    logger.debug(f"First item type: {type(value[0])}")
                                    if isinstance(value[0], dict):
                                        logger.debug(f"First item keys: {list(value[0].keys())}")
                    
                    # Handle different response formats
                    if isinstance(data, dict):
                        if data.get("status") is False:
                            logger.debug(f"API returned error: {data.get('message')}")
                            if attempt < max_retries - 1:
                                time.sleep(2)
                                continue
                            return data
                        
                        # Check for hotels in different paths
                        if "data" in data and "hotels" in data["data"]:
                            logger.debug(f"Found {len(data['data']['hotels'])} hotels in data.hotels")
                            return data
                        elif "hotels" in data:
                            logger.debug(f"Found {len(data['hotels'])} hotels in root")
                            return {"data": {"hotels": data["hotels"]}}
                        elif "results" in data:
                            logger.debug(f"Found {len(data['results'])} results")
                            return {"data": {"hotels": data["results"]}}
                    elif isinstance(data, list):
                        logger.debug(f"Response is a list with {len(data)} items")
                        return {"data": {"hotels": data}}
                    
                    logger.warning("Unexpected API response format")
                    logger.debug(f"Available keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
                    return data
                elif response.status == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.debug(f"Rate limit hit, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                logger.debug(f"API request failed: {response.status}")
                return {"status": False, "message": f"API request failed with status {response.status}"}
                
            except Exception as e:
                logger.error(f"Error searching hotels: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return {"status": False, "message": str(e)}
                
            finally:
                if 'conn' in locals():
                    conn.close()
        
        return {"status": False, "message": "Max retries exceeded"}
    
    def analyze_options(self, raw_hotels: Dict, preferences: Dict) -> Dict:
        """
        Use OpenAI to analyze raw hotel data based on user preferences.
        
        Args:
            raw_hotels: Raw hotel data from Booking.com API
            preferences: User preferences dictionary
        
        Returns:
            Analyzed and ranked hotel recommendations
        """
        try:
            # Get hotel list with format handling
            hotels = []
            if isinstance(raw_hotels, dict):
                if "data" in raw_hotels and "hotels" in raw_hotels["data"]:
                    hotels = raw_hotels["data"]["hotels"]
                elif "hotels" in raw_hotels:
                    hotels = raw_hotels["hotels"]
                elif "results" in raw_hotels:
                    hotels = raw_hotels["results"]
            elif isinstance(raw_hotels, list):
                hotels = raw_hotels
            
            logger.debug(f"Found {len(hotels)} hotels to analyze")
            
            if not hotels:
                logger.warning("No hotels found in response")
                return []
            
            # Limit number of hotels to analyze
            hotels_to_analyze = hotels[:20]  # Process up to 20 hotels
            logger.debug(f"Analyzing {len(hotels_to_analyze)} hotels")
            
            # Preprocess hotels data
            processed_hotels = []
            for hotel in hotels_to_analyze:
                processed = {
                    'id': hotel['hotel_id'],
                    'name': hotel['property']['name'],
                    'price': hotel['property']['priceBreakdown']['grossPrice']['value'],
                    'score': hotel['property'].get('reviewScore', 0)
                }
                processed_hotels.append(processed)
            
            hotels_json = json.dumps(processed_hotels, indent=2)
            pref_json = json.dumps(preferences, indent=2)
            
            logger.debug("Analyzing options with:")
            logger.debug(f"Number of hotels: {len(processed_hotels)}")
            logger.debug(f"Preferences: {pref_json}")
            logger.debug(f"Model: {self.model}")
            
            prompt = f"""Analyze these hotel options based on the following criteria:

            Available Hotels:
            {hotels_json}

            User Preferences:
            {pref_json}

            Selection Criteria (in order of importance):
            1. Price optimization
            2. Review score
            3. Location convenience
            4. Overall value for money

            Return exactly 3 recommendations as a json object with this structure:
            {{
              "recommendations": [
                {{
                  "id": "123",
                  "name": "Hotel Name",
                  "price": 150,
                  "score": 8.5,
                  "reason": "Brief reason for recommendation"
                }}
              ],
              "summary": "Brief comparison of the recommendations"
            }}

            Additional Requirements:
            - Each recommendation must include all specified fields
            - Focus on concrete benefits (e.g., "best value", "highest rated")
            - Include specific reasons for each recommendation
            """
            
            try:
                logger.debug("Sending request to OpenAI...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "You are a hotel recommendation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                logger.debug("Got response from OpenAI")
                content = response.choices[0].message.content
                logger.debug(f"Response content: {content}")
                
                recommendations = json.loads(content)
                logger.debug("Successfully parsed recommendations")
                
                # Validate response structure
                if not isinstance(recommendations, dict):
                    raise ValueError("OpenAI response is not a dictionary")
                
                if "recommendations" not in recommendations:
                    raise ValueError("OpenAI response missing 'recommendations' field")
                
                if not isinstance(recommendations["recommendations"], list):
                    raise ValueError("'recommendations' field is not a list")
                
                # Validate each recommendation
                for i, rec in enumerate(recommendations["recommendations"]):
                    required_fields = ["id", "name", "price", "score", "reason"]
                    if not all(field in rec for field in required_fields):
                        raise ValueError(f"Recommendation {i} missing required fields")
                
                logger.debug("OpenAI response validation successful")
                return recommendations
                
            except json.JSONDecodeError as json_error:
                logger.error(f"Failed to parse OpenAI response as JSON: {json_error}")
                return {"recommendations": [], "summary": "Error: Invalid JSON response from OpenAI"}
            except ValueError as val_error:
                logger.error(f"OpenAI response validation failed: {val_error}")
                return {"recommendations": [], "summary": f"Error: Invalid response format - {str(val_error)}"}
            except Exception as e:
                logger.error(f"Failed to analyze options: {str(e)}")
                logger.error(f"Exception type: {type(e)}")
                logger.error(f"Full exception details: {e.__dict__}")
                return {"recommendations": [], "summary": f"Error analyzing hotels: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Error analyzing options: {e}")
            return []
