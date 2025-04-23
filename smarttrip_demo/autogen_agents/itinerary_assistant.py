"""
Minimal Itinerary Assistant Agent implementation using AutoGen.
Creates basic travel itineraries using OpenAI.
"""

from typing import Dict, List, Optional, Union
from .base_agent import BaseAutoGenAgent
import json
import logging
from datetime import datetime, timedelta
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

class ItineraryAssistant(BaseAutoGenAgent):
    """
    Minimal Itinerary Assistant that creates basic travel plans using OpenAI.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        system_message: Optional[str] = None,
        llm_config: Optional[Dict] = None
    ):
        """Initialize the Itinerary Assistant."""
        super().__init__(
            agent_type="ITINERARY_ASSISTANT",
            name=name,
            system_message=system_message,
            llm_config=llm_config
        )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=API_KEYS["OPENAI_API_KEY"])
        self.model = AGENT_SETTINGS["BOOKING"]["MODEL"]
        
        # Initialize feedback tracking
        self.task_state = {
            "original_request": {
                "destination": "",
                "start_date": "",
                "end_date": "",
                "preferences": {},
                "flight_info": {},  # Flight information
                "hotel_info": {},   # Hotel information
                "locations": {      # All relevant locations
                    "origin": "",
                    "destination": "",
                    "hotel_location": "",
                    "activity_locations": []
                }
            },
            "current_state": {
                "destination": "",
                "dates": {},
                "preferences": {},
                "itinerary": {},
                "locations": {
                    "origin": "",
                    "destination": "",
                    "hotel_location": "",
                    "activity_locations": []
                }
            },
            "feedback_history": [],  # History of feedback iterations
            "current_preferences": {},  # Current accumulated preferences
            "selected": {}  # Currently selected itinerary
        }
    
    def check_message_relevance(self, message: Union[str, Dict]) -> bool:
        """Check if message is relevant to itinerary planning."""
        try:
            # Convert message to string for keyword checking
            if isinstance(message, dict):
                msg_str = json.dumps(message).lower()
            else:
                msg_str = str(message).lower()
            
            # Check for itinerary-related keywords
            keywords = [
                "itinerary", "activities", "schedule", "plan",
                "visit", "sightseeing", "tour", "restaurants"
            ]
            return any(kw in msg_str for kw in keywords)
            
        except Exception as e:
            logger.error(f"Error in check_message_relevance: {e}")
            return False
    
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
            if "start_date" not in content or "end_date" not in content:
                missing_info.extend(["start_date", "end_date"])
                needed_agents.append("FLIGHT_ASSISTANT")
            
            # Check location information
            if "destination" not in content:
                missing_info.append("destination")
                needed_agents.append("FLIGHT_ASSISTANT")
            
            # Check hotel information
            if "hotel" not in content:
                needed_agents.append("HOTEL_ASSISTANT")
            
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
            
            if request_type == "get_trip_duration":
                # Return trip duration information
                if "selected" in self.task_state:
                    itinerary = self.task_state["selected"]
                    return json.dumps({
                        "start_date": itinerary["dates"]["start"],
                        "end_date": itinerary["dates"]["end"]
                    })
                return json.dumps({"error": "No itinerary selected yet"})
                
            elif request_type == "update_hotel_location":
                # Update hotel location information
                hotel_info = content.get("hotel", {})
                self.task_state["hotel_info"] = hotel_info
                
                # If itinerary exists, update activity locations
                if "selected" in self.task_state:
                    self._update_activities_for_hotel(hotel_info)
                
                return json.dumps({
                    "status": "success",
                    "message": "Hotel information received and itinerary updated"
                })
                
            elif request_type == "get_information":
                # Handle general information request
                if isinstance(content, dict) and "request" in content:
                    result = self.create_itinerary(content["request"])
                    return json.dumps(result)
                
            return json.dumps({"error": f"Unknown request type: {request_type}"})
            
        except Exception as e:
            return json.dumps({"error": f"Error handling request: {str(e)}"})

    def _update_activities_for_hotel(self, hotel_info: Dict) -> None:
        """
        Update itinerary activities based on hotel location.
        
        Args:
            hotel_info: Hotel information
        """
        try:
            if "selected" not in self.task_state:
                return
                
            itinerary = self.task_state["selected"]
            hotel_location = hotel_info.get("location", "")
            
            # Use OpenAI to replan activities considering hotel location
            prompt = f"""Analyze and optimize this itinerary based on hotel location.

            Hotel Location: {hotel_location}
            
            Current Itinerary:
            {json.dumps(itinerary, indent=2)}
            
            Please provide a JSON response with optimized activities considering:
            1. Distance from hotel
            2. Efficient route planning
            3. Time of day for each activity
            
            The response should maintain the exact same JSON structure as the input.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a travel planning expert. Return your response as a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            updated_itinerary = json.loads(response.choices[0].message.content)
            self.task_state["selected"] = updated_itinerary
            
        except Exception as e:
            logger.error(f"Error updating activities for hotel: {e}")

    def _extract_request_from_itinerary(self, itinerary: Dict) -> Dict:
        """
        Extract request parameters from existing itinerary.
        
        Args:
            itinerary: Current itinerary
            
        Returns:
            Dict: Extracted request parameters
        """
        try:
            # Extract basic information
            request = {
                "destination": itinerary.get("destination", ""),
                "start_date": itinerary.get("dates", {}).get("start", ""),
                "end_date": itinerary.get("dates", {}).get("end", ""),
                "interests": [],
                "preferences": {
                    "dining": "",
                    "activities": "",
                    "style": "balanced"
                }
            }
            
            # Analyze current activities to infer preferences
            activities = []
            restaurants = []
            for day in itinerary.get("itinerary", []):
                for activity in day.get("activities", []):
                    time = activity.get("time", "")
                    name = activity.get("activity", "")
                    if time in ["13:00", "19:00"]:
                        restaurants.append(name)
                    else:
                        activities.append(name)
            
            # Use OpenAI to analyze activities and infer preferences
            prompt = f"""
            Analyze these activities and restaurants to infer travel preferences:
            
            Activities: {json.dumps(activities, indent=2)}
            Restaurants: {json.dumps(restaurants, indent=2)}
            
            Return a JSON object with inferred preferences:
            {{
                "interests": ["inferred", "interests"],
                "preferences": {{
                    "dining": "inferred dining style",
                    "activities": "inferred activity preferences",
                    "style": "inferred overall style"
                }}
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a travel preference analyzer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            inferred = json.loads(response.choices[0].message.content)
            request["interests"] = inferred.get("interests", [])
            request["preferences"].update(inferred.get("preferences", {}))
            
            return request
            
        except Exception as e:
            logger.error(f"Error extracting request from itinerary: {e}")
            return {}

    def _merge_preferences(self, current_prefs: Dict, new_prefs: Dict) -> Dict:
        """
        Merge current preferences with new preferences from feedback.
        
        Args:
            current_prefs: Current preference settings
            new_prefs: New preferences from feedback
            
        Returns:
            Dict: Merged preferences
        """
        merged = current_prefs.copy()
        
        # Update interests
        current_interests = set(current_prefs.get("interests", []))
        new_interests = set(new_prefs.get("interests", []))
        merged["interests"] = list(current_interests.union(new_interests))
        
        # Update preferences
        current_style = current_prefs.get("preferences", {})
        new_style = new_prefs.get("preferences", {})
        
        merged["preferences"] = {
            "dining": new_style.get("dining") or current_style.get("dining", ""),
            "activities": new_style.get("activities") or current_style.get("activities", ""),
            "style": new_style.get("style") or current_style.get("style", "balanced")
        }
        
        return merged

    def _handle_feedback(self, feedback_data: Dict) -> Dict:
        """
        Handle user feedback and adjust itinerary based on preferences.
        
        Args:
            feedback_data: Dictionary containing feedback and original request
            
        Returns:
            Dict: Updated itinerary plan
        """
        try:
            feedback_text = feedback_data.get("feedback", "")
            
            # Get current preferences
            current_prefs = self.task_state.get("current_preferences", {})
            if not current_prefs and "selected" in self.task_state:
                current_prefs = self._extract_request_from_itinerary(self.task_state["selected"])
            
            if not current_prefs:
                logger.error("No current preferences found")
                return {"error": "Could not find current preferences"}
            
            # Save feedback to history
            self.task_state["feedback_history"].append({
                "feedback": feedback_text,
                "timestamp": datetime.now().isoformat(),
                "current_prefs": current_prefs.copy()
            })
            
            # Use OpenAI to understand feedback and modify preferences
            prompt = f"""
            Current travel preferences:
            {json.dumps(current_prefs, indent=2)}
            
            Previous feedback history:
            {json.dumps(self.task_state["feedback_history"], indent=2)}
            
            New feedback:
            {feedback_text}
            
            Analyze the feedback history and new feedback to modify the preferences.
            Consider:
            1. All previous feedback and their impact on preferences
            2. The new feedback's specific requests
            3. How to maintain consistency with previous changes
            4. The overall direction of all feedback
            
            Important:
            - Each feedback should build upon previous changes
            - Maintain the good elements from previous iterations
            - Be specific about new adjustments needed
            
            Return a JSON object with the modified preferences:
            {{
                "destination": "same as original",
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD",
                "interests": ["cumulative", "interests", "list"],
                "preferences": {{
                    "dining": "cumulative dining preferences",
                    "activities": "cumulative activity preferences",
                    "style": "overall style considering all feedback"
                }}
            }}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a travel preference analyzer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            modified_request = json.loads(response.choices[0].message.content)
            logger.debug(f"Modified request based on feedback: {json.dumps(modified_request, indent=2)}")
            
            # Merge with current preferences
            merged_request = self._merge_preferences(current_prefs, modified_request)
            logger.debug(f"Merged preferences: {json.dumps(merged_request, indent=2)}")
            
            # Update current preferences
            self.task_state["current_preferences"] = merged_request
            
            # Generate new itinerary with merged preferences
            result = self.create_itinerary(merged_request)
            
            if "error" in result:
                return {"error": result["error"]}
            
            # Save the new itinerary
            self.task_state["selected"] = result
            
            # If hotel information exists, update activity locations
            if self.task_state.get("hotel_info"):
                self._update_activities_for_hotel(self.task_state["hotel_info"])
                result = self.task_state["selected"]
            
            # Add success status to feedback history
            self.task_state["feedback_history"][-1]["result"] = "success"
            
            return {
                "status": "success",
                "itinerary": result.get("itinerary", []),
                "trip_details": {
                    "destination": result.get("destination", ""),
                    "duration": result.get("duration", 0),
                    "dates": result.get("dates", {}),
                    "trip_type": "leisure"
                },
                "message_type": "feedback_response",
                "modified_preferences": modified_request.get("preferences", {})
            }
            
        except Exception as e:
            logger.error(f"Error handling feedback: {e}")
            return {"error": f"Failed to process feedback: {str(e)}"}

    def generate_response(self, message: Union[str, Dict]) -> Dict:
        """
        Generate response to itinerary-related messages.
        
        Args:
            message: Message to respond to
            
        Returns:
            Dict: Itinerary plan and details
        """
        try:
            # Handle feedback type messages
            if isinstance(message, dict):
                content = message.get("content", {})
                if isinstance(content, dict) and content.get("type") == "feedback":
                    logger.debug("Processing feedback message")
                    return self._handle_feedback(content)
                
                # Handle regular feedback
                if "feedback" in message:
                    logger.debug("Processing feedback request")
                    return self._handle_feedback(message)
            
            # Extract message content
            if isinstance(message, dict) and "content" in message:
                message = message["content"]
            
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
                            # Update date information
                            if isinstance(message, dict):
                                message.update({
                                    "start_date": flight_info.get("departure_date"),
                                    "end_date": flight_info.get("return_date")
                                })
                        elif agent_type == "HOTEL_ASSISTANT":
                            # Get hotel information
                            response = self.send_request(
                                agent_type,
                                "get_hotel_details",
                                {"request": message}
                            )
                            hotel_info = json.loads(response)
                            if isinstance(message, dict):
                                message["hotel"] = hotel_info
            
            # Parse request
            request_data = self._parse_with_openai(message)
            logger.debug(f"Parsed request data: {json.dumps(request_data, indent=2)}")
            
            if "error" in request_data:
                return {"error": request_data["error"]}
            
            # Save original request if not exists
            if not self.task_state["original_request"]:
                self.task_state["original_request"] = request_data.copy()
                self.task_state["current_preferences"] = request_data.copy()
                logger.debug("Saved original request and initialized preferences")
            
            # Generate itinerary
            result = self.create_itinerary(request_data)
            logger.debug(f"Generated itinerary result: {json.dumps(result, indent=2)}")
            
            if "error" in result:
                return {"error": result["error"]}
            
            # Save generated itinerary
            self.task_state["selected"] = result
            
            # If hotel information exists, update activity locations
            if self.task_state.get("hotel_info"):
                self._update_activities_for_hotel(self.task_state["hotel_info"])
                result = self.task_state["selected"]
            
            return {
                "status": "success",
                "itinerary": result.get("itinerary", []),
                "trip_details": {
                    "destination": result.get("destination", ""),
                    "duration": result.get("duration", 0),
                    "dates": result.get("dates", {}),
                    "trip_type": "leisure"
                },
                "message_type": "itinerary_plan"
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": f"Failed to generate response: {str(e)}"}
    
    def _parse_with_openai(self, text: str) -> Dict:
        """Parse user input to extract itinerary requirements."""
        try:
            # If JSON format, parse directly
            if isinstance(text, str) and text.strip().startswith("{"):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            
            # Use OpenAI to parse natural language
            prompt = f"""Extract travel itinerary details from this request:

            User input: {text}

            Return ONLY a JSON object with these fields:
            {{
                "destination": "city name",
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD",
                "interests": ["interest1", "interest2", ...],
                "preferences": {{
                    "dining": "any food preferences",
                    "activities": "any activity preferences",
                    "style": "overall trip style (e.g., cultural, adventurous, relaxed, local, luxury)"
                }}
            }}

            Style categories to consider:
            - cultural/artistic: Museums, galleries, theaters, historical sites
            - adventurous/outdoor: Parks, hiking, sports, adventure activities
            - relaxed: Cafes, shopping, leisurely activities
            - local experience: Authentic venues, off-the-beaten-path spots
            - luxury/upscale: High-end restaurants, exclusive experiences
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are a travel request parser."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            parsed = json.loads(response.choices[0].message.content)
            logger.debug(f"Parsed request: {json.dumps(parsed, indent=2)}")
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing request: {e}")
            return {"error": f"Failed to parse request: {str(e)}"}
    
    def create_itinerary(self, request: Dict) -> Dict:
        """Create basic daily itinerary."""
        try:
            logger.debug(f"Creating itinerary with request: {json.dumps(request, indent=2)}")
            
            # Validate required fields
            destination = request.get("destination", "")
            if not destination:
                return {"status": "error", "error": "No destination specified"}
            
            # Parse dates
            try:
                start_date = datetime.strptime(request.get("start_date", ""), "%Y-%m-%d")
                end_date = datetime.strptime(request.get("end_date", ""), "%Y-%m-%d")
            except ValueError:
                return {"status": "error", "error": "Invalid or missing dates"}
            
            days = (end_date - start_date).days + 1
            interests = request.get("interests", [])
            preferences = request.get("preferences", {})
            
            # Create itinerary using OpenAI
            prompt = f"""Create a {days}-day itinerary for {destination} based on these preferences:

            Interests: {', '.join(interests) if interests else 'General sightseeing'}
            Dining preferences: {preferences.get('dining', 'No specific preferences')}
            Activity preferences: {preferences.get('activities', 'No specific preferences')}
            Style: {preferences.get('style', 'Balanced mix of activities')}
            Duration: {days} days

            Important style considerations:
            - If style is cultural/artistic:
              * Morning activities: Museums, art galleries, historical sites
              * Afternoon activities: Theaters, cultural centers, architectural tours
              * Restaurants: Mix of fine dining and authentic local cuisine
            
            - If style is adventurous/outdoor:
              * Morning activities: Hiking trails, nature reserves, adventure sports
              * Afternoon activities: Parks, gardens, outdoor attractions
              * Restaurants: Casual dining with local flavor
            
            - If style is relaxed:
              * Morning activities: Cafes, markets, easy walking tours
              * Afternoon activities: Shopping areas, parks, leisure spots
              * Restaurants: Comfortable, relaxed atmosphere
            
            - If style is local experience:
              * Morning activities: Local markets, neighborhood walks
              * Afternoon activities: Off-the-beaten-path attractions, community spaces
              * Restaurants: Authentic local establishments, hidden gems
            
            - If style is luxury/upscale:
              * Morning activities: Private tours, exclusive attractions
              * Afternoon activities: High-end shopping, spa experiences
              * Restaurants: Fine dining, acclaimed establishments

            Activity Distribution:
            - Maintain consistent style throughout the itinerary
            - Each day should have at least one signature activity matching the style
            - Mix in complementary activities that enhance the main style
            - For nature/outdoor requests, prioritize outdoor venues and natural attractions
            
            For each day, provide activities in this exact format:
            10:00: [Activity/Venue name] - Choose morning activities that match the style preference
            13:00: [Restaurant name] - Select restaurants that align with dining preferences
            15:00: [Activity/Venue name] - Afternoon activities should complement morning ones
            19:00: [Restaurant name] - Evening dining should offer variety from lunch

            Important:
            - Use exact times (10:00, 13:00, 15:00, 19:00)
            - Provide just the name without adding 'at' or other connecting words
            - For restaurants, just use the restaurant name
            - For activities, use the venue or attraction name
            
            Example:
            10:00: Country Music Hall of Fame and Museum
            13:00: Arnold's Country Kitchen
            15:00: Centennial Park
            19:00: The Catbird Seat
            
            Return in this JSON format:
            {{
                "itinerary": [
                    {{
                        "day": 1,
                        "date": "YYYY-MM-DD",
                        "activities": [
                            {{
                                "time": "10:00",
                                "activity": "Country Music Hall of Fame and Museum"
                            }},
                            {{
                                "time": "13:00",
                                "activity": "Arnold's Country Kitchen"
                            }},
                            {{
                                "time": "15:00",
                                "activity": "Centennial Park"
                            }},
                            {{
                                "time": "19:00",
                                "activity": "The Catbird Seat"
                            }}
                        ]
                    }},
                    ...
                ]
            }}

            Note: Each activity should be just the name without any additional words like 'at' or 'visit'.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a travel planning expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Process response
            result = json.loads(response.choices[0].message.content)
            
            # Update dates
            current_date = start_date
            for day in result["itinerary"]:
                day["date"] = current_date.strftime("%Y-%m-%d")
                current_date += timedelta(days=1)
            
            return {
                "status": "success",
                "destination": destination,
                "duration": days,
                "dates": {
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d")
                },
                "itinerary": result["itinerary"]
            }
            
        except Exception as e:
            logger.error(f"Error creating itinerary: {e}")
            return {
                "status": "error",
                "error": f"Failed to create itinerary: {str(e)}"
            }
