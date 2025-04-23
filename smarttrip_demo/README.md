Welcome to SmartTrip! I can help you plan your journey.
You can tell me about your travel plans in natural language or provide structured details.

For example:
1. Natural: 'I want to fly from New York to London next month for a week-long vacation'
2. Structured: 'Flight: NYC to London; Hotel: Central London; Activities: Museums and Shopping'

You: I'd like to plan a trip to Nashville from 2025-05-02 to 2025-05-07. I'm currently in Providence and would love an itinerary packed with great restaurants

Response from flight_assistant:

Recommendations:
{
  "recommendations": [
    {
      "outbound": {
        "flight_id": "15638-2505021122--32586-2-10037-2505022224",
        "airline": "Allegiant Air",
        "schedule": {
          "departure": "2025-05-02T11:22:00",
          "arrival": "2025-05-02T22:24:00",
          "origin": "Providence",
          "destination": "Nashville"
        },
        "price": "$216",
        "is_direct": false
      },
      "return": {
        "flight_id": "10037-2505071335--32171-1-15638-2505072341",
        "airline": "jetBlue",
        "schedule": {
          "departure": "2025-05-07T13:35:00",
          "arrival": "2025-05-07T23:41:00",
          "origin": "Nashville",
          "destination": "Providence"
        },
        "price": "$141",
        "is_direct": false
      },
      "total_cost": "$357",
      "benefits": [
        "Best overall value",
        "Convenient midday flights"
      ],
      "recommendation_reason": "This combination offers a balance between cost and convenience, with midday flights that may be less disruptive to the user's schedule. The total cost is the lowest among the options that do not have excessively long durations."
    },
    {
      "outbound": {
        "flight_id": "15638-2505020600--30755,-31825-1-10037-2505021440",
        "airline": "Breeze Airways",
        "schedule": {
          "departure": "2025-05-02T06:00:00",
          "arrival": "2025-05-02T14:40:00",
          "origin": "Providence",
          "destination": "Nashville"
        },
        "price": "$236",
        "is_direct": false
      },
      "return": {
        "flight_id": "10037-2505070906--32573-1-15638-2505071426",
        "airline": "American Airlines",
        "schedule": {
          "departure": "2025-05-07T09:06:00",
          "arrival": "2025-05-07T14:26:00",
          "origin": "Nashville",
          "destination": "Providence"
        },
        "price": "$234",
        "is_direct": false
      },
      "total_cost": "$470",
      "benefits": [
        "Shortest overall duration",
        "Morning departures"
      ],
      "recommendation_reason": "This option provides the shortest total travel time, which minimizes the time spent in transit. Morning departures allow for more time at the destination on both travel days."
    },
    {
      "outbound": {
        "flight_id": "15638-2505021219--32573-1-10037-2505021526",
        "airline": "American Airlines",
        "schedule": {
          "departure": "2025-05-02T12:19:00",
          "arrival": "2025-05-02T15:26:00",
          "origin": "Providence",
          "destination": "Nashville"
        },
        "price": "$344",
        "is_direct": false
      },
      "return": {
        "flight_id": "10037-2505071929--32573-1-15638-2505080046",
        "airline": "American Airlines",
        "schedule": {
          "departure": "2025-05-07T19:29:00",
          "arrival": "2025-05-08T00:46:00",
          "origin": "Nashville",
          "destination": "Providence"
        },
        "price": "$234",
        "is_direct": false
      },
      "total_cost": "$578",
      "benefits": [
        "Premium airline service",
        "Convenient afternoon and evening flights"
      ],
      "recommendation_reason": "Although this is the most expensive option, it is offered by a premium airline known for better service. The flight times are convenient, allowing for a leisurely morning on the day of departure and an evening return, which may be preferred by some travelers."
    }
  ],
  "summary": "The first recommendation offers the best overall value, balancing cost and convenience. The second recommendation is ideal for travelers looking to minimize travel time. The third recommendation is for those who prefer premium service and more convenient flight times, despite a higher price.",
  "debug_info": {
    "outbound_flights": 5,
    "return_flights": 5,
    "processed_count": 3
  }
}

Response from hotel_assistant:

Recommendations:
{
  "recommendations": [
    {
      "id": "319612",
      "name": "Red Roof Inn PLUS+ Nashville North Goodlettsville",
      "price": 356.2,
      "score": 6.9,
      "reason": "Most affordable option with decent reviews, offering significant savings for budget-conscious travelers."
    },
    {
      "id": "3395672",
      "name": "Tru By Hilton Smyrna Nashville",
      "price": 562.53,
      "score": 8.5,
      "reason": "Balances a relatively low price with a high review score, providing excellent value for money."
    },
    {
      "id": "182490",
      "name": "Holiday Inn Express & Suites Nashville Southeast - Antioch by IHG",
      "price": 852,
      "score": 8.9,
      "reason": "Highest rated among the options, with a reasonable price, ensuring a quality stay for guests prioritizing reviews."
    }
  ],
  "summary": "The Red Roof Inn PLUS+ offers the lowest price for budget travelers, while Tru By Hilton Smyrna Nashville presents a balance between cost and high guest satisfaction. The Holiday Inn Express & Suites Nashville Southeast - Antioch, while more expensive, has the highest review score, making it suitable for those willing to pay a bit more for top-rated accommodations. Each option caters to different aspects of the selection criteria, ensuring a range of choices for various preferences."
}

Response from itinerary_assistant:

Proposed Itinerary:

Day 1 - 2025-05-02:
10:00: Nashville Farmers' Market
13:00: Monell's Dining & Catering
15:00: The Parthenon
19:00: Prince's Hot Chicken Shack

Day 2 - 2025-05-03:
10:00: East Nashville Art Walk
13:00: Lockeland Table
15:00: Shelby Bottoms Nature Center and Greenway
19:00: City House

Day 3 - 2025-05-04:
10:00: 12 South Neighborhood
13:00: The Flipside
15:00: Tennessee Agricultural Museum
19:00: Rolf and Daughters

Day 4 - 2025-05-05:
10:00: Marathon Village
13:00: Arnold's Country Kitchen
15:00: Fort Negley Park
19:00: Husk

Day 5 - 2025-05-06:
10:00: Germantown Neighborhood
13:00: Butchertown Hall
15:00: Bicentennial Capitol Mall State Park
19:00: The Patterson House

Day 6 - 2025-05-07:
10:00: Historic Edgefield
13:00: Mitchell Delicatessen
15:00: Riverfront Park
19:00: Margot Cafe & Bar

How does this look? Feel free to tell me if you'd like to adjust anything.

You: Actually, I'd like to plan a trip to Nashville from 2025-05-02 to 2025-05-07. I'm currently in Providence and but would love an itinerary packed with great nature resort.

Response from flight_assistant:

Recommendations:
{
  "recommendations": [
    {
      "outbound": {
        "flight_id": "15638-2505021122--32586-2-10037-2505022224",
        "airline": "Allegiant Air",
        "schedule": {
          "departure": "2025-05-02T11:22:00",
          "arrival": "2025-05-02T22:24:00",
          "origin": "Providence",
          "destination": "Nashville"
        },
        "price": "$216",
        "is_direct": false
      },
      "return": {
        "flight_id": "10037-2505071335--32171-1-15638-2505072341",
        "airline": "jetBlue",
        "schedule": {
          "departure": "2025-05-07T13:35:00",
          "arrival": "2025-05-07T23:41:00",
          "origin": "Nashville",
          "destination": "Providence"
        },
        "price": "$141",
        "is_direct": false
      },
      "total_cost": "$357",
      "benefits": [
        "Best overall value",
        "Convenient midday flights"
      ],
      "recommendation_reason": "This combination offers a balance of reasonable price and convenient midday departure times, allowing for a comfortable travel schedule without early mornings or late nights. Allegiant Air and jetBlue are reputable airlines with good service quality."
    },
    {
      "outbound": {
        "flight_id": "15638-2505020600--30755,-31825-1-10037-2505021440",
        "airline": "Breeze Airways",
        "schedule": {
          "departure": "2025-05-02T06:00:00",
          "arrival": "2025-05-02T14:40:00",
          "origin": "Providence",
          "destination": "Nashville"
        },
        "price": "$236",
        "is_direct": false
      },
      "return": {
        "flight_id": "10037-2505070906--32573-1-15638-2505071426",
        "airline": "American Airlines",
        "schedule": {
          "departure": "2025-05-07T09:06:00",
          "arrival": "2025-05-07T14:26:00",
          "origin": "Nashville",
          "destination": "Providence"
        },
        "price": "$234",
        "is_direct": false
      },
      "total_cost": "$470",
      "benefits": [
        "Shortest overall duration",
        "Morning departures"
      ],
      "recommendation_reason": "This option provides the shortest combined duration of flights, minimizing travel time. Both Breeze Airways and American Airlines have a good reputation, and the morning departures allow for more time spent at the destination on both travel days."
    },
    {
      "outbound": {
        "flight_id": "15638-2505021219--32573-1-10037-2505021526",
        "airline": "American Airlines",
        "schedule": {
          "departure": "2025-05-02T12:19:00",
          "arrival": "2025-05-02T15:26:00",
          "origin": "Providence",
          "destination": "Nashville"
        },
        "price": "$344",
        "is_direct": false
      },
      "return": {
        "flight_id": "10037-2505071929--32573-1-15638-2505080046",
        "airline": "American Airlines",
        "schedule": {
          "departure": "2025-05-07T19:29:00",
          "arrival": "2025-05-08T00:46:00",
          "origin": "Nashville",
          "destination": "Providence"
        },
        "price": "$234",
        "is_direct": false
      },
      "total_cost": "$578",
      "benefits": [
        "Shortest outbound flight",
        "Evening return"
      ],
      "recommendation_reason": "This pairing offers the shortest outbound flight duration, allowing for more time at the destination upon arrival. The evening return flight is convenient for those wanting to maximize their last day at the destination. American Airlines is known for its quality service."
    }
  ],
  "summary": "The first recommendation is the best overall value, balancing cost and convenience. The second recommendation minimizes travel time with the shortest combined flight duration. The third recommendation offers the shortest outbound flight and a convenient evening return, albeit at a higher cost. All recommendations consider reputable airlines and provide a mix of benefits to cater to different preferences.",
  "debug_info": {
    "outbound_flights": 5,
    "return_flights": 5,
    "processed_count": 3
  }
}

Response from hotel_assistant:

Recommendations:
{
  "recommendations": [
    {
      "id": "319612",
      "name": "Red Roof Inn PLUS+ Nashville North Goodlettsville",
      "price": 356.2,
      "score": 6.9,
      "reason": "Most affordable option with decent reviews, offering the best price optimization."
    },
    {
      "id": "3395672",
      "name": "Tru By Hilton Smyrna Nashville",
      "price": 562.53,
      "score": 8.5,
      "reason": "Balances a high review score with a moderate price, providing good overall value for money."
    },
    {
      "id": "182490",
      "name": "Holiday Inn Express & Suites Nashville Southeast - Antioch by IHG",
      "price": 852,
      "score": 8.9,
      "reason": "Highest rated among the options, with a location convenient to Nashville, offering a great nature resort experience."
    }
  ],
  "summary": "The Red Roof Inn PLUS+ Nashville North Goodlettsville is the most budget-friendly option with acceptable reviews, making it ideal for price-sensitive travelers. Tru By Hilton Smyrna Nashville offers a strong combination of affordability and high guest satisfaction, representing a balanced choice. The Holiday Inn Express & Suites Nashville Southeast - Antioch by IHG, while more expensive, has the highest review score, suggesting a premium experience that aligns with the desire for a great nature resort."
}

Response from itinerary_assistant:

Proposed Itinerary:

Day 1 - 2025-05-02:
10:00: Cheekwood Estate & Gardens
13:00: Cafe at Thistle Farms
15:00: The Mall at Green Hills
19:00: Margot Cafe & Bar

Day 2 - 2025-05-03:
10:00: Belle Meade Plantation
13:00: The Loveless Cafe
15:00: Radnor Lake State Park
19:00: Cafe Monell's

Day 3 - 2025-05-04:
10:00: Percy Warner Park
13:00: Biscuit Love Gulch
15:00: Shelby Bottoms Nature Center & Greenway
19:00: The Southern Steak & Oyster

Day 4 - 2025-05-05:
10:00: Frist Art Museum
13:00: Pinewood Social
15:00: Cumberland Park
19:00: Rolf and Daughters

Day 5 - 2025-05-06:
10:00: Nashville Farmers' Market
13:00: Marche Artisan Foods
15:00: Centennial Park
19:00: City House

Day 6 - 2025-05-07:
10:00: Warner Parks Nature Center
13:00: The Picnic Cafe
15:00: McKay Used Books & CDs
19:00: Lockeland Table
