from langchain.tools import tool
import requests


# ==================== SERVICE 1: WEATHER API ====================
@tool
def get_weather(location: str) -> str:
    """
    Get current weather for a specific location.
    Uses the Open-Meteo Weather API.
    
    Args:
        location: City or location name (e.g., "London", "New York")
    
    Returns:
        Weather information including temperature, conditions, humidity, and wind speed
    """
    try:
        if not location or location.lower() in ["weather"]:
            location = "New York"  # Default location
        
        # Use Open-Meteo Geocoding API to get coordinates
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": location, "count": 1, "language": "en", "format": "json"}
        
        geo_response = requests.get(geo_url, params=geo_params, timeout=5)
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return f"I couldn't find weather information for '{location}'. Try another location."
        
        # Get coordinates
        result = geo_data["results"][0]
        latitude = result["latitude"]
        longitude = result["longitude"]
        name = result.get("name", location)
        country = result.get("country", "")
        
        # Get weather data
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "temperature_unit": "fahrenheit"
        }
        
        weather_response = requests.get(weather_url, params=weather_params, timeout=5)
        weather_data = weather_response.json()
        
        if "current" not in weather_data:
            return "Unable to fetch weather data at this time."
        
        current = weather_data["current"]
        
        # Transform data to natural language
        temp = current["temperature_2m"]
        humidity = current["relative_humidity_2m"]
        wind_speed = current["wind_speed_10m"]
        
        # Simple weather code interpretation
        weather_codes = {
            0: "clear sky",
            1: "mostly clear",
            2: "partly cloudy",
            3: "overcast",
            45: "foggy",
            48: "foggy with rime",
            51: "light drizzle",
            53: "moderate drizzle",
            55: "dense drizzle",
            61: "slight rain",
            63: "moderate rain",
            65: "heavy rain",
            71: "slight snow",
            73: "moderate snow",
            75: "heavy snow",
            80: "rain showers",
            81: "moderate rain showers",
            82: "violent rain showers",
            85: "snow showers",
            86: "heavy snow showers",
            95: "thunderstorm",
            96: "thunderstorm with hail",
            99: "severe thunderstorm"
        }
        
        weather_description = weather_codes.get(current["weather_code"], "unknown conditions")
        
        response = f"🌍 Weather in {name}{', ' + country if country else ''}:\n"
        response += f"Temperature: {temp}°F\n"
        response += f"Conditions: {weather_description.capitalize()}\n"
        response += f"Humidity: {humidity}%\n"
        response += f"Wind Speed: {wind_speed} mph"
        
        return response
        
    except requests.Timeout:
        return "Weather service request timed out. Please try again."
    except Exception as e:
        return f"Unable to fetch weather data: {str(e)}"
