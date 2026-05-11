from unittest import loader

from langchain.tools import tool
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
import requests
import json
from pydantic import BaseModel
from openai import OpenAI
import os


# ==================== SERVICE 1: WEATHER API ====================

def load_weather_json(json_data):
    text = json.dumps(json_data, indent=2)
    doc = Document(page_content=text, metadata={"source": "weather_api"})
    return [doc]

def summarize_weather_data(weather_data):
    print(f"Summarizing weather data: {weather_data}")
    # Define the structured output model
    class WeatherSummary(BaseModel):
        Temp: str
        Humidity: str
        Windspeed: str
        Summary: str
        WeatherCode: str
        WeatherDescription: str
        Tone: str
        InputTokens: int
        OutputTokens: int

    # Initialize OpenAI client
    client = OpenAI(default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')},
        base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1')

    # Choose a tone for the summary
    tone = "Casual weather report style" 

    # Define the system (developer) prompt
    system_prompt = f"""You are an expert summarizer tasked with analyzing and summarizing weather reports. Given the weather data, you will extract key information and provide a concise summary in a {tone} style.

    Your responsibilities include:
    - Extracting the temperature, humidity, wind speed, and weather conditions (including weather code and description).
    - Creating a concise and succinct summary of the weather data, no longer than 200 tokens, written in {tone} style.

    Output the results in the specified structured format."""

    # Define the user prompt with dynamic context
    user_prompt = f"""Please analyze and summarize the weather data provided below.

    {weather_data}

    Provide the structured output as specified."""

    # Make the API call with structured output
    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        text_format=WeatherSummary
    )

    # Parse the response
    weatherSummary = response.output_parsed
    print(f"Raw API response: {response}")
    print(f"Parsed Weather Summary: {weatherSummary}")

    # Display the summary
    print(f"Summary: {weatherSummary}")
    print(f"Input Tokens: {weatherSummary.InputTokens}")
    print(f"Output Tokens: {weatherSummary.OutputTokens}") 
    print(f"Tone: {weatherSummary.Tone}")

    return weatherSummary

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

        print(f"Weather API response status: {weather_response.status_code}")
        print(f"Weather API response content: {weather_response.text}")
        weather_data = load_weather_json(weather_response.json())

        print(f"\nWeather data:\n{weather_data}\n")

        weatherSummary = summarize_weather_data(weather_data)
        
        return weatherSummary
        
    except requests.Timeout:
        return "Weather service request timed out. Please try again."
    except Exception as e:
        return f"Unable to fetch weather data: {str(e)}"
