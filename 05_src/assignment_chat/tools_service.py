from langchain.tools import tool
import json
import random


# ==================== SERVICE 3: TOOLS & CALCULATIONS ====================
@tool
def calculate_statistics(numbers: list) -> str:
    """
    Calculate basic statistics on a list of numbers.
    
    Args:
        numbers: List of numbers to analyze
    
    Returns:
        Statistics including count, sum, average, min, and max
    """
    if not numbers:
        return "Error: No numbers provided"
    
    stats = {
        "count": len(numbers),
        "sum": sum(numbers),
        "average": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }
    
    return f"📊 Statistics:\n{json.dumps(stats, indent=2)}"


@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert temperature between Celsius, Fahrenheit, and Kelvin.
    
    Args:
        value: Temperature value to convert
        from_unit: Source unit (C, F, or K)
        to_unit: Target unit (C, F, or K)
    
    Returns:
        Converted temperature value
    """
    try:
        # Convert to Celsius first
        if from_unit.upper() == "F":
            celsius = (value - 32) * 5 / 9
        elif from_unit.upper() == "K":
            celsius = value - 273.15
        else:
            celsius = value
        
        # Convert from Celsius to target unit
        if to_unit.upper() == "F":
            result = celsius * 9 / 5 + 32
        elif to_unit.upper() == "K":
            result = celsius + 273.15
        else:
            result = celsius
        
        return f"🌡️ Temperature Conversion: {value} {from_unit.upper()} → {result:.2f} {to_unit.upper()}"
    except Exception as e:
        return f"Error converting temperature: {str(e)}"


@tool
def generate_random_fact() -> str:
    """
    Generate a random interesting fact.
    
    Returns:
        A random fact string
    """
    facts = [
        "Honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs that is still edible.",
        "A group of flamingos is called a 'flamboyance'.",
        "The Eiffel Tower can be 15 cm taller during the summer due to thermal expansion.",
        "Octopuses have three hearts and blue blood.",
        "A day on Venus is longer than its year.",
        "The smell of fresh-cut grass is actually a plant defense mechanism.",
        "Coffee is the second most traded commodity in the world, after crude oil.",
        "Bananas are berries, but strawberries aren't.",
        "The human brain uses 20% of the body's energy despite being only 2% of body weight.",
        "A group of owls is called a 'parliament'."
    ]
    fact = random.choice(facts)
    return f"🎯 Fun Fact: {fact}"
