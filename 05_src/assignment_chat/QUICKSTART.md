# Quick Start Guide - Shweta's Chat

Get up and running with Shweta in 3 minutes!

## 1️⃣ Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 2️⃣ Set API Key

```bash
# macOS/Linux
export API_GATEWAY_KEY="your-api-gateway-key"
# OR
export OPENAI_API_KEY="your-openai-key"

# Windows PowerShell
$env:API_GATEWAY_KEY="your-api-gateway-key"
```

## 3️⃣ Run the App

```bash
python app.py
```

Visit `http://localhost:7861` in your browser

---

## Example Conversations

### Weather Service
**User**: "What's the weather in Tokyo?"  
**Shweta**: 🌍 Weather in Tokyo, Japan:
- Temperature: 72°F
- Conditions: Partly cloudy
- Humidity: 65%
- Wind Speed: 12 mph

### Knowledge Base
**User**: "Explain machine learning to me"  
**Shweta**: 📚 Knowledge Base Results:
1. Machine learning is a subset of artificial intelligence... (Relevance: 95%)
2. Deep learning uses neural networks with multiple layers... (Relevance: 85%)
...

### Fun Fact
**User**: "Tell me something interesting"  
**Shweta**: 🎉 Fun Fact: Honey never spoils. Archaeologists have found 3,000-year-old honey in Egyptian tombs that is still edible.

---

## Features to Try

- 💬 **Normal Chat**: Just chat naturally, Shweta will respond
- 🌍 **Weather**: Ask about weather in any city
- 📚 **Learn**: Ask about AI, machine learning, data science
- 🎉 **Fun**: Ask for facts and interesting information
- 🛡️ **Test Guardrails**: Try asking about cats or Taylor Swift

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'chromadb'" | Run `pip install -r requirements.txt` |
| "API Key not found" | Set `API_GATEWAY_KEY` or `OPENAI_API_KEY` environment variable |
| "Connection refused" | Ensure app is running at http://localhost:7861 |
| "Weather API error" | Try a different city name |

---

## Architecture

```
User Input
    ↓
Guardrail Checks (restricted topics, prompt injection)
    ↓
Service Intent Detection (weather/knowledge/tools)
    ↓
Service Execution (API call / semantic search / function)
    ↓
LLM Processing (with service results as context)
    ↓
Memory Update + Response
```

---

For detailed documentation, see `readme.md`
