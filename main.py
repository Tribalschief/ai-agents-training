import os
import asyncio
from uuid import uuid4
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, WebSearchTool, set_default_openai_client
from google import generativeai as genai

# Step 1: Load environment variables
load_dotenv()
google_api_key = os.getenv('Google_Api')

if not google_api_key:
    raise ValueError("Google_Api environment variable not set.")

# Step 2: Custom guardrail implementation (fallback since Guardrail import failed)
class CustomGuardrail:
    def __init__(self, name, description, validate, error_message):
        self.name = name
        self.description = description
        self.validate = validate
        self.error_message = error_message

    def apply(self, input_str):
        if not self.validate(input_str):
            raise ValueError(self.error_message)
        return input_str

# Step 3: Custom model adapter for Gemini
class GeminiModel:
    def __init__(self, model_name="gemini-2.0-flash", api_key=google_api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_tokens = 1000
        self.temperature = 0.5

    async def generate(self, prompt, tools=None):
        try:
            # Extract content from prompt (mimicking OpenAI SDK structure)
            content = prompt.get("messages", [{}])[-1].get("content", "")
            if tools and isinstance(tools, list):
                content += "\nNote: Use web search tool if required."
            response = self.model.generate_content(
                content,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            )
            return {"choices": [{"message": {"content": response.text}}]}
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

# Step 4: Configure a dummy OpenAI client (required by the SDK)
client = AsyncOpenAI(
    api_key="dummy-key",  # Not used, as Gemini handles requests
    base_url="https://dummy-url.com",  # Placeholder
    default_headers={"x-thread-id": str(uuid4()), "x-run-id": str(uuid4())}
)
set_default_openai_client(client, use_for_tracing=False)

# Step 5: Define agent instructions
weather_instructions = """
You are a helpful Weather Assistant AI. Your purpose is to provide accurate and useful weather information to users.

When responding to queries:
1. Always ask for the user's location if not provided
2. Provide current weather conditions including temperature, humidity, and precipitation chance
3. Include a short forecast for the next 24 hours
4. For longer forecasts, summarize key weather patterns
5. Alert users to any severe weather warnings in their area
6. Answer weather-related questions clearly and concisely
7. Use friendly, conversational language
8. If you don't have data for a specific location, acknowledge this limitation
9. Suggest appropriate clothing or activities based on weather conditions when relevant
10. Cite your weather data source when providing information
"""

research_instructions = """
You are a research assistant. Use the web search tool to find accurate and up-to-date information to answer user queries. Provide concise and relevant responses.
"""

# Step 6: Define a guardrail for input validation
input_guardrail = CustomGuardrail(
    name="InputValidation",
    description="Ensure the query is not empty",
    validate=lambda x: bool(x.strip()),
    error_message="Error: Query cannot be empty."
)

# Step 7: Create the agents
weather_agent = Agent(
    name="WeatherAssistant",
    model=GeminiModel(model_name="gemini-2.0-flash"),
    instructions=weather_instructions
)

research_agent = Agent(
    name="ResearchAgent",
    model=GeminiModel(model_name="gemini-2.0-flash"),
    instructions=research_instructions,
    tools=[WebSearchTool()]
)

# Step 8: Define a function to run the agent asynchronously
async def run_agent(agent, query):
    try:
        # Apply guardrail manually for research agent
        if agent.name == "ResearchAgent":
            input_guardrail.apply(query)
        result = await Runner.run(agent, query)
        return result.final_output
    except Exception as e:
        return f"Error running agent: {str(e)}"

# Step 9: Main function to test the agents
async def main():
    test_queries = [
        ("research_agent", "What are the latest advancements in AI agents as of June 2025?"),
        ("weather_agent", "What is the weather in New York today?"),
        ("research_agent", "How does Gemini 2.0 Flash compare to GPT-4o?"),
        ("research_agent", "Please research and summarize: AI advancements June 2025. Only return the found links with very minimal text.")
    ]

    for agent_name, query in test_queries:
        agent = research_agent if agent_name == "research_agent" else weather_agent
        print(f"\nQuery: {query}")
        response = await run_agent(agent, query)
        print(f"Response: {response}")

# Step 10: Run the script
if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\nTracing may be limited without LangDB. Check OpenAI Agents SDK logs for details.")
    except Exception as e:
        print(f"Error: {str(e)}")