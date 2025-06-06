import os
from dotenv import load_dotenv
from agents import Agent , Runner , OpenAIChatCompletionsModel , AsyncOpenAI , RunConfig
import chainlit as cl

import asyncio
load_dotenv()

api_key = os.getenv('Google_Api')
if not api_key:
    raise ValueError("Google_Api environment variable not set.")
 # Initialize the agent with the API key

async def main():
   external_config = AsyncOpenAI(
       api_key=api_key,
       base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
   ) 
   
   model = OpenAIChatCompletionsModel(
       openai_client=external_config,
       model="gemini-1.5-flash",
       ) 
   
   client = RunConfig(
       model=model,
       model_provider=external_config,
       tracing_disabled=True,
   )
    # Create an agent and run it with a sample task
   agent =  Agent(
       name = "Test Agent",
       model=model,
       instructions="You are a test agent. You will be given a task to complete. You will be given a list of tools that you can use. You will be given a list of inputs that you can use. You will be given a list of outputs that you can use. Your job is to use the tools to get the job done. If you are unsure of the answer, just say 'I don't know'.",
   )
   result = await Runner.run(agent , "What is the capital of France?")
   print(result)
   
if __name__ == "__main__":
    asyncio.run(main())
