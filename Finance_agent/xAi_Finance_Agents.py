import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.os import AgentOS 
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini model
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    exit(1)

# Create the AI finance agent
try:
   agent = Agent(
    name="Gemini Finance Agent",
    model=gemini_model,
    tools=[
        DuckDuckGoTools(),
        YFinanceTools()  # ✅ no parameters needed in Agno 2.x
    ],
    instructions=[
        "Always use tables to display financial/numerical data. "
        "For text data, use bullet points and short paragraphs."
    ],
    markdown=True,
)

except Exception as e:
    print(f"Error initializing agent: {e}")
    exit(1)

# ✅ Updated UI handling for Agno 2.x
try:
    agent_os = AgentOS(agents=[agent])
    app = agent_os.get_app()
except Exception as e:
    print(f"Error setting up AgentOS UI: {e}")
    exit(1)

if __name__ == "__main__":
    try:
        # serve the app (like serve_playground_app previously did)
        agent_os.serve(app="xAi_Finance_Agents:app", reload=True)
    except Exception as e:
        print(f"Error serving app: {e}")
        exit(1)
