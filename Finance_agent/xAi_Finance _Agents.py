import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground, serve_playground_app
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini model
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")  # Example: Use Gemini 1.5 Pro
except Exception as e:
    print(f"Error configuring Gemini: {e}")
    exit(1)

# Create the AI finance agent
try:
    # Note: This assumes agno can accept a custom model object or has a Gemini adapter.
    # If agno requires a specific model class, consult agno documentation.
    agent = Agent(
        name="Gemini Finance Agent",
        model=gemini_model,  # Replace with agno-compatible Gemini model if available
        tools=[
            DuckDuckGoTools(),
            YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)
        ],
        instructions=["Always use tables to display financial/numerical data. For text data use bullet points and small paragraphs."],
        show_tool_calls=True,
        markdown=True,
    )
except Exception as e:
    print(f"Error initializing agent: {e}")
    exit(1)

# UI for finance agent
try:
    app = Playground(agents=[agent]).get_app()
except Exception as e:
    print(f"Error setting up Playground UI: {e}")
    exit(1)

if __name__ == "__main__":
    try:
        serve_playground_app("gemini_finance_agent:app", reload=True)
    except Exception as e:
        print(f"Error serving app: {e}")
        exit(1)
