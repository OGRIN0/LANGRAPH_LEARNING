from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai 
import os 

openai.api_key = os.getenv("OPENAI_API_KEY")
API_KEY = "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"

groq_model = Groq(id="llama3-70b-8192", api_key=API_KEY)

web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    markdown=True,
)

finance_agent = Agent(
    name="Finance AI Agent",
    model=groq_model,
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data"],
    markdown=True,
)

multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display the data"],
    markdown=True,
)

multi_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest news for NVIDIA",
    stream=True,
)
