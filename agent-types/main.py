import os
from langchain_classic.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun


def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        api_key=os.environ.get("GOOGLE_API_KEY")
    )
    tools = [DuckDuckGoSearchRun()]

    agent = initialize_agent(
         tools,
         llm,
         agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
         verbose=True
    )

    inputs = {"input": "What is the last highest price of gold per ounce in India?"}

    response = agent.invoke(inputs)

    print(response["output"])


if __name__ == "__main__":
    main()