# LangChain Agent Types

LangChain agents are classified by their **reasoning strategy** — how they decide which tool to use, in what order, and when to stop.
There are **7 core agent types** in LangChain.

## 1. Zero-Shot ReAct Agent

It implements the **ReAct** (Reasoning + Acting) pattern. The LLM reasons in a `Thought → Action → Observation` loop until it reaches a final answer — with no memory of prior conversations. That means the agent considers one single interaction with the agent — it will have no memory.

"Zero-shot" means it selects tools based solely on their description — no examples needed.

**The ReAct loop:****Code example:**

```python
import os
from langchain_classic.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=os.environ.get("GOOGLE_API_KEY")
)
tools = [DuckDuckGoSearchRun()]

agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
)

inputs = {"input": "What is the last highest price of gold per ounce in India?"}

response = agent.invoke(inputs)
```

The LLM's scratchpad looks like this at runtime:

```text
Thought: I need to find the current gold price. I should use the search tool.
Action: duckduckgo_search
Action Input: "current gold price per ounce 2025"
Observation: Gold is trading at $3,245/oz as of May 2025...
Thought: I now have the answer.
Final Answer: Gold is currently trading at approximately $3,245 per ounce.
```

**When to use:** General-purpose question answering with tools; no chat history needed; single-session tasks.

**Limitation:** No memory — each call is stateless. Long chains of reasoning can drift.

---

## 2. Conversational ReAct Agent

Same as Zero-Shot ReAct but adds a **`ConversationBufferMemory`** (or other memory types) to maintain chat history. The agent can reference earlier turns in the conversation.

**Key difference** — the system prompt includes prior exchanges so the LLM can reason with context like "you already told me you like Python, so I'll suggest Python tools."

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

agent.run("My name is Nitesh. What is LangChain?")
agent.run("What was the first thing I told you about myself?")
# Agent remembers: "Your name is Nitesh"
```

**When to use:** Chatbots, multi-turn assistants, any agent that needs cross-turn context.

**Memory options you can swap in:** `ConversationBufferWindowMemory`, `ConversationSummaryMemory`, `ConversationEntityMemory`, `VectorStoreRetrieverMemory`.

---

## 3. Structured Input ReAct Agent

The standard ReAct agent only supports **single-string inputs** to tools (e.g., a search query). The Structured Input variant allows tools to accept **multi-argument, typed inputs** via Pydantic schemas.

This is essential for tools like database query builders, API callers, or file operations where you need to pass `query`, `limit`, `filters` etc. simultaneously.

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel

class DBQueryInput(BaseModel):
    table: str
    filter_column: str
    filter_value: str
    limit: int = 10

def query_database(table: str, filter_column: str,
                   filter_value: str, limit: int) -> str:
    # Execute actual DB query
    return f"Found {limit} rows in {table} where {filter_column}={filter_value}"

db_tool = StructuredTool.from_function(
    func=query_database,
    name="query_database",
    description="Query a database table with filters",
    args_schema=DBQueryInput
)

agent = initialize_agent(
    tools=[db_tool],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
)
```

**When to use:** Any tool with complex inputs — database tools, REST APIs, file I/O, code execution environments.

---

## 4. Self-Ask with Search Agent

Inspired by the **"Self-Ask"** prompting technique. Instead of going straight to an answer, the agent explicitly asks itself **follow-up questions** and answers each one through search before synthesizing the final answer.

This is ideal for **compositional questions** that require chaining multiple facts.

**The reasoning trace looks like:**

```text
Question: Who was the president of the US when the Eiffel Tower was built?

Are follow up questions needed here: Yes.
Follow up: When was the Eiffel Tower built?
Intermediate answer: The Eiffel Tower was built in 1889.

Follow up: Who was the US president in 1889?
Intermediate answer: Grover Cleveland was president in 1889.

So the final answer is: Grover Cleveland
```

```python
from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun

# Note: Self-Ask requires exactly ONE tool named "Intermediate Answer"
search = DuckDuckGoSearchRun(name="Intermediate Answer")

agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent=AgentType.SELF_ASK_WITH_SEARCH,
    verbose=True
)

agent.run(
    "What is the nationality of the founder of the company "
    "that makes the chip in the iPhone 15?"
)
```

**Constraint:** Requires a tool named `"Intermediate Answer"` — the prompt template is hardcoded to expect it.

**When to use:** Multi-hop factual reasoning; research tasks requiring chained lookups; fact verification.

---

## 5. OpenAI Functions Agent

Uses **OpenAI's native function-calling API** (`functions` parameter) instead of text-based ReAct prompting. The LLM returns structured JSON specifying which function to call and with what arguments — far more reliable than parsing free-text.

This was the first major departure from pure prompt-based tool use.

```python
from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    return f"Weather in {city}: 28°C, partly cloudy"

@tool
def get_population(city: str) -> str:
    """Get the population of a city."""
    return f"Population of {city}: 3.2 million"

llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = initialize_agent(
    tools=[get_weather, get_population],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

agent.run("What's the weather and population of Pune?")

```

**Key advantages over ReAct:** No prompt parsing errors; structured arguments; LLM decides with confidence; much more reliable on complex tools.

**When to use:** Any OpenAI model (`gpt-3.5-turbo`, `gpt-4`, `gpt-4o`); production systems requiring reliability; complex multi-argument tools.

---

## 6. OpenAI Tools Agent

The **successor to OpenAI Functions** — uses the newer `tools` parameter in the OpenAI API. The critical upgrade: it can call **multiple tools in parallel** within a single LLM turn, making it significantly faster for tasks requiring simultaneous lookups.

```python
from langchain.agents import AgentType, initialize_agent

# Same tools as above — just change the agent type
agent = initialize_agent(
    tools=[get_weather, get_population],
    llm=llm,
    agent=AgentType.OPENAI_TOOLS,  # Uses parallel tool calls
    verbose=True
)

# The agent can now call BOTH tools simultaneously in one LLM pass
agent.run("Give me the weather AND population of Mumbai and Delhi")
# → Calls get_weather(Mumbai), get_weather(Delhi),
#   get_population(Mumbai), get_population(Delhi) all at once
```

**Modern LCEL equivalent (recommended):**

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "Weather and population of Pune and Mumbai"})
```

**When to use:** This is the **recommended default** for all OpenAI-compatible models today. Use over `OPENAI_FUNCTIONS` for any task that may benefit from parallelism.

---

## 7. Plan-and-Execute Agent

A fundamentally different architecture. Instead of the reactive `Thought → Action → Observation` loop, this agent **first creates a complete plan**, then executes each step sequentially with an executor agent.

Two LLM calls drive it: a **Planner** (creates the full task list) and an **Executor** (carries out each step).

```python

from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)
from langchain_openai import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun

llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [DuckDuckGoSearchRun(), WikipediaQueryRun()]

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run(
    "Research the top 3 AI chip manufacturers in 2025, "
    "compare their latest GPU specs, and summarize the findings."
)

```

**Planner output (internal):**

```text
Step 1: Search for top AI chip manufacturers in 2025
Step 2: Find latest GPU specs for company 1
Step 3: Find latest GPU specs for company 2
Step 4: Find latest GPU specs for company 3
Step 5: Summarize and compare all findings
```

**When to use:** Long-horizon multi-step research tasks; tasks with clear sequential dependencies; when you want auditability of the plan before execution.

**Trade-off:** Slower (two LLM chains); the plan is fixed upfront and may not adapt well to unexpected tool results mid-way — LangGraph's agent loop handles dynamic replanning better.

---

## Comparison & Selection Guide

## The Modern Recommendation

LangChain's own documentation now points away from `initialize_agent()` with `AgentType` enums and toward the **LCEL + LangGraph** paradigm:

```python
# Modern approach — works with any tool-calling model
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

For anything requiring **state, branching, cycles, or multi-agent coordination**, the recommendation is to move to **LangGraph** directly — where each node is a function, state is explicitly typed, and the graph structure replaces the monolithic `AgentExecutor`. The agent types above are essentially baked into single-node LangGraph graphs, but LangGraph lets you build arbitrarily complex topologies.

Would you like me to go deeper on any specific agent — for example, the ReAct loop internals, how to build a custom tool for the Structured agent, or how these map onto LangGraph nodes?
