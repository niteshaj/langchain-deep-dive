# Prompt

## Creating a PromptTemplate Object

There are two ways to create a PromptTemplate object.

1. Using the from_template() method
2. Creating a PromptTemplate object and a prompt all at once.

### Using the from_template() method

```python
from langchain_core.prompts import PromptTemplate

template = 'what is the capital of {country}?'

prompt = PromptTemplate.from_template(template)

chain = prompt | llm

chain.invoke('India').content
```

### Creating a PromptTemplate object and a prompt all at once

```python
template = 'what is the capital of {country}?'

prompt = PromptTemplate(
    template=template,
    input_variables=['country'],
)

chain = prompt | llm

chain.invoke("India").content
# Output: 'The capital of India is New Delhi.'
```

### partial_variables

Using partial_variables , you can partially apply functions. This is particularly useful when there are common variables to be shared.

Suppose you want to specify the current date in your prompt, hardcoding the date into the prompt or passing it along with other input variables may not be practical. In this case, using a function that returns the current date to modify the prompt partially is much more convenient.

```python
    def get_today():
        return datetime.now().strftime("%B %d")

    prompt = PromptTemplate(
        template="Today's date is {today}. Please list {n} celebrities whose birthday is today. Please specify their date of birth.",
        input_variables=["n"],
        partial_variables={
            today=get_today
        }
    )
```

```python
template = 'What are the capitals of {country1} and {country2}, respectively?'

prompt = PromptTemplate(
    template=template,
    input_variables=['country1'],
    partial_variables={
        "country2": "United States of America"  # Pass `partial_variables` in dictionary form
    }
)

prompt.format(country1="South Korea")
#Output:'What are the capitals of South Korea and United States of America, respectively?'

# we can override the partial variable by call partial method in prompt
prompt_partial = prompt.partial(country2="India")
prompt_partial.format(country1="South Korea")
# Output: 'What are the capitals of South Korea and India, respectively?'

chain = prompt_partial | llm

chain.invoke("United States of America").content
# Output: 'The capital of the United States of America is Washington, D.C., and the capital of India is New Delhi.'

chain.invoke({"country1": "United States of America", "country2": "India"}).content
# Output: 'The capital of the United States of America is Washington, D.C., and the capital of India is New Delhi.'

```

### Load Prompt Templates from YAML Files

You can manage prompt templates in separate yaml files and load using load_prompt.

```python
from langchain_core.prompts import load_prompt

prompt = load_prompt("prompts/fruit_color.yaml", encoding="utf-8")
prompt.format(fruit="an apple")

# Output: 'What is the color of an apple?'
```

### ChatPromptTemplate

ChatPromptTemplate can be used to include a conversation history as a prompt. Messages are structured as tuples in the format (role , message ) and are created as a list.

role

- system : A system setup message, typically used for global settings-related prompts.
- human : A user input message.
- ai : An AI response message.

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        # role, message
        ("system", "You are a friendly AI assistant. Your name is {name}."),
        ("human", "Nice to meet you!"),
        ("ai", "Hello! How can I assist you?"),
        ("human", "{user_input}"),
    ]
)
chain = chat_template | llm

chain.invoke({"name": "Teddy", "user_input": "What is your name?"}).content
# Output: 'My name is Teddy. How can I help you today?'
```

### MessagePlaceholder

LangChain also provides a MessagePlaceholder , which provides complete control over rendering messages during formatting.

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a summarization specialist AI assistant. Your mission is to summarize conversations using key points.",
        ),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "Summarize the conversation so far in {word_count} words."),
    ]
)
chain = chat_prompt | llm | StrOutputParser()

chain.invoke(
    {
        "word_count": 5,
        "conversation": [
            (
                "human",
                "Hello! I'm Teddy. Nice to meet you.",
            ),
            ("ai", "Nice to meet you! I look forward to working with you."),
        ],
    }
)

# Output: 'Teddy introduces himself, exchanges greetings.'

```

---

## Few-Shot Templates

LangChain's few-shot prompting provides a robust framework for guiding language models to generate high-quality outputs by supplying carefully selected examples. This technique minimizes the need for extensive model fine-tuning while ensuring precise, context-aware results across diverse use cases.

### FewShotPromptTemplate

