import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    API_KEY = os.environ.get('GOOGLE_API_KEY')

    template = """
            Answer the question based on the context below. If the 
            question cannot be answered using the information provided answer
            with "I don't know".

            Context: Renaissance Art was a cultural movement that began in Europe
            during the 14th century and emphasized realism, human emotion, and
            perspective in paintings and sculptures. Famous artists such as Leonardo
            da Vinci, Michelangelo, and Raphael created masterpieces that continue
            to influence modern art and design.

            Question: {query}

            Answer:"""

    prompt = PromptTemplate(
        input_variables=['query'],
        template=template
    )

    model = ChatGoogleGenerativeAI(
        model='gemini-2.5-flash-lite',
        api_key= API_KEY
    )

    print(prompt.format(query="Who were the famous artists of the Renaissance period?"))

    response = model.invoke(prompt.format(query="Who were the famous artists of the Renaissance period?"))

    print(response.content)


if __name__ == "__main__":
    main()
