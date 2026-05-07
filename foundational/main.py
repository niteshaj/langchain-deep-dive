import os
from urllib import response
from langchain_google_genai import ChatGoogleGenerativeAI


def main():
    API_KEY = os.environ.get('GOOGLE_API_KEY')
    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', api_key=API_KEY)
    response = model.invoke("What's the capital of the Moon?")
    print(response.content)

if __name__ == "__main__":
    main()
