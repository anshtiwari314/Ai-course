import asyncio
import os 
from dotenv import load_dotenv
from openai import AsyncOpenAI 

load_dotenv()

class ChatServices():
    def __init__(self):
        api_key = os.environ["OPENROUTER_API_KEY"]

        #print(api_key)

        self.client = AsyncOpenAI(
            api_key = api_key,
            
            base_url = "https://openrouter.ai/api/v1"
        )

    async def ask(self,question:str)->str :
        response = await self.client.chat.completions.create(
            model = os.environ["OPENROUTER_MODEL"],
            messages = [
                {
                    "role":"user",
                    "content":question
                }
            ]
        ) 
        return response.choices[0].message.content 

async def main():
    chatservice = ChatServices()
    query = input("query: ")
    result = await chatservice.ask(query)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

