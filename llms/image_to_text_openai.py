import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import asyncio
import urllib3
from dotenv import load_dotenv
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load api key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize gpt-4o-mini model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    openai_api_key=openai_api_key
    )

system_message = """You are a helpful assistant whose task is to generate clear and informative textual descriptions of hotel room images. These descriptions will be used in a hotel room search system that matches user queries to images of hotel rooms, either via keyword-based search or semantic (vector-based) similarity.

Your objective is to describe each hotel room image in a way that:
- Accurately reflects visible room features
- Mentions relevant amenities, room layout, and view
- Includes information that would help match the image to a user’s room preferences

Context:
- All images will be of hotel rooms.
- User queries may include specific requests such as:
    + Room type: single, double, triple
    + Features: balcony, sea view, city view, air conditioning, desk
    + Capacity: maximum number of people the room can accommodate
- Descriptions will be compared with user queries to find matching images.
- Below are some examples of possible user queries:
    + Double rooms with a sea view
    + Rooms with a balcony and air conditioning, with a city view
    + Triple rooms with a desk
    + Rooms with a maximum capacity of 4 people
    
Instructions:
When describing an image:
1.	Focus only on what is clearly visible in the image.
2.	Use clear, concise, and natural language.
3.	Include details such as:
    + Number and type of beds (or room capacity)
    + View type (e.g., sea view, city view)
    + Visible amenities (e.g., air conditioner, desk, TV, balcony, closet)
    + Layout and spaciousness
    + Seating areas (e.g., chairs, sofas)
    + Bathroom visibility (if any)
4.	Avoid making assumptions about features not visible in the image.
5.	Do not refer to the image itself (e.g., don’t say “This image shows…”).

Output Format:
Generate a 1–3 sentence description for each image. Example outputs:
- “A double room with a large window offering a sea view, two single beds, and a wall-mounted TV. A small desk and an armchair are placed near the balcony.”
- “Spacious triple room with three single beds, city view, air conditioning unit, and a wardrobe.”"""

prompt = ChatPromptTemplate(
    [
        {
            "role": "system",
            "content": system_message,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Please give a text description for the provided hotel room image:",
                },
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": "image/jpeg",
                    "data": "{image_base64}",
                },
            ],
        },
    ]
)

image_to_text_llm = prompt | llm




if __name__ == "__main__":

    async def main():
        from RoomSearch import RoomSearch

        # load_dotenv()
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Get image urls from parameter file
        with open("parameters.json", "r") as f:
            parameters = json.load(f)
        image_urls = parameters["IMAGE_URLS"]
        user_queries = parameters["USER_QUERIES"]

        room_search = await RoomSearch.create_instance_async(image_urls, user_queries)

        image_base64 = room_search.images_base64[image_urls[0]]

        prompt_value = prompt.invoke({"image_base64": image_base64})

        response = image_to_text_llm.invoke({"image_base64": image_base64})

        print(type(response))
        print()
        response.pretty_print()
        print()

    asyncio.run(main())
