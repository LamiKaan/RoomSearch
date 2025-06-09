import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import asyncio
import urllib3
from dotenv import load_dotenv
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from typing import List
from typing_extensions import Annotated, TypedDict
from pydantic import BaseModel, Field, field_validator

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

class SimilarityOutputOld(TypedDict):
    """Result of checking the similarities between user-provided queries and a hotel room image."""

    similarities: Annotated[List[float], ..., "List of similarity scores for each user query, where each score is a float between 0 and 1."]

class SimilarityOutput(BaseModel):
    """Result of checking the similarities between user-provided queries and a hotel room image."""

    similarities: List[float] = Field(..., description="List of similarity scores for each user query, where each score is a float between 0 and 1.")

    @field_validator("similarities", mode="after")
    @classmethod
    def validate_similarities(cls, values: List[float]) -> List[float]:
        # Validate that the similarities list is not empty
        if len(values) == 0:
            raise ValueError("The 'similarities' list cannot be empty.")
        
        # Validate that all scores are floats between 0.0 and 1.0
        if not all(isinstance(score, float) and (0.0 <= score <= 1.0) for score in values):
            raise ValueError("All similarity scores must be floats between 0.0 and 1.0.")
        
        return values
    
structured_llm = llm.with_structured_output(SimilarityOutput)

system_message = """You are a helpful assistant tasked with evaluating how well a given hotel room image satisfies a set of user queries related to room preferences. Your goal is to compute a similarity score between the content of the image and each query.

Scenario Context:
- You will be given:
    + A single image of a hotel room
    + One or more textual user queries describing room preferences
- You must analyze the image and compare it with each query independently
- Return a numerical score between 0 and 1 for each query:
	+ 1.0 → The room in the image fully satisfies the query
	+ 0.0 → The room clearly does not satisfy the query at all
    
Instructions:
1. Interpret the user query: Identify features requested in each query (e.g., room type, view, amenities).
2. Analyze the image content: Focus only on visible elements in the room (do not assume hidden features).
3. Evaluate similarity:
	+ Check how well the visible features in the image match the requested features in the query.
	+ Consider partial matches (e.g., query asks for balcony and AC, and only AC is present → score accordingly).
4. Assign a floating point score between [0, 1] to each query.
5. Do not return explanations—only return a list of scores in the same order as the queries.

Output Format:
Return a Python-style list (as the only value of a dict with key 'similarities') of floating-point similarity scores in the same order as the input queries.

# Example 1
Example Input 1:
- Image: (assume a base64-encoded image string)
- Queries: ['Double rooms with sea view', 'Rooms with balcony and air conditioning', 'Triple rooms with desk', 'Rooms with maximum capacity of 4 people']
Example Output 1:
- {{ "similarities": [0.45, 0.67, 0.32, 0.41] }}

# Example 2
Example Input 2:
- Image: (assume a base64-encoded image string)
- Queries: ['Double rooms with a forest view', 'Rooms with a TV and air conditioning, with a nature view']
Example Output 2:
- {{ "similarities": [0.84, 0.91] }}

# Example 3
Example Input 3:
- Image: (assume a base64-encoded image string)
- Queries: ['Room with a double bed and a balcony']
Example Output 3:
- {{ "similarities": [0.50] }}"""

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
                    "text": "Please give the similarity score list for the user query/queries: {queries} and the provided hotel room image:",
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

image_to_similarity_llm = prompt | structured_llm


if __name__ == "__main__":

    async def main():
        from RoomSearch import RoomSearch

        # load_dotenv()
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Get image urls and user queries from parameter file
        with open("parameters.json", "r") as f:
            parameters = json.load(f)
        image_urls = parameters["IMAGE_URLS"]
        user_queries = parameters["USER_QUERIES"]

        room_search = await RoomSearch.create_instance_async(image_urls, user_queries)

        image_base64 = room_search.images_base64[image_urls[0]]

        response = image_to_similarity_llm.invoke({"queries": repr(user_queries), "image_base64": image_base64})

        print(type(response))
        print(response)
        print()
        print(type(response["similarities"]))
        print(response["similarities"])

    asyncio.run(main())