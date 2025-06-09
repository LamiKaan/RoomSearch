import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import urllib3
import json

from typing import Type, List, Annotated
from pydantic import BaseModel, Field, field_validator

from langchain_core.tools import BaseTool, InjectedToolArg
from langchain_core.runnables import RunnableConfig

from utils import pretty_print_object

import matplotlib.pyplot as plt
import math

# Schema for the input to the room search tool
class RoomSearchInput(BaseModel):
    config: RunnableConfig = Field(..., description="Configuration dictionary with additional runtime information (which should include an instance of RoomSearch class).")
    similarity_type: Annotated[str, InjectedToolArg] = Field(..., description="Type of similarity to use for room search. It should be one of 'openai', 'keyword', or 'semantic'.")
    user_query: str = Field(..., description="Query string describing the room preferences of the user.")

    @field_validator("similarity_type", mode="after")
    @classmethod
    def validate_similarity_type(cls, value: str) -> str:
        # Validate that the similarity type is one of the allowed values
        if value not in ["openai", "keyword", "semantic"]:
            raise ValueError("The 'similarity_type' must be one of 'openai', 'keyword', or 'semantic'.")
        
        return value
    

class RoomSearchTool(BaseTool):
    name: str = "room_image_search"
    description: str = "Finds the most similar hotel room images based on the user query and the provided similarity type."
    args_schema: Type[BaseModel] = RoomSearchInput
    # response_format: str = "content_and_artifact"

    async def _arun(
        self,
        config,
        similarity_type,
        user_query
    ) -> List[str]:
        """Search for hotel rooms based on the user query and similarity type. Return the list of image URLs of the most similar rooms."""

        # Convert user query to lowercase
        user_query = user_query.lower()

        # Retrieve the RoomSearch instance from the configuration dictionary
        room_search: RoomSearch = config["configurable"]["room_search_object"]

        # Check if the user query is one of cached/default queries
        if user_query in room_search.user_queries:
            try:
                # Retrieve the similarity dictionary for the given user query and similarity type
                similarities = room_search.similarities[user_query][similarity_type]

                # Filter the urls that have a similarity score greater than 0.5
                filtered_similarities = {url: score for url, score in similarities.items() if score > 0.5}

                # If there are such images (with similarity score > 0.5)
                if filtered_similarities:
                    # Sort their urls from highest to lowest similarity score
                    sorted_urls = sorted(filtered_similarities, key=filtered_similarities.get, reverse=True)

                    # Get the URLs of top 6 most similar room images
                    best_urls = sorted_urls[:6]

                    # Get the RGB image corresponding to the best URLs
                    best_images = [room_search.images_rgb[url] for url in best_urls]

                    # DISPLAY THE BEST IMAGES
                    # Get the number of images and determine the grid size (rows and columns) for clean layout
                    num_images = len(best_images)
                    # 2 images per row
                    num_rows = math.ceil(num_images / 2)
                    num_columns = math.ceil(num_images / num_rows)

                    # Create a figure and axes
                    figure, axes = plt.subplots(num_rows, num_columns, figsize=(num_rows * 5, num_columns * 5))
                    # Set the title of the figure
                    figure_title = f"User Query: {user_query}"
                    if len(filtered_similarities) > 6:
                        figure_subtitle = f"Most similar 6 images (best 6 matches) - {similarity_type} similarity"
                    else:
                        figure_subtitle = f"Most similar images - {similarity_type} similarity"
                    figure.suptitle(f"{figure_title}\n{figure_subtitle}", fontsize=16)
                    # Flatten axes for easy iteration if grid is 2D
                    if num_images > 1:
                        axes = axes.flatten()
                    else:
                        axes = [axes]
                    # Loop through the grid/images and display them
                    for i in range(num_rows * num_columns):
                        axis = axes[i]
                        if i < num_images:
                            axis.imshow(best_images[i])
                            image_title = os.path.basename(best_urls[i])
                            axis.set_title(image_title)
                            image_rank = i + 1
                            axis.set_xlabel(image_rank, fontsize=9)
                        else:
                            axis.axis('off')
                        axis.set_xticks([])
                        axis.set_yticks([])

                    # Adjust layout to leave space for the title
                    plt.tight_layout(rect=[0, 0, 1, 0.92])
                    # Show the figure
                    plt.show()

                    success = True
                    # Return the best URLs
                    return success, best_urls
            
            except Exception as e:
                error_message = f"An error occurred while searching for most similar room images:\n{e}"

                success = False
                return success, [error_message]
        
        # If it's a new query
        else:
            # Add the new query to the list of user queries
            room_search.user_queries.append(user_query)

            # Add an empty dictionary for the new query in the similarities dictionary
            room_search.similarities[user_query] = {}

            # If the similarity type is "openai"
            if similarity_type == "openai":

                # Add an empty dictionary of that similarity type to the similarities dictionary for the new query
                room_search.similarities[user_query]["openai"] = {}

                try:
                    # Make a batched call to the image similarity model to get the similarity scores of each image for the new query
                    success, similarities_batch = await room_search.get_similarity_openai_batch(urls=room_search.image_urls, queries=[user_query])

                    if success:
                        # Get the similarity score of each image in the batch for the new query, and store them
                        for url, similarity_score in zip(room_search.image_urls, similarities_batch):
                            
                            # similarity_score is a list of floats with only 1 element (for single query) in that case, so we take it
                            score = similarity_score[0]

                            # If the async gather function returned an exception
                            if isinstance(score, Exception):
                                # Reraise the exception to be handled in the except block
                                raise score
                            
                            # If the score is a valid float value, store it
                            else:
                                room_search.similarities[user_query]["openai"][url] = score
                    
                    # If batch similarity calculation was unsuccessful
                    else:
                        # Again raise an exception with a custom error message
                        raise Exception("Batch similarity calculation was unsuccessful (most probably due to a problem in the api call to the llm). Please try again later.")
                    
                    # If all went well, then repeat the same process for displaying images and returning the best URLs

                    # Retrieve the similarity dictionary for the given user query and similarity type
                    similarities = room_search.similarities[user_query]["openai"]

                    # Filter the urls that have a similarity score greater than 0.5
                    filtered_similarities = {url: score for url, score in similarities.items() if score > 0.5}

                    # If there are such images (with similarity score > 0.5)
                    if filtered_similarities:
                        # Sort their urls from highest to lowest similarity score
                        sorted_urls = sorted(filtered_similarities, key=filtered_similarities.get, reverse=True)

                        # Get the URLs of top 6 most similar room images
                        best_urls = sorted_urls[:6]

                        # Get the RGB image corresponding to the best URLs
                        best_images = [room_search.images_rgb[url] for url in best_urls]

                        # DISPLAY THE BEST IMAGES
                        # Get the number of images and determine the grid size (rows and columns) for clean layout
                        num_images = len(best_images)
                        # 2 images per row
                        num_rows = math.ceil(num_images / 2)
                        num_columns = math.ceil(num_images / num_rows)

                        # Create a figure and axes
                        figure, axes = plt.subplots(num_rows, num_columns, figsize=(num_rows * 5, num_columns * 5))
                        # Set the title of the figure
                        figure_title = f"User Query: {user_query}"
                        if len(filtered_similarities) > 6:
                            figure_subtitle = f"Most similar 6 images (best 6 matches) - {similarity_type} similarity"
                        else:
                            figure_subtitle = f"Most similar images - {similarity_type} similarity"
                        figure.suptitle(f"{figure_title}\n{figure_subtitle}", fontsize=16)
                        # Flatten axes for easy iteration if grid is 2D
                        if num_images > 1:
                            axes = axes.flatten()
                        else:
                            axes = [axes]
                        # Loop through the grid/images and display them
                        for i in range(num_rows * num_columns):
                            axis = axes[i]
                            if i < num_images:
                                axis.imshow(best_images[i])
                                image_title = os.path.basename(best_urls[i])
                                axis.set_title(image_title)
                                image_rank = i + 1
                                axis.set_xlabel(image_rank, fontsize=9)
                            else:
                                axis.axis('off')
                            axis.set_xticks([])
                            axis.set_yticks([])

                        # Adjust layout to leave space for the title
                        plt.tight_layout(rect=[0, 0, 1, 0.92])
                        # Show the figure
                        plt.show()

                        success = True
                        # Return the best URLs
                        return success, best_urls

                except Exception as e:
                    error_message = f"An error occurred while searching for most similar room images (in batch):\n{e}"

                    # Revert similarities dictionary to its previous state (remove the new query)
                    room_search.similarities.pop(user_query, None)
                    room_search.user_queries.remove(user_query)

                    success = False
                    return success, [error_message]

            # If the similarity type is "keyword"
            elif similarity_type == "keyword":
                # Add an empty dictionary of that similarity type to the similarities dictionary for the new query
                room_search.similarities[user_query]["keyword"] = {}

                try:
                    # Get the similarity scores of each image for the new query using the keyword similarity method
                    success, similarities_keyword = room_search.get_similarities_keyword(room_search.image_urls, [user_query], room_search.descriptions)

                    if success:
                        # Get the similarity score of each image for the new query, and store them
                        for url, similarity_score in zip(room_search.image_urls, similarities_keyword):
                            
                            # similarity_score is a list of floats with only 1 element (for single query) in that case, so we take it
                            score = similarity_score[0]

                            # And store it
                            room_search.similarities[user_query]["keyword"][url] = score
                    
                    # If similarity calculation was unsuccessful
                    else:
                        # Again raise an exception with a custom error message
                        raise Exception("Keyword-based similarity calculation was unsuccessful. Please try again later.")
                    
                    # If all went well, then repeat the same process for displaying images and returning the best URLs

                    # Retrieve the similarity dictionary for the given user query and similarity type
                    similarities = room_search.similarities[user_query]["keyword"]

                    # Filter the urls that have a similarity score greater than 0.5
                    filtered_similarities = {url: score for url, score in similarities.items() if score > 0.5}

                    # If there are such images (with similarity score > 0.5)
                    if filtered_similarities:
                        # Sort their urls from highest to lowest similarity score
                        sorted_urls = sorted(filtered_similarities, key=filtered_similarities.get, reverse=True)

                        # Get the URLs of top 6 most similar room images
                        best_urls = sorted_urls[:6]

                        # Get the RGB image corresponding to the best URLs
                        best_images = [room_search.images_rgb[url] for url in best_urls]

                        # DISPLAY THE BEST IMAGES
                        # Get the number of images and determine the grid size (rows and columns) for clean layout
                        num_images = len(best_images)
                        # 2 images per row
                        num_rows = math.ceil(num_images / 2)
                        num_columns = math.ceil(num_images / num_rows)

                        # Create a figure and axes
                        figure, axes = plt.subplots(num_rows, num_columns, figsize=(num_rows * 5, num_columns * 5))
                        # Set the title of the figure
                        figure_title = f"User Query: {user_query}"
                        if len(filtered_similarities) > 6:
                            figure_subtitle = f"Most similar 6 images (best 6 matches) - {similarity_type} similarity"
                        else:
                            figure_subtitle = f"Most similar images - {similarity_type} similarity"
                        figure.suptitle(f"{figure_title}\n{figure_subtitle}", fontsize=16)
                        # Flatten axes for easy iteration if grid is 2D
                        if num_images > 1:
                            axes = axes.flatten()
                        else:
                            axes = [axes]
                        # Loop through the grid/images and display them
                        for i in range(num_rows * num_columns):
                            axis = axes[i]
                            if i < num_images:
                                axis.imshow(best_images[i])
                                image_title = os.path.basename(best_urls[i])
                                axis.set_title(image_title)
                                image_rank = i + 1
                                axis.set_xlabel(image_rank, fontsize=9)
                            else:
                                axis.axis('off')
                            axis.set_xticks([])
                            axis.set_yticks([])

                        # Adjust layout to leave space for the title
                        plt.tight_layout(rect=[0, 0, 1, 0.92])
                        # Show the figure
                        plt.show()

                        success = True
                        # Return the best URLs
                        return success, best_urls

                except Exception as e:
                    error_message = f"An error occurred while searching for most similar room images (keyword-based):\n{e}"

                    # Revert similarities dictionary to its previous state (remove the new query)
                    room_search.similarities.pop(user_query, None)
                    room_search.user_queries.remove(user_query)

                    success = False
                    return success, [error_message]

            # If the similarity type is "semantic"
            elif similarity_type == "semantic":
                # Add an empty dictionary of that similarity type to the similarities dictionary for the new query
                room_search.similarities[user_query]["semantic"] = {}

                try:
                    # Get the similarity scores of each image for the new query using the semantic similarity method
                    success, similarities_semantic = room_search.get_similarities_semantic(room_search.image_urls, [user_query], room_search.descriptions)

                    if success:
                        # Get the similarity score of each image for the new query, and store them
                        for url, similarity_score in zip(room_search.image_urls, similarities_semantic):
                            
                            # similarity_score is a list of floats with only 1 element (for single query) in that case, so we take it
                            score = similarity_score[0]

                            # And store it
                            room_search.similarities[user_query]["semantic"][url] = score
                    
                    # If similarity calculation was unsuccessful
                    else:
                        # Again raise an exception with a custom error message
                        raise Exception("Semantic-based similarity calculation was unsuccessful. Please try again later.")
                    
                    # If all went well, then repeat the same process for displaying images and returning the best URLs

                    # Retrieve the similarity dictionary for the given user query and similarity type
                    similarities = room_search.similarities[user_query]["semantic"]

                    # Filter the urls that have a similarity score greater than 0.5
                    filtered_similarities = {url: score for url, score in similarities.items() if score > 0.5}

                    # If there are such images (with similarity score > 0.5)
                    if filtered_similarities:
                        # Sort their urls from highest to lowest similarity score
                        sorted_urls = sorted(filtered_similarities, key=filtered_similarities.get, reverse=True)

                        # Get the URLs of top 6 most similar room images
                        best_urls = sorted_urls[:6]

                        # Get the RGB image corresponding to the best URLs
                        best_images = [room_search.images_rgb[url] for url in best_urls]

                        # DISPLAY THE BEST IMAGES
                        # Get the number of images and determine the grid size (rows and columns) for clean layout
                        num_images = len(best_images)
                        # 2 images per row
                        num_rows = math.ceil(num_images / 2)
                        num_columns = math.ceil(num_images / num_rows)

                        # Create a figure and axes
                        figure, axes = plt.subplots(num_rows, num_columns, figsize=(num_rows * 5, num_columns * 5))
                        # Set the title of the figure
                        figure_title = f"User Query: {user_query}"
                        if len(filtered_similarities) > 6:
                            figure_subtitle = f"Most similar 6 images (best 6 matches) - {similarity_type} similarity"
                        else:
                            figure_subtitle = f"Most similar images - {similarity_type} similarity"
                        figure.suptitle(f"{figure_title}\n{figure_subtitle}", fontsize=16)
                        # Flatten axes for easy iteration if grid is 2D
                        if num_images > 1:
                            axes = axes.flatten()
                        else:
                            axes = [axes]
                        # Loop through the grid/images and display them
                        for i in range(num_rows * num_columns):
                            axis = axes[i]
                            if i < num_images:
                                axis.imshow(best_images[i])
                                image_title = os.path.basename(best_urls[i])
                                axis.set_title(image_title)
                                image_rank = i + 1
                                axis.set_xlabel(image_rank, fontsize=9)
                            else:
                                axis.axis('off')
                            axis.set_xticks([])
                            axis.set_yticks([])

                        # Adjust layout to leave space for the title
                        plt.tight_layout(rect=[0, 0, 1, 0.92])
                        # Show the figure
                        plt.show()

                        success = True
                        # Return the best URLs
                        return success, best_urls

                except Exception as e:
                    error_message = f"An error occurred while searching for most similar room images (semantic-based):\n{e}"

                    # Revert similarities dictionary to its previous state (remove the new query)
                    room_search.similarities.pop(user_query, None)
                    room_search.user_queries.remove(user_query)

                    success = False
                    return success, [error_message]


    def _run(
        self,
        config,
        similarity_type,
        user_query
    ):
        # Call the async version from the sync method
        return asyncio.run(self._arun(config, similarity_type, user_query))


if __name__ == "__main__":

    async def main():
    
        from RoomSearch import RoomSearch
        
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Get image urls and user queries from parameter file
        with open("parameters.json", "r") as f:
            parameters = json.load(f)
        image_urls = parameters["IMAGE_URLS"]
        user_queries = parameters["USER_QUERIES"]

        room_search = await RoomSearch.create_instance_async(image_urls, user_queries)
        
        room_search_tool = RoomSearchTool()

        config = config = {
            "configurable": {
                "thread_id": 1,
                "room_search_object": room_search
            }
        }

        # Example 1: Search for rooms with an existing user query
        similarity_type = "openai"
        user_query = user_queries[1]

        success, urls = asyncio.run(room_search_tool.ainvoke({
        "config": config,
        "similarity_type": similarity_type,
        "user_query": user_query
        }))
        
        print(success)
        print(urls)

        # Example 2: Search for rooms with a new user query
        user_query = "Rooms with modern interior design, a balcony with sea view, and exceptional cleanliness"
        
        success, urls = asyncio.run(room_search_tool.ainvoke({
        "config": config,
        "similarity_type": similarity_type,
        "user_query": user_query
        }))
        
        print(type(success))
        print(urls)

    asyncio.run(main())