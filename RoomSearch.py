import os
import urllib3
import base64
import time
import pickle
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import cv2

import asyncio
import requests
import backoff
from tqdm import tqdm
import aioconsole

from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from llms.image_to_text_openai import image_to_text_llm
from llms.image_to_similarity_openai import image_to_similarity_llm
from assistant.room_search_graph import room_search_graph
from assistant.room_search_agent import room_search_system_prompt
from utils import pretty_print_object

class RoomSearch:
    def __init__(self, image_urls, user_queries):
        # List of image addresses (URLs) and user queries
        self.image_urls = image_urls
        self.user_queries = [query.lower() for query in user_queries]
        # Dict of downloaded images in RGB format
        self.images_rgb = {}
        self.images_base64 = {}
        # Dict of text descriptions of images
        self.descriptions = {}
        # Dict of similarities of images to user queries
        self.similarities = {}
        # Cache the sentence transformer model for semantic similarity to avoid reloading it multiple times
        self.sentence_transformer = None
        # Initialize the room search by downloading images, generating descriptions, and calculating similarities for default user queries
        # self.initialization_task()

    @classmethod
    async def create_instance_async(cls, image_urls, user_queries):
        # Create an instance of RoomSearch
        self = cls(image_urls, user_queries)
        # Run the initialization task asynchronously
        await self.initialization_task()
        # Return the instance
        return self
    
    def download_and_save_image(self, url):
        try:
            # Get the full path to the image file
            file_name = os.path.basename(url)
            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/images", file_name)
            
            # If the file does not already exist, download and save it
            if not os.path.exists(file_path):
                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Download the image
                response = requests.get(url, timeout=5, verify=False)
                # Check if the request was successful
                response.raise_for_status()
                
                # Save the image content to a file
                if response.content:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)

                    # Also, store image in RGB format             
                    image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
                    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    self.images_rgb[url] = image_rgb
                    
                    # And also store it in base64 format
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                    self.images_base64[url] = image_base64

                    return True
                
            # If the file exists but RGB or base64 version is not stored
            elif ( (self.images_rgb.get(url, None) is None)  or  (self.images_base64.get(url, None) is None) ):

                if self.images_rgb.get(url, None) is None:
                    # Read image from file, convert to RGB and store it
                    image_rgb = cv2.imread(file_path, cv2.IMREAD_COLOR_RGB)
                    self.images_rgb[url] = image_rgb

                if self.images_base64.get(url, None) is None:
                    # Read the image file and convert it to base64
                    with open(file_path, 'rb') as f:
                        image_bytes = f.read()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        self.images_base64[url] = image_base64

                return True

            # If both the file, RGB and base64 versions exist, return True
            else:
                return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")

            return False
        
    def convert_image_to_text(self, url):
        try:
            # Get the full path to the description file
            file_name = os.path.basename(url).replace('.jpg', '.txt')
            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/descriptions", file_name)
            
            # If the file does not already exist, create and save it
            if not os.path.exists(file_path):
                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # Get the image in base64 format
                image_base64 = self.images_base64[url]
                # Invoke the image-to-text LLM
                response = image_to_text_llm.invoke({"image_base64": image_base64})
                description = response.content

                self.descriptions[url] = description

                # Save the description to a file
                with open(file_path, 'w') as f:
                    f.write(description)

                return True
                
            # If the file exists, but the description is not stored
            elif ( self.descriptions.get(url, None) is None ):

                # Save the description
                with open(file_path, 'r') as f:
                    description = f.read()
                    self.descriptions[url] = description

                return True

            # If both the file exists and the description is stored, return True
            else:
                return True
            
        except Exception as e:
            print(f"Error while creating/saving description for the image: {url}\n{e}")

            return False
    
    async def convert_image_to_text_async(self, url, semaphore):
        # While the semaphore is available
        async with semaphore:
            try:
                # Get the full path to the description file
                file_name = os.path.basename(url).replace('.jpg', '.txt')
                file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/descriptions", file_name)

                # Define an async function to invoke the image-to-text LLM with backoff (to handle rate limits)
                @backoff.on_exception(backoff.constant, Exception, interval=5, max_tries=3, jitter=None)
                async def call_llm(image_base64):
                    # Invoke the image-to-similarity LLM asynchronously
                    response = await image_to_text_llm.ainvoke({"image_base64": image_base64})
                    return response
                
                # If the file does not already exist, create and save it
                if not os.path.exists(file_path):
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    # Get the image in base64 format
                    image_base64 = self.images_base64[url]
                    # Invoke the image-to-text LLM
                    response = await call_llm(image_base64)
                    description = response.content

                    self.descriptions[url] = description

                    # Save the description to a file
                    with open(file_path, 'w') as f:
                        f.write(description)

                    return True
                    
                # If the file exists, but the description is not stored
                elif ( self.descriptions.get(url, None) is None ):

                    # Save the description
                    with open(file_path, 'r') as f:
                        description = f.read()
                        self.descriptions[url] = description

                    return True

                # If both the file exists and the description is stored, return True
                else:
                    return True
                
            except Exception as e:
                print(f"Error while creating/saving description for the image: {url}\n{e}")

                return False
    
    async def convert_images_to_text_batch(self, urls, max_concurrency=1):
        # Create a semaphore to limit the number of concurrent requests
        semaphore = asyncio.Semaphore(max_concurrency)

        # Create the result list to hold calculated similarity score for each image/url
        results = [None] * len(urls)

        # Use tqdm to show a progress bar for the async tasks
        with tqdm(total=len(urls), desc="Converting images to text", unit="image", leave=False) as pbar:

            # Define an async function to get description for a single image and update the progress bar
            async def run_and_track(i, url):
                # Get the description for the url/image
                result = await self.convert_image_to_text_async(url, semaphore)
                results[i] = result
                # Update the progress bar for each completed task
                pbar.update(1)

            # Create a list of tasks to run for all URLs
            tasks = [run_and_track(i, url) for i, url in enumerate(urls)]

            # Run the tasks concurrently and wait for all of them to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        # Return (success, results)
        return all(r is not None and not isinstance(r, Exception) for r in results), results
    
    def get_similarity_openai(self, url, queries):
        try:
            # Get the full path to the similarity file
            file_name = os.path.basename(url).replace('.jpg', '.pkl')
            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_openai", file_name)
            
            # If the file does not already exist, create and save it
            if not os.path.exists(file_path):
                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # Get the image in base64 format
                image_base64 = self.images_base64[url]
                # Invoke the image-to-similarity LLM
                response = image_to_similarity_llm.invoke({"queries": repr(queries), "image_base64": image_base64})
                similarities = response.similarities

                # Save the similarities to a file
                with open(file_path, 'wb') as f:
                    pickle.dump(similarities, f)

                # Also, store the similarities in the dict
                for i, query in enumerate(queries):
                    
                    # If the query is not in the dict, add it
                    if ( self.similarities.get(query, None) is None ):
                        self.similarities[query] = {}

                    # If the openai key is not in the dict for query, add it
                    if ( self.similarities[query].get("openai", None) is None ):
                        self.similarities[query]["openai"] = {}

                    # Store the similarity for the query and url
                    self.similarities[query]["openai"][url] = similarities[i]

                return True
                
            # If the file exists
            else:
                # Get the similarities from the file
                with open(file_path, 'rb') as f:
                    similarities = pickle.load(f)

                for i, query in enumerate(queries):

                    # If the query is not in the dict, add it
                    if ( self.similarities.get(query, None) is None ):
                        self.similarities[query] = {}

                    # If the openai key is not in the dict for query, add it
                    if ( self.similarities[query].get("openai", None) is None ):
                        self.similarities[query]["openai"] = {}

                    # Store the similarity for the query and url
                    self.similarities[query]["openai"][url] = similarities[i]

                return True
            
        except Exception as e:
            print(f"Error while getting openai similarity for the image: {url}\n{e}")
            
            return False
    
    async def get_similarity_openai_async(self, url, queries, semaphore):
        # While the semaphore is available
        async with semaphore:
            
            # Get the base64 image representation of the url
            image_base64 = self.images_base64[url]

            # Define an async function to invoke the image-to-similarity LLM with backoff (to handle rate limits)
            @backoff.on_exception(backoff.constant, Exception, interval=5, max_tries=3, jitter=None)
            async def call_llm():
                # Invoke the image-to-similarity LLM asynchronously
                response = await image_to_similarity_llm.ainvoke({"queries": repr(queries), "image_base64": image_base64})
                return response
            
            try:
                # Call the LLM with backoff
                response = await call_llm()

                # Get the similarities from the response
                similarities = response.similarities

                # Return the similarities for the single call given url
                return similarities
            
            except Exception as e:
                print(f"Error while getting (async) OpenAI similarity for the image: {url}\n{e}")
                # Return None if there was an error
                return None
    
    async def get_similarity_openai_batch(self, urls, queries, max_concurrency=1):
        # Create a semaphore to limit the number of concurrent requests
        semaphore = asyncio.Semaphore(max_concurrency)

        # Create the result list to hold calculated similarity score for each image/url
        results = [None] * len(urls)

        # Use tqdm to show a progress bar for the async tasks
        with tqdm(total=len(urls), desc="Calculating similarities", unit="image", leave=False) as pbar:

            # Define an async function to calculate similarity for a single image and update the progress bar
            async def run_and_track(i, url):
                # Get the similarity for the url
                result = await self.get_similarity_openai_async(url, queries, semaphore)
                results[i] = result
                # Update the progress bar for each completed task
                pbar.update(1)

            # Create a list of tasks to run for all URLs
            tasks = [run_and_track(i, url) for i, url in enumerate(urls)]

            # Run the tasks concurrently and wait for all of them to complete
            await asyncio.gather(*tasks, return_exceptions=True)

        # print("--------------------------------------------------------------------------------")
        # print("RESULTS OF BATCH OPENAI SIMILARITY CALCULATION:")
        # print(results)
        # print("RESULTS OF BATCH OPENAI SIMILARITY CALCULATION-2:")
        # print(repr(results))
        # print("--------------------------------------------------------------------------------")

        # Return (success, results)
        return all(r is not None and not isinstance(r, Exception) for r in results), results
    
    def get_similarities_keyword(self, urls, queries, descriptions, normalize=True):
        try:
            assert isinstance(urls, list), "URLs should be a list of strings."
            assert isinstance(queries, list), "Queries should be a list of strings."
            assert isinstance(descriptions, dict), "Descriptions should be a dictionary with URLs as keys and descriptions as values."
            
            # Create list of descriptons for the URLs
            descriptions_list = [descriptions[url] for url in urls]

            # All text to be used for similarity calculations
            texts = queries + descriptions_list

            # Create a TF-IDF vectorizer
            vectorizer = TfidfVectorizer()
            # Fit and transform the texts to get TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform(texts)
            # Extract the TF-IDF vectors for queries and descriptions
            query_vectors = tfidf_matrix[:len(queries)]
            description_vectors = tfidf_matrix[len(queries):]
            # Calculate cosine similarities between each query and description
            similarities_keyword = cosine_similarity(query_vectors, description_vectors)
            
            if normalize:
                # Apply row-wise (per query) min-max scaling to the similarity matrix such that the largest score is 1.0 and the lowest is 0.0
                min_values = similarities_keyword.min(axis=1, keepdims=True)
                max_values = similarities_keyword.max(axis=1, keepdims=True)
                similarities_keyword_scaled = (similarities_keyword - min_values) / (max_values - min_values)
            else:
                similarities_keyword_scaled = similarities_keyword
            
            return (True, similarities_keyword_scaled)
        except Exception as e:
            print(f"Error while calculating keyword similarities: {e}")
            return (False, None)
        
    def get_similarities_semantic(self, urls, queries, descriptions, show_progress=True, normalize=True):
        try:

            assert isinstance(urls, list), "URLs should be a list of strings."
            assert isinstance(queries, list), "Queries should be a list of strings."
            assert isinstance(descriptions, dict), "Descriptions should be a dictionary with URLs as keys and descriptions as values."

            if show_progress:
                # Show a progress bar for the semantic similarity calculation
                pbar = tqdm(total=3, desc="Calculating semantic similarities", unit="step", leave=False)
                pbar.set_description("Loading SentenceTransformer model")

            # Create a SentenceTransformer model for semantic similarity
            if self.sentence_transformer is None:
                # Load the model for the first time
                self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            # Use the cached model
            model = self.sentence_transformer

            if show_progress:
                pbar.update(1)
                pbar.set_description("Calculating embeddings for queries")
            
            # Create list of descriptons for the URLs
            descriptions_list = [descriptions[url] for url in urls]

            # Get embeddings for queries and descriptions
            query_embeddings = model.encode(queries, convert_to_tensor=True)

            if show_progress:
                pbar.update(1)
                pbar.set_description("Calculating embeddings for image descriptions")

            description_embeddings = model.encode(descriptions_list, convert_to_tensor=True)

            # Calculate cosine similarities between each query and description
            similarity_matrix = util.cos_sim(query_embeddings, description_embeddings)
            # Convert to numpy array
            similarities_semantic = similarity_matrix.cpu().numpy()

            if normalize:
                # Apply row-wise (per query) min-max scaling to the similarity matrix such that the largest score is 1.0 and the lowest is 0.0
                min_values = similarities_semantic.min(axis=1, keepdims=True)
                max_values = similarities_semantic.max(axis=1, keepdims=True)
                similarities_semantic_scaled = (similarities_semantic - min_values) / (max_values - min_values)
            else:
                similarities_semantic_scaled = similarities_semantic

            if show_progress:
                pbar.update(1)
            
            return (True, similarities_semantic_scaled)
        
        except Exception as e:
            print(f"Error while calculating semantic similarities: {e}")
            return (False, None)
    
    async def initialization_task(self):

        with tqdm(total=5, desc="Initializing RoomSearch", unit="step") as pbar:
        
            # GET IMAGES
            pbar.set_description("Step 1/5: Downloading images")
            time.sleep(1)
            # For each image
            for url in self.image_urls:
                # Download and save images from the URLs
                success = self.download_and_save_image(url)
                # If the download wasn't successful, try once more after a short delay
                if not success:
                    time.sleep(2)
                    success = self.download_and_save_image(url)
                    # If it still fails
                    if not success:
                        raise ValueError(f"Failed to download image from '{url}' after two attempts.")
                    
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            
            pbar.update(1)      
            # GET DESCRIPTIONS
            pbar.set_description("Step 2/5: Generating descriptions")
            time.sleep(1)
            # For all images at once (batch processing)
            success, results = await self.convert_images_to_text_batch(self.image_urls, max_concurrency=1)
            # If it fails
            if not success:
                # Get exceptions from the results
                exceptions = [result for result in results if isinstance(result, Exception)]
                # If there are any exceptions, raise a ValueError with exceptions
                raise ValueError(f"Failed to convert image to text in batch. Errors: {repr(exceptions)}")
            
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            
            pbar.update(1)
            # GET OPENAI SIMILARITIES
            pbar.set_description("Step 3/5: Calculating OpenAI similarities")
            time.sleep(1)
            # Check if keyword similarity files already exist
            already_exist = all(os.path.exists(os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_openai", os.path.basename(url).replace('.jpg', '.pkl'))) for url in self.image_urls)

            # If not all files exist, calculate OpenAI similarities
            if not already_exist:
                # For all images at once (batch processing)
                success, results = await self.get_similarity_openai_batch(self.image_urls, self.user_queries, max_concurrency=1)
                # If it fails
                if not success:
                    # Get exceptions from the results
                    exceptions = [result for result in results if isinstance(result, Exception)]
                    # If there are any exceptions, raise a ValueError with exceptions
                    raise ValueError(f"Failed to get openai similarities in batch. Errors: {repr(exceptions)}")
            
            # Store the OpenAI similarity values
            for i, url in enumerate(self.image_urls):
                # Get the full path to the similarity file
                file_name = os.path.basename(url).replace('.jpg', '.pkl')
                file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_openai", file_name)

                # If the openai similarity file for the image already exists
                if os.path.exists(file_path):
                    
                    # Read the similarities from the file
                    with open(file_path, 'rb') as f:
                        similarities = pickle.load(f)

                    # Ensure that the number of similarities matches the number of user queries
                    assert len(similarities) == len(self.user_queries), f"Number of similarities ({len(similarities)}) does not match number of user queries ({len(self.user_queries)}) for image {url}."

                    # Store in the similarities dict
                    for j, query in enumerate(self.user_queries):
                        
                        # If the similarities dict doesn't have the current query, add it
                        if self.similarities.get(query, None) is None:
                            self.similarities[query] = {}

                        # If the openai key is not in the dict for query, add it
                        if self.similarities[query].get("openai", None) is None:
                            self.similarities[query]["openai"] = {}

                        # Store the similarity for the query and url
                        self.similarities[query]["openai"][url] = similarities[j]
                
                # If the file does not exist, create and save it
                else:
                    
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    # Create the list for similarity of the image to each user query
                    image_similarities = []

                    # For each user query
                    for j, query in enumerate(self.user_queries):
                        
                        # If the similarities dict doesn't have the current query, add it
                        if self.similarities.get(query, None) is None:
                            self.similarities[query] = {}

                        # If the openai key is not in the dict for query, add it
                        if self.similarities[query].get("openai", None) is None:
                            self.similarities[query]["openai"] = {}

                        # Get the similarity for the query and url
                        similarity = results[i][j]
                        image_similarities.append(similarity)

                        # Store the similarity for the query and url
                        self.similarities[query]["openai"][url] = similarity

                    # Save the similarities to a file
                    with open(file_path, 'wb') as f:
                        pickle.dump(image_similarities, f)

            # Write all openai similarities to a file for readability/debugging
            all_openai_similarities_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_openai", "0_all.txt")
            
            with open(all_openai_similarities_file_path, 'w') as f:
                for i, url in enumerate(self.image_urls):
                    # Get the path for pickle file that contains the similarities for the image
                    pickle_file_path = all_openai_similarities_file_path.replace("0_all.txt", f"{i+1}.pkl")
                    # Read the similarities from the pickle file
                    with open(pickle_file_path, 'rb') as pf:
                        current_image_similarities = pickle.load(pf)

                    formatted_similarities = [f"{similarity:.2f}" for similarity in current_image_similarities]
                    formatted_representation = "[" + ", ".join(formatted_similarities) + "]"

                    # Write the image similarities to the file
                    f.write(f"Image {i+1:<2}: {formatted_representation}\n\n")
            

            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------


            pbar.update(1)
            # GET KEYWORD SIMILARITIES
            pbar.set_description("Step 4/5: Calculating keyword similarities")
            time.sleep(1)
            # Check if keyword similarity files already exist
            already_exist = all(os.path.exists(os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_keyword", os.path.basename(url).replace('.jpg', '.pkl'))) for url in self.image_urls)
            
            # If not all files exist, calculate keyword similarities
            if not already_exist:
                
                success, similarities_keyword = self.get_similarities_keyword(self.image_urls, self.user_queries, self.descriptions)
                
                # If the keyword similarity calculation wasn't successful, try once more after a short delay
                if not success:
                    
                    time.sleep(5)
                    success, similarities_keyword = self.get_similarities_keyword(self.image_urls, self.user_queries, self.descriptions)
                    
                    # If it still fails
                    if not success:
                        raise ValueError("Failed to calculate keyword similarities after two attempts.")
            
            # Store the keyword similarity values
            for i, url in enumerate(self.image_urls):
                
                # Get the full path to the similarity file
                file_name = os.path.basename(url).replace('.jpg', '.pkl')
                file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_keyword", file_name)

                # If the keyword similarity file for the image already exists
                if os.path.exists(file_path):
                    
                    # Read the similarities from the file
                    with open(file_path, 'rb') as f:
                        similarities = pickle.load(f)

                    # Ensure that the number of similarities matches the number of user queries
                    assert len(similarities) == len(self.user_queries), f"Number of similarities ({len(similarities)}) does not match number of user queries ({len(self.user_queries)}) for image {url}."

                    # Store in the similarities dict
                    for j, query in enumerate(self.user_queries):
                        
                        # If the similarities dict doesn't have the current query, add it
                        if self.similarities.get(query, None) is None:
                            self.similarities[query] = {}

                        # If the keyword key is not in the dict for query, add it
                        if self.similarities[query].get("keyword", None) is None:
                            self.similarities[query]["keyword"] = {}

                        # Store the similarity for the query and url
                        self.similarities[query]["keyword"][url] = similarities[j] 
                
                # If the file does not exist, create and save it
                else:
                    
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    # Create the list for similarity of the image to each user query
                    image_similarities = []

                    # For each user query
                    for j, query in enumerate(self.user_queries):
                        
                        # If the similarities dict doesn't have the current query, add it
                        if self.similarities.get(query, None) is None:
                            self.similarities[query] = {}

                        # If the keyword key is not in the dict for query, add it
                        if self.similarities[query].get("keyword", None) is None:
                            self.similarities[query]["keyword"] = {}

                        # Get the similarity for the query and url
                        similarity = similarities_keyword[j][i]
                        image_similarities.append(similarity)

                        # Store the similarity for the query and url
                        self.similarities[query]["keyword"][url] = similarity

                    # Save the similarities to a file
                    with open(file_path, 'wb') as f:
                        pickle.dump(image_similarities, f)

            # Write all keyword similarities to a file for readability/debugging
            all_keyword_similarities_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_keyword", "0_all.txt")
            
            with open(all_keyword_similarities_file_path, 'w') as f:
                for i, url in enumerate(self.image_urls):
                    # Get the path for pickle file that contains the similarities for the image
                    pickle_file_path = all_keyword_similarities_file_path.replace("0_all.txt", f"{i+1}.pkl")
                    # Read the similarities from the pickle file
                    with open(pickle_file_path, 'rb') as pf:
                        current_image_similarities = pickle.load(pf)

                    formatted_similarities = [f"{similarity:.2f}" for similarity in current_image_similarities]
                    formatted_representation = "[" + ", ".join(formatted_similarities) + "]"

                    # Write the image similarities to the file
                    f.write(f"Image {i+1:<2}: {formatted_representation}\n\n")

            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            # ------------------------------------------------------------------

            pbar.update(1)
            # GET SEMANTIC SIMILARITIES
            pbar.set_description("Step 5/5: Calculating semantic similarities")
            time.sleep(1)
            # Check if semantic similarity files already exist
            already_exist = all(os.path.exists(os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_semantic", os.path.basename(url).replace('.jpg', '.pkl'))) for url in self.image_urls)
            
            # If not all files exist, calculate keyword similarities
            if not already_exist:
                
                success, similarities_semantic = self.get_similarities_semantic(self.image_urls, self.user_queries, self.descriptions, show_progress=True)
                
                # If the semantic similarity calculation wasn't successful, try once more after a short delay
                if not success:
                    
                    time.sleep(5)
                    success, similarities_semantic = self.get_similarities_semantic(self.image_urls, self.user_queries, self.descriptions, show_progress=False)
                    
                    # If it still fails
                    if not success:
                        raise ValueError("Failed to calculate semantic similarities after two attempts.")
            
            # Store the semantic similarity values
            for i, url in enumerate(self.image_urls):
                
                # Get the full path to the similarity file
                file_name = os.path.basename(url).replace('.jpg', '.pkl')
                file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_semantic", file_name)

                # If the semantic similarity file for the image already exists
                if os.path.exists(file_path):
                    
                    # Read the similarities from the file
                    with open(file_path, 'rb') as f:
                        similarities = pickle.load(f)

                    # Ensure that the number of similarities matches the number of user queries
                    assert len(similarities) == len(self.user_queries), f"Number of similarities ({len(similarities)}) does not match number of user queries ({len(self.user_queries)}) for image {url}."

                    # Store in the similarities dict
                    for j, query in enumerate(self.user_queries):
                        
                        # If the similarities dict doesn't have the current query, add it
                        if self.similarities.get(query, None) is None:
                            self.similarities[query] = {}

                        # If the semantic key is not in the dict for query, add it
                        if self.similarities[query].get("semantic", None) is None:
                            self.similarities[query]["semantic"] = {}

                        # Store the similarity for the query and url
                        self.similarities[query]["semantic"][url] = similarities[j] 
                
                # If the file does not exist, create and save it
                else:
                    
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    # Create the list for similarity of the image to each user query
                    image_similarities = []

                    # For each user query
                    for j, query in enumerate(self.user_queries):
                        
                        # If the similarities dict doesn't have the current query, add it
                        if self.similarities.get(query, None) is None:
                            self.similarities[query] = {}

                        # If the semantic key is not in the dict for query, add it
                        if self.similarities[query].get("semantic", None) is None:
                            self.similarities[query]["semantic"] = {}

                        # Get the similarity for the query and url
                        similarity = similarities_semantic[j][i]
                        image_similarities.append(similarity)

                        # Store the similarity for the query and url
                        self.similarities[query]["semantic"][url] = similarity

                    # Save the similarities to a file
                    with open(file_path, 'wb') as f:
                        pickle.dump(image_similarities, f)

            # Write all semantic similarities to a file for readability/debugging
            all_semantic_similarities_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data/similarities_semantic", "0_all.txt")
            
            with open(all_semantic_similarities_file_path, 'w') as f:
                for i, url in enumerate(self.image_urls):
                    # Get the path for pickle file that contains the similarities for the image
                    pickle_file_path = all_semantic_similarities_file_path.replace("0_all.txt", f"{i+1}.pkl")
                    # Read the similarities from the pickle file
                    with open(pickle_file_path, 'rb') as pf:
                        current_image_similarities = pickle.load(pf)

                    formatted_similarities = [f"{similarity:.2f}" for similarity in current_image_similarities]
                    formatted_representation = "[" + ", ".join(formatted_similarities) + "]"

                    # Write the image similarities to the file
                    f.write(f"Image {i+1:<2}: {formatted_representation}\n\n")
                
            pbar.update(1)

    async def start_chat(self):

        # Create a config object with a unique thread_id for memory tracking, and other runtime information (RoomSearch object)
        config = {
            "configurable": {
                "thread_id": 1,
                "room_search_object": self
            }
        }

        # Create initial state with a system message for context
        initial_state = {
            "messages": [SystemMessage(content=room_search_system_prompt)],
            "latest_tool_call": None,
            "user_query": None,
            "user_query_tool": None,
            "similarity_type": "openai",
            "room_search_results": None
        }
        
        # Invoke graph once to initialize the state
        state = await room_search_graph.ainvoke(initial_state, config=config)
        # pretty_print_object(state)

        while True:
            
            # Get input from the user
            # user_input = input("\nUser: ").strip()
            user_input = (await aioconsole.ainput("\nUser: ")).strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Get the current state of the graph
            graph_state = room_search_graph.get_state(config).values
            # pretty_print_object(graph_state)
            # Merge the graph state with the current state (updating the state with the latest values)
            state = {**state, **graph_state}
        
            # Invoke the graph with the user input and get the (async) stream of responses (a coroutine object)
            astream = room_search_graph.astream(
                input={**state, "messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates"
            )

            # For every step (a node's execution and its corresponding updates in the state) in the stream
            async for step in astream:

                # For every state dictionary delta (dictionary of updated fields and update values)
                for state_delta in step.values():

                    # If there is an update to the messages field in the state
                    if (state_delta is not None) and ("messages" in state_delta):
                        # For every message in the list of message updates
                        for message in state_delta["messages"]:
                            # If the message is an ai response and contains content (not a tool call), print it back to the user
                            if message.type == "ai" and message.content != "":
                                print(f"\nAssistant: {message.content}")

    @staticmethod
    async def collect_astream(astream):
        results = []
        async for step in astream:
            results.append(step)

        return results
    
    def start_chat_sync(self):
        # Create a config object with a unique thread_id for memory tracking, and other runtime information (RoomSearch object)
        config = {
            "configurable": {
                "thread_id": 1,
                "room_search_object": self
            }
        }

        # Create initial state with a system message for context
        initial_state = {
            "messages": [SystemMessage(content=room_search_system_prompt)],
            "latest_tool_call": None,
            "user_query": None,
            "user_query_tool": None,
            "similarity_type": "openai",
            "room_search_results": None
        }
        
        # Invoke graph once to initialize the state
        state = asyncio.run(room_search_graph.ainvoke(initial_state, config=config))
        # pretty_print_object(state)

        while True:
            
            # Get input from the user
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Get the current state of the graph
            graph_state = room_search_graph.get_state(config).values
            # pretty_print_object(graph_state)
            # Merge the graph state with the current state (updating the state with the latest values)
            state = {**state, **graph_state}
        
            # Invoke the graph with the user input and get the (async) stream of responses (a coroutine object)
            astream = room_search_graph.astream(
                input={**state, "messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates"
            )

            # For every step (a node's execution and its corresponding updates in the state) in the stream
            for step in asyncio.run(self.collect_astream(astream)):

                # For every state dictionary delta (dictionary of updated fields and update values)
                for state_delta in step.values():

                    # If there is an update to the messages field in the state
                    if (state_delta is not None) and ("messages" in state_delta):
                        # For every message in the list of message updates
                        for message in state_delta["messages"]:
                            # If the message is an ai response and contains content (not a tool call), print it back to the user
                            if message.type == "ai" and message.content != "":
                                print(f"\nAssistant: {message.content}")



if __name__ == "__main__":
        
        async def main():
            # Suppress SSL warnings for insecure requests
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Get image urls and user queries from parameter file
            with open("parameters.json", "r") as f:
                parameters = json.load(f)
            image_urls = parameters["IMAGE_URLS"]
            user_queries = parameters["USER_QUERIES"]

            # Create an instance of the RoomSearch class with image urls and user queries
            room_search = await RoomSearch.create_instance_async(image_urls, user_queries)

            # Start chatting with the RoomSearch instance
            await room_search.start_chat()

        def main_sync():

            # Suppress SSL warnings for insecure requests
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Get image urls and user queries from parameter file
            with open("parameters.json", "r") as f:
                parameters = json.load(f)
            image_urls = parameters["IMAGE_URLS"]
            user_queries = parameters["USER_QUERIES"]

            # Create an instance of the RoomSearch class with image urls and user queries
            room_search = asyncio.run(RoomSearch.create_instance_async(image_urls, user_queries))

            # Start chatting with the RoomSearch instance
            room_search.start_chat_sync()

        
        asyncio.run(main())