import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)))))

import urllib3
import asyncio
import json
import aioconsole

from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from typing import Annotated, Any, Optional, List
from typing_extensions import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph.pregel.io import AddableValuesDict

from assistant.tools.room_search_tool import RoomSearchTool
from assistant.room_search_agent import room_search_llm, room_search_system_prompt
from utils import pretty_print_object


# -----------------------------------------------------------------------------------
# Create tool instance
room_search_tool = RoomSearchTool()
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# Define the schema for the state of the graph with state variables
class RoomSearchState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    # Variable to store the "tool_call" dictionary for the tool that is expected to be called next
    latest_tool_call: Optional[dict[str, Any]]
    # User query to perform the room search (user input)
    user_query: Optional[str]
    # User query that is given as input to the room search tool
    user_query_tool: Optional[str]
    # Type of similarity chosen for image search
    similarity_type: Literal["openai", "keyword", "semantic"]
    # Variable to store the room search results retrieved by the room search tool
    room_search_results: Optional[List[Any]]
# -----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
async def room_search_node(state: RoomSearchState, config: RunnableConfig) -> Command[Literal["room_search_agent", "room_search_node"]]:
    
    try:
        # Get the latest tool call from the state
        latest_tool_call = state["latest_tool_call"]
        # Ensure that the tool call is in "approved" state
        assert latest_tool_call["status"] == "approved"

        # Get similarity type and input user query from the state
        similarity_type = state["similarity_type"]
        user_query = state["user_query_tool"]

        # Invoke the room search tool
        success, results = await room_search_tool.ainvoke({"config": config, "similarity_type": similarity_type, "user_query": user_query})

        # If the tool invocation was successful
        if success:
            # Mark latest tool call as completed
            latest_tool_call["status"] = "completed"

            # Print the urls of the most similar images to the console
            print(f"\nThe URLs of the most similar images for the query '{user_query.capitalize()}':")
            for i, url in enumerate(results, start=1):
                print(f"{i}. {url}")

            # Construct a tool response to notify the room search agent that the tool call was completed successfully
            tool_response = ToolMessage(
                tool_call_id=latest_tool_call["id"],
                content=f"{repr(results)}",
                status = "success"
            )

            # Once it's complete, update state variables and route back to the room search agent node
            # return Command(update={"messages": [SystemMessage(content=completion_message)], "latest_tool_call": latest_tool_call, "room_search_results": results}, goto="room_search_agent")
            return Command(update={"messages": [tool_response], "latest_tool_call": latest_tool_call, "room_search_results": results}, goto="room_search_agent")
        
        # If the tool invocation failed
        else:
            # Mark latest tool call as failed
            latest_tool_call["status"] = "failed"

            # Get the returned error message from the tool invocation
            error_message = results[0]

            # Construct a tool response to notify the room search agent that the tool call was rejected
            tool_response = ToolMessage(
                tool_call_id=latest_tool_call["id"],
                content=f"The room search tool has failed to retrieve the room images with the current parameters. The error message returned by the tool is: {error_message}.\n Please acknowledge the failure gracefully and inform the user about the issue. Ask them how they would like to proceed (e.g. try again, try with new query etc.).",
                status = "error"
            )

            # Update the state with the tool response and route back to the room search agent node
            return Command(update={"messages": [tool_response], "latest_tool_call": latest_tool_call, "room_search_results": results}, goto="room_search_agent")
    
    except Exception as e:
        # Inform user
        print(f"\nAn error has occured during the room image search: {str(e)}. \nTrying again...")

        # And try again by routing back to this node with the same state
        return Command(goto="room_search_node")
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# Node to prompt the user to review/configure/approve/reject the tool call, and manage its routing
async def human_tool_reviewer(state: RoomSearchState, config: RunnableConfig) -> Command[Literal["room_search_agent", "room_search_node", "human_tool_reviewer"]]:

    # Get the latest tool call from the state
    latest_tool_call = state["latest_tool_call"]
    # Ensure that it is at pending state
    assert latest_tool_call["status"] == "pending"

    # Get the user query from the arguments of the tool call
    user_query = latest_tool_call["args"]["user_query"]

    # Get the RoomSearch object from the config
    room_search: RoomSearch = config.get("configurable", {}).get("room_search_object", None)

    # Prompt the user to take action on the tool call
    prompt_text = (f"\nPlease enter '1', '2', '3', or '4' to search for most similar hotel rooms based on the default/examplary queries given below:" +
    f"\n\033[1m1.\033[0m {room_search.user_queries[0].capitalize()}" +
    f"\n\033[1m2.\033[0m {room_search.user_queries[1].capitalize()}" +
    f"\n\033[1m3.\033[0m {room_search.user_queries[2].capitalize()}" +
    f"\n\033[1m4.\033[0m {room_search.user_queries[3].capitalize()}" +
    f"\n" +
    f"\nEnter '5' to search for rooms based on your last user query:" +
    f"\n\033[1mUser query:\033[0m {user_query.capitalize()}" +
    f"\n" +
    f"\nEnter '0' to cancel current search and continue the conversation.")
    print(prompt_text)

    counter = 1
    while True:
        # If user provided an invalid input
        if counter > 1:
                print("\nInvalid input. Please enter '1', '2', '3', '4', '5' or '0'.")

        # Take user input
        print("\nUser input:")
        # user_choice = input().strip()
        user_choice = (await aioconsole.ainput()).strip()
        try:
            choice = int(user_choice)
        except ValueError:
            # If the input is not an integer, continue to prompt the user
            counter += 1
            continue
        
        # If default queries are selected
        if choice in [1, 2, 3, 4]:
            # Prompt the user a second time for the similarity type
            prompt_text2 = (f"\nPlease enter '1', '2', or '3' to select the similarity metric/type for image search:" +
            f"\n\033[1m1.\033[0m OpenAI (Images interpreted and compared directly with the query by GPT)" +
            f"\n\033[1m2.\033[0m Keyword-Based Similarity" +
            f"\n\033[1m3.\033[0m Vector-Based Semantic Similarity" +
            f"\n" +
            f"\nEnter '0' to cancel current search and continue the conversation." +
            f"\n" +
            f"\nEnter '00' to go back to the query selection screen.")
            print(prompt_text2)

            # Set another counter and prompt the user to select the similarity type for the room search
            counter2 = 1
            while True:
                # If user provided an invalid input
                if counter2 > 1:
                    print("\nInvalid input. Please enter '1', '2', '3', '0' or '00'.")

                # Take the second user input
                print("\nUser input:")
                # user_choice2 = input().strip()
                user_choice2 = (await aioconsole.ainput()).strip()

                # If the user selected a similarity type
                if user_choice2 in ['1', '2', '3']:
                    # Update the status of the tool call to "approved" and route to the room search node with the selected query and similarity type for tool invocation
                    latest_tool_call["status"] = "approved"
                    return Command(update={
                        "user_query": user_query,
                        "user_query_tool": room_search.user_queries[choice - 1],
                        "similarity_type": ["openai", "keyword", "semantic"][int(user_choice2) - 1],
                        "latest_tool_call": {
                            **latest_tool_call,
                            "args": {
                                **latest_tool_call["args"],
                                "user_query": room_search.user_queries[choice - 1],
                                "similarity_type": ["openai", "keyword", "semantic"][int(user_choice2) - 1]
                            }
                        }
                    }, goto="room_search_node")
                
                # If the user wants to cancel the search and continue the conversation
                elif user_choice2 == '0':
                    # Update the status of the tool call to "rejected"
                    latest_tool_call["status"] = "rejected"
                    # Construct a tool response to notify the room search agent that the tool call was rejected
                    tool_response = ToolMessage(
                        tool_call_id=latest_tool_call["id"],
                        content="The user has manually rejected the tool call for searching room images with the current parameters. This is not an error — it indicates that the user likely wants to update or change the search criteria (e.g., their search query). Please acknowledge the rejection gracefully and ask the user how they'd like to proceed or what they'd like to change about their request.",
                        status = "error"
                    )
                    # Update the state with the tool response and route back to the room search agent node
                    return Command(update={"messages": [tool_response]}, goto="room_search_agent")
               
                # If the user wants to go back to the query selection screen
                elif user_choice2 == '00':
                    # Route back to this node to prompt the user again
                    return Command(goto="human_tool_reviewer")
                # If the user entered an invalid choice
                else:
                    counter2 += 1
                    continue
        
        # If the user selected the 5th option to search for rooms based on their last user query
        elif choice == 5:
            # Prompt the user a second time for the similarity type
            prompt_text2 = (f"\nPlease enter '1', '2', or '3' to select the similarity metric/type for image search:" +
            f"\n\033[1m1.\033[0m OpenAI (Images interpreted and compared directly with the query by GPT)" +
            f"\n\033[1m2.\033[0m Keyword-Based Similarity" +
            f"\n\033[1m3.\033[0m Vector-Based Semantic Similarity" +
            f"\n" +
            f"\nEnter '0' to cancel current search and continue the conversation." +
            f"\n" +
            f"\nEnter '00' to go back to the query selection screen.")
            print(prompt_text2)

            # Set another counter and prompt the user to select the similarity type for the room search
            counter2 = 1
            while True:
                # If user provided an invalid input
                if counter2 > 1:
                    print("\nInvalid input. Please enter '1', '2', '3', '0' or '00'.")

                # Take the second user input
                print("\nUser input:")
                # user_choice2 = input().strip()
                user_choice2 = (await aioconsole.ainput("")).strip()

                # If the user selected a similarity type
                if user_choice2 in ['1', '2', '3']:
                    # Update the status of the tool call to "approved" and route to the room search node with the selected query and similarity type for tool invocation
                    latest_tool_call["status"] = "approved"
                    return Command(update={
                        "user_query": user_query,
                        "user_query_tool": user_query,
                        "similarity_type": ["openai", "keyword", "semantic"][int(user_choice2) - 1],
                        "latest_tool_call": {
                            **latest_tool_call,
                            "args": {
                                **latest_tool_call["args"],
                                "user_query": user_query,
                                "similarity_type": ["openai", "keyword", "semantic"][int(user_choice2) - 1]
                            }
                        }
                    }, goto="room_search_node")
                
                # If the user wants to cancel the search and continue the conversation
                elif user_choice2 == '0':
                    # Update the status of the tool call to "rejected"
                    latest_tool_call["status"] = "rejected"
                    # Construct a tool response to notify the room search agent that the tool call was rejected
                    tool_response = ToolMessage(
                        tool_call_id=latest_tool_call["id"],
                        content="The user has manually rejected the tool call for searching room images with the current parameters. This is not an error — it indicates that the user likely wants to update or change the search criteria (e.g., their search query). Please acknowledge the rejection gracefully and ask the user how they'd like to proceed or what they'd like to change about their request.",
                        status = "error"
                    )
                    # Update the state with the tool response and route back to the room search agent node
                    return Command(update={"messages": [tool_response]}, goto="room_search_agent")
               
                # If the user wants to go back to the query selection screen
                elif user_choice2 == '00':
                    # Route back to this node to prompt the user again
                    return Command(goto="human_tool_reviewer")
                # If the user entered an invalid choice
                else:
                    counter2 += 1
                    continue

        # If the user selected to cancel the current search and continue the conversation
        elif choice == 0:
            # Update the status of the tool call to "rejected"
            latest_tool_call["status"] = "rejected"
            # Construct a tool response to notify the room search agent that the tool call was rejected
            tool_response = ToolMessage(
                tool_call_id=latest_tool_call["id"],
                content="The user has manually rejected the tool call for searching room images with the current parameters. This is not an error — it indicates that the user likely wants to update or change the search criteria (e.g., their search query). Please acknowledge the rejection gracefully and ask the user how they'd like to proceed or what they'd like to change about their request.",
                status = "error"
            )
            # Update the state with the tool response and route back to the room search agent node
            return Command(update={"messages": [tool_response]}, goto="room_search_agent")
        
        # If the user entered an invalid choice
        else:
            counter += 1
            continue
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# Main node of the room search agent pipeline that handles user interactions and tool calls
def room_search_agent(state: RoomSearchState) -> Command[Literal["room_search_agent", "human_tool_reviewer", END]]:

    # Retrieve the last message from the state
    last_message = state["messages"][-1]

    # If the last message is a system message
    if last_message.type == "system":
        return Command(goto=END)
    
    # If the last message is a user, or tool message
    elif last_message.type in ["human", "tool"]:
        # If the last message is a successfull tool response, go to END node (because results are already presented to the user, manually in the room_search_node)
        if last_message.type == "tool" and last_message.status == "success":
            return Command(goto=END)
        else:
            # Then invoke the llm with the current state messages and return the updated state
            response = room_search_llm.invoke(state["messages"])
            return Command(update={"messages": [response]}, goto="room_search_agent")
    
    # If the last message is an ai message
    else:
        # Get the valid and invalid tool calls attributes of the last message
        tool_calls = last_message.tool_calls
        invalid_tool_calls = last_message.invalid_tool_calls
        # Since we disabled parallel tool calls, there should be at most one tool call
        assert len(tool_calls) <= 1

        # If the last messaage is a valid tool call
        if tool_calls:
            # Get the tool call dictionary and associated tool name
            tool_call = tool_calls[0]
            tool_name = tool_call["name"]
            # Room search agent is bound only to the room search tool, so this should be the only possible tool call
            assert tool_name == "room_image_search"

            # Route to the human tool reviewer node which requests human approval before calling the tool (human in the loop)
            # Also, add an additional status key to the tool call dictionary to track its status
            tool_call["status"] = "pending"
            return Command(update={"latest_tool_call": tool_call}, goto="human_tool_reviewer")
    
        # If the last message is an invalid tool call (message type that llm can
        # handle by making further reasoning by itself)
        elif invalid_tool_calls:
            # Notify the user
            print("\n[Tool Error] Invalid tool call received:")
            pretty_print_object(invalid_tool_calls)

            # Then invoke the llm with the current state messages and return the updated state
            response = room_search_llm.invoke(state["messages"])
            return Command(update={"messages": [response]}, goto="room_search_agent")
    
        else:
            # Then the last message is assistant/ai type (which already is a response from the llm)
            # So there is no need for invocation, return the current state with no changes and halt the pipeline (END --> until next user interaction)
            return Command(goto=END)
# -----------------------------------------------------------------------------------



 # Build the graph
builder = StateGraph(RoomSearchState)
builder.add_node("room_search_agent", room_search_agent)
builder.add_node("room_search_node", room_search_node)
builder.add_node("human_tool_reviewer", human_tool_reviewer)
builder.add_edge(START, "room_search_agent")

# Create memory and compile the graph with the memory
checkpointer = MemorySaver()
room_search_graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":

    async def main():

        from RoomSearch import RoomSearch

        try:
            # Generate the graph image and save it to the current file's directory
            image_path = os.path.join(os.path.dirname(__file__), "room_search_graph.png")
            room_search_graph.get_graph().draw_mermaid_png(output_file_path=image_path)
        except Exception:
            pass

        # Suppress SSL warnings for insecure requests
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        # Get image urls and user queries from parameter file
        with open("parameters.json", "r") as f:
            parameters = json.load(f)
        image_urls = parameters["IMAGE_URLS"]
        user_queries = parameters["USER_QUERIES"]

        room_search = await RoomSearch.create_instance_async(image_urls, user_queries)


        # Create a config object with a unique thread_id for memory tracking, and other runtime information (RoomSearch object)
        config = {
            "configurable": {
                "thread_id": 1,
                "room_search_object": room_search
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
            user_input = input("\nUser: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            graph_state = room_search_graph.get_state(config).values
            pretty_print_object(graph_state)
            state = {**state, **graph_state}
        
            # Invoke the graph with the user input and get the stream of responses
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

    asyncio.run(main())