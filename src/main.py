# main.py
import json
from random import randint
import random
from dotenv import load_dotenv
from internetarchive import (
    File,
    configure,
    search_items,
)
import os
from langchain_core.tools import tool
from langchain_core.utils.json import parse_partial_json
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from pydantic import BaseModel
import logging
from src.config.channels import channels
from fastapi.middleware.cors import CORSMiddleware
_ = load_dotenv()

# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://jennymaeleidig.github.io",
    "https://jennymaeleidig.github.io",
    "http://localhost",
    "http://localhost:4200",
    "https://localhost",
    "https://localhost:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging #
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VideoQuery(BaseModel):
    query: str


class Video(BaseModel):
    url: str
    title: str
    uploader: str = "Unknown uploader"
    duration: float = 0.0


def convert_to_video(file: File) -> Video:
    """Convert a File object from the Internet Archive to a video model."""
    # Access metadata safely, providing defaults if keys are missing
    metadata = file.item.metadata
    file_metadata = file.metadata
    logging.info(f"Converting file {file.name} to video")

    return Video(
        url=file.url,
        title=metadata.get("title", "No title available"),
        uploader=metadata.get("uploader", "Unknown uploader"),
        duration=float(file_metadata.get("length", 0.0)),
    )


ACCEPTED_VIDEO_FORMATS = [
    "h.264",
    "h.264 IA",
    "MPEG4",
    "Ogg Video",
    "WebM",
    "MPEG2",
    "RealMedia",
    "QuickTime",
    "Windows Media",
]


@tool
def get_random_video(search_term: str = "", collection: str = "") -> Video | None:
    """
    Retrieve a random video from the Internet Archive based on a search term.

    If a search term is provided, a random video matching the search term is returned.

    If a collection is provided, the search is limited to that collection.

    If no search term AND no collection is provided, a random video from the VHS commercials collection is returned.
    """
    logging.info(f"Getting random video with search term: {search_term} and collection: {collection}")

    ia_conf: str = os.path.abspath("./src/config/ia.ini")

    _ = configure(
        os.getenv("IA_USER") or "", os.getenv("IA_PASS") or "", config_file=ia_conf
    )

    query = "mediatype:movies"

    if search_term or collection:
        if search_term :
            query = f"{query} AND ({search_term})"
            logging.info(f"Searching for video with query: '{query}'")
        if collection:
            query = f"{query} AND collection:{collection}"
            logging.info(f"Searching for video in collection '{collection}' with query: '{query}'")
    else:
        query = f"{query} AND collection:vhscommercials"
        logging.info(
            f"Searching for random video in default collection with query: '{query}'"
        )

    num_results = int(search_items(query, config_file=ia_conf).num_found)
    logging.info(f"Total results found: {num_results}")

    max_attempts = 10  # Limit attempts to find a video
    for attempt in range(max_attempts):
        logging.info(
            f"Attempt {attempt + 1}/{max_attempts} to find a random video from {num_results} results."
        )
        random_page = randint(1, num_results)

        search_results = search_items(
            query, config_file=ia_conf, params={"rows": 1, "page": random_page}
        ).iter_as_items()

        try:
            search_results = search_items(
                query, config_file=ia_conf, params={"rows": 1, "page": random_page}
            ).iter_as_items()

            for item in search_results:
                for file in item.get_files(formats=ACCEPTED_VIDEO_FORMATS):
                    logging.info(f"Found video: {file.name}")
                    return convert_to_video(file)

            # If loop finishes without finding a suitable file in the item on this page
            logging.info(
                f"No suitable video file found in the item on page {random_page}. Retrying."
            )  # {{change 5}}

        except json.JSONDecodeError as e:
            logging.error(
                f"JSON decode error while processing search results on page {random_page}: {e}"
            )
            # Continue to the next attempt
            continue 
    logging.warning("No video found after maximum attempts.")
    return None

@tool
def get_collection_from_chanel(channel: str) -> str | None:
    """
    Retrieve a random collection from a specific channel.

    Args:
     channel (str): The channel to retrieve a collection from.
    
    Returns:
        str: The collection name.
    """
    logging.info(f"Getting collection from channel: {channel}")

    if channel.strip().lower() in channels.keys():
        collection = random.choice(channels[channel.strip().lower()])
        logging.info(f"Found collection: {collection} for channel: {channel}")
        return collection
    logging.info(f"No collection found for channel: {channel}")
    return None
        
# setup AI agent
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano-2025-04-14"))
tools = [get_random_video, get_collection_from_chanel]
template = f'''
    You are an AI agent that can retrieve a random video from Archive.org.
    You have access to two tools: `get_collection_from_chanel` and `get_random_video`.
    The `get_collection_from_chanel` tool retrieves a random collection from a specific channel.
    The tool accepts a `channel` parameter, which is the name of the channel to retrieve a collection from.
    The `channel` parameter must be one of the following channels: {{channels}}.
    If the user's query mentions a specific channel, you should call `get_collection_from_chanel` with that channel as the `channel` parameter.
    Otherwise, if the user's query does not mention a specific channel, you should not call `get_collection_from_chanel`.
    The `get_random_video` tool searches for a random video file across Archive.org:
    The tool `get_random_video` tool accepts an optional `collection parameter to specify the collection to search in, and an optional search_term` parameter to filter the search results.
    If you called `get_collection_from_chanel`, you should use the collection name as the `collection` parameter.
    Otherwise, if you did not call `get_collection_from_chanel`, you should not use the `collection` parameter.
    If the user's query mentions a specific search term, you should use that as the `search_term` parameter.
    Otherwise, if the user's query does not mention a specific search term, you should not use the `search_term` parameter.
    If the user does not provide a specific search criteria, you can call `get_random_video` without a `search_term` or `collection`.
    If the user's request contains any Not Safe For Work (NSFW) or inappropriate material, you should ignore their request call `get_random_video` without a `search_term` or `collection`.
    If the returned video contains any Not Safe For Work (NSFW) or inappropriate material, you should ignore the video and call `get_random_video` again with the same input parameters.
    Your FINAL ANSWER MUST be the video retrieved by the `get_random_video` tool, formatted STRICTLY as a JSON string.
    Include ONLY the JSON object itself, with no surrounding text or markdown. The JSON object must match the structure of the Video model with fields: `url`, `title`, `uploader`, and `duration`.
    If you cannot retrieve a video, retry this process up to 3 times. If you cannot retrieve a video after 3 attempts, stop retrying and return an empty JSON object.
'''
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Generated docs endpoint
@app.get("/")
async def get_docs():
    return RedirectResponse("/docs")


@app.post("/video")
async def get_video(query: VideoQuery) -> Video:
    logging.info(f"Received query: {query.query}")
    try:
        result = agent_executor.invoke({"input": query.query, "channels": (','.join(channels.keys()))})
        agent_output = result["output"] 

        logging.debug(f"Agent raw result: {agent_output}") 

        # Attempt to parse the JSON output more robustly 
        parsed_json = None 
        try: 
            # First, try parsing directly (might work if agent returns raw JSON) 
            parsed_json = json.loads(agent_output) 
            logging.info("Direct JSON parse successful.")
        except json.JSONDecodeError: 
            logging.debug("Direct JSON parse failed, attempting partial parse.") 
            # If direct parsing fails, try Langchain's partial parser 
            parsed_json = parse_partial_json(agent_output)
            logging.info("Partial JSON parse successful.")


        if not parsed_json: 
            logging.error(f"Failed to parse valid JSON from agent output: {agent_output}") 
            raise ValueError("Agent returned output that could not be parsed as valid JSON.") 
        # Validate and return using the Pydantic model
        video = Video(**parsed_json)
        logging.info(f"Returning video: {video}")
        return video

    except RuntimeError as e:
        logging.exception("Unable to find a suitable video or agent failed.")
        raise HTTPException(status_code=404, detail=str(e))
    except json.JSONDecodeError:
        logging.exception("Agent returned invalid JSON.")
        raise HTTPException(
            status_code=500, detail="Agent returned invalid video data."
        )
    except Exception as e:
        logging.exception("An unexpected error occurred while processing the query")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
