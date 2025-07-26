# main.py
import ast
import asyncio
import json
from random import randint
import random
from typing import Any
from dotenv import load_dotenv
from internetarchive import (
    File,
    configure,
    search_items,
)
import os
from langchain_core.tools import tool
from langchain_core.utils.json import parse_partial_json
from langchain_openai import AzureChatOpenAI, AzureOpenAI, ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import logging
from src.config.channels import channels
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright

_ = load_dotenv()

IMG_URL_ROOT = "https://free-images.com/"

# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200",
    "https://localhost",
    "https://localhost:4200",
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

class ImageArray(BaseModel):
    images: list[list[str]] = Field(description="A list of lists of image URLs")

class GoogleImageArray(BaseModel):
    images: list[str] = Field(description="A list of image URLs from Google")

class ImageSubject(BaseModel):
    subject: str = Field(description="The subject of the image")


@tool
async def get_subject_from_image(image_url: str) -> str:
    """
    Analyzes an image and returns its subject.
    Args:
        image_url (str): The URL of the image to analyze.
    Returns:
        str: The subject of the image.
    """
    vision_model = AzureChatOpenAI(
        model=os.getenv("OPENAI_VISION_MODEL", "gpt-4-mini"),
        api_version=os.getenv("AZURE_API_VERSION", "2024-05-01-preview"),
        azure_ad_token_provider=get_ailab_bearer_token_provider(),
        azure_endpoint=os.getenv(
            "AZURE_ENDPOINT", "https://ct-enterprisechat-api.azure-api.net/"
        ),
    )
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "What is the subject of this image? Respond with only the subject and nothing else."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    )
    
    response = await vision_model.ainvoke([message])
    return str(response.content)


@tool
async def get_main_image_array() -> list[list[str]]:
    """
    Retrieve a list of main images from Free-Images.com

    Returns:
        list[list[str]]: A list of lists containing image URLs.
        Each sublist contains the images for a specific category.
        The first sublist contains the images from the Stock Photos category.
        The second sublist contains the images from the Black and White category.
        The third sublist contains the images from the Vector category.
        The fourth sublist contains the images from the Art category.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(IMG_URL_ROOT)

        button_selectors = ["#dsws", "#dswb", "#dswc", "#dswa"]
        all_images = []

        for selector in button_selectors:
            await page.click(selector)
            await page.wait_for_timeout(1000)  # Wait for images to load
            
            images = await page.query_selector_all("#piccont img, #spiccont img")
            image_urls = [await img.get_attribute("src") for img in images]
            all_images.append(image_urls)

        await browser.close()
        return all_images

# Function to scroll to the bottom of the page
async def scroll_to_bottom(page):
    """
    Scroll to the bottom of the web page using Playwright.

    Args:
        page (Page): The Playwright page object to scroll.

    Returns:
        None
    """
    print("Scrolling...")
    previous_height = await page.evaluate("document.body.scrollHeight")
    while True:
        # Scroll to the bottom of the page
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == previous_height:
            break
        previous_height = new_height
    print("Reached the bottom of the page.")

@tool
async def get_images_from_google(subject: str, max_images: int = 10) -> list[str]:
    """
    Retrieve a list of high-quality images from Google Images.
    Args:
        subject (str): The subject to search for.
        max_images (int): The maximum number of images to retrieve.
    Returns:
       list[str]: A list of image URLs.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Navigate to Google Images with safe search on
        url = f"https://www.google.com/search?q={subject}&tbm=isch&safe=active"
        await page.goto(url, wait_until="networkidle")

        # Scroll to the bottom of the page to load more images
        await scroll_to_bottom(page)
        await page.wait_for_selector('div[data-id="mosaic"]')  # Wait for the image section to appear

        # Find all image elements on the page
        image_elements = await page.query_selector_all('div[data-attrid="images universal"]')
        print(f"Found {len(image_elements)} image elements on the page.")

        images_found = 0
        image_data_list = []

        # Iterate through the image elements
        for idx, image_element in enumerate(image_elements):
            if images_found >= max_images:
                print(f"Reached max image limit of {max_images}.")
                break
            try:
                print(f"Processing image {idx + 1}...")
                # Click on the image to get a full view
                await image_element.click()
                await page.wait_for_selector("img.sFlh5c.FyHeAf.iPVvYb[jsaction]")

                img_tag = await page.query_selector("img.sFlh5c.FyHeAf.iPVvYb[jsaction]")
                if not img_tag:
                    print(f"Failed to find image tag for element {idx + 1}")
                    continue

                # Get the image URL
                img_url = await img_tag.get_attribute("src")

                image_data_list.append(img_url)
                images_found += 1
            except Exception as e:
                print(f"Error processing image {idx + 1}: {e}")
                continue
        
        return image_data_list


def get_ailab_bearer_token_provider():
    from azure.identity import (
        DefaultAzureCredential,
        get_bearer_token_provider as _get_bearer_token_provider,
    )

    token_provider = _get_bearer_token_provider(
        DefaultAzureCredential(),
        os.getenv("AZURE_AD_TOKEN_PROVIDER", "api://ailab/Model.Access"),
    )
    return token_provider


# setup AI agent
model = AzureChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-10-01-preview"),
    azure_ad_token_provider=get_ailab_bearer_token_provider(),
    azure_endpoint=os.getenv(
        "AZURE_ENDPOINT", "https://ct-enterprisechat-api.azure-api.net/"
    ),
)
tools = [get_main_image_array, get_images_from_google, get_subject_from_image]

# Create parsers for different outputs
main_image_parser = PydanticOutputParser(pydantic_object=ImageArray)
google_image_parser = PydanticOutputParser(pydantic_object=GoogleImageArray)
subject_parser = PydanticOutputParser(pydantic_object=ImageSubject)


template = """
You are an AI agent that retrieves images and identifies their subjects.

If the user asks for "main images" or "images from free-images.com", use the `get_main_image_array` tool.
Your FINAL ANSWER MUST be the direct output of the tool `get_main_image_array`.
{main_image_format_instructions}

If the user asks for images from "Google" or "Google Images" for a specific subject, use the `get_images_from_google` tool.
The `get_images_from_google` tool allows for two parameters: `subject` and `max_images`.
The `subject` parameter is required, and the `max_images` parameter is optional.
You must use the exact subject provided as the `subject` parameter.
If the user provides a max number of images: you must use that number as the `max_images` parameter.
Your FINAL ANSWER MUST be the direct output of the tool `get_images_from_google`.
{google_image_format_instructions}

If the user provides an image URL and asks for its subject, use the `get_subject_from_image` tool.
Your FINAL ANSWER MUST be the direct output of the tool `get_subject_from_image`.
{subject_format_instructions}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(
    main_image_format_instructions=main_image_parser.get_format_instructions(),
    google_image_format_instructions=google_image_parser.get_format_instructions(),
    subject_format_instructions=subject_parser.get_format_instructions(),
)
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Subject-determining agent
subject_model = AzureChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-10-01-preview"),
    azure_ad_token_provider=get_ailab_bearer_token_provider(),
    azure_endpoint=os.getenv(
        "AZURE_ENDPOINT", "https://ct-enterprisechat-api.azure-api.net/"
    ),
)
subject_template = """
You are an AI agent that determines the subject of a user's query.
The user will provide a query, and you must respond with only the subject of that query.
Do not add any other text or modifications to the subject.
"""
subject_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", subject_template),
        ("human", "{input}"),
    ]
)
subject_agent = subject_prompt | subject_model


# Generated docs endpoint
@app.get("/")
async def get_docs():
    return RedirectResponse("/docs")


@app.get("/images/main")
async def get_main_array() -> Any:
    result = await agent_executor.ainvoke(
        {"input": "Get the main array of images"}
    )
    agent_output = result["output"]
    try:
        # Attempt to parse the full JSON output
        parsed_output = main_image_parser.parse(agent_output)
        return parsed_output.images
    except Exception:
        try:
            # If full parsing fails, try to parse partial JSON
            partial_json_output = parse_partial_json(agent_output)
            if isinstance(partial_json_output, dict) and "images" in partial_json_output:
                return partial_json_output["images"]
            return {"error": "Failed to parse agent output", "output": agent_output}
        except Exception as e:
            return {"error": "Failed to parse agent output", "details": str(e), "output": agent_output}


@app.get("/images/subject")
async def get_subject(image_url: str) -> Any:
    """
    Get the subject of an image.
    """
    # fully qualify the image url
    if not image_url.startswith("http"):
        image_url = f"{IMG_URL_ROOT}{image_url}"
        
    result = await agent_executor.ainvoke(
        {"input": f"What is the subject of this image? {image_url}"}
    )
    agent_output = result["output"]
    try:
        # Attempt to parse the full JSON output
        parsed_output = subject_parser.parse(agent_output)
        return parsed_output
    except Exception:
        try:
            # If full parsing fails, try to parse partial JSON
            partial_json_output = parse_partial_json(agent_output)
            if isinstance(partial_json_output, dict) and "subject" in partial_json_output:
                return partial_json_output
            return {"error": "Failed to parse agent output", "output": agent_output}
        except Exception as e:
            return {"error": "Failed to parse agent output", "details": str(e), "output": agent_output}

@app.get("/images/google")
async def get_google_images(subject: str, max_images: int = 10) -> Any:
    """
    Get images from Google Images for a given subject.
    """
    # Determine the subject of the query
    subject_response = await subject_agent.ainvoke({"input": subject})
    image_subject = subject_response.content

    result = await agent_executor.ainvoke(
        {"input": f"Get images from Google for the subject: {image_subject}. Return a maximum of {max_images} images."}
    )
    agent_output = result["output"]
    try:
        # Attempt to parse the full JSON output
        parsed_output = google_image_parser.parse(agent_output)
        return parsed_output.images
    except Exception:
        try:
            # If full parsing fails, try to parse partial JSON
            partial_json_output = parse_partial_json(agent_output)
            if isinstance(partial_json_output, dict) and "images" in partial_json_output:
                # If the partial JSON contains a list of lists, flatten it
                if partial_json_output["images"] and isinstance(partial_json_output["images"][0], list):
                    return [item for sublist in partial_json_output["images"] for item in sublist]
                return partial_json_output["images"]
            return {"error": "Failed to parse agent output", "output": agent_output}
        except Exception as e:
            return {"error": "Failed to parse agent output", "details": str(e), "output": agent_output}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
