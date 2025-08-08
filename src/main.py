# main.py
import asyncio
import json
from typing import Any, Optional
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_core.utils.json import parse_partial_json
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import logging
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright, Browser
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

_ = load_dotenv()

# Global browser instance for connection pooling
_browser_instance: Optional[Browser] = None
_browser_lock = asyncio.Lock()

# Cache for frequently accessed data
_image_cache = {}
_cache_lock = asyncio.Lock()

# Thread pool for CPU-intensive tasks
_thread_pool = ThreadPoolExecutor(max_workers=4)

async def get_browser():
    """Get or create a shared browser instance for better performance."""
    global _browser_instance
    async with _browser_lock:
        if _browser_instance is None or not _browser_instance.is_connected():
            playwright = await async_playwright().start()
            _browser_instance = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding'
                ]
            )
    return _browser_instance

async def close_browser():
    """Close the shared browser instance."""
    global _browser_instance
    async with _browser_lock:
        if _browser_instance and _browser_instance.is_connected():
            await _browser_instance.close()
            _browser_instance = None

def repair_truncated_json(json_str: str) -> str:
    """
    Attempt to repair truncated JSON by closing incomplete structures.
    """
    json_str = json_str.strip()
    
    # Handle case where JSON is truncated mid-string (like "hand_ke...")
    if '...' in json_str:
        # Find the last complete entry before the truncation
        truncation_pos = json_str.find('...')
        # Find the last complete quote before truncation
        last_complete_quote = json_str.rfind('"', 0, truncation_pos)
        if last_complete_quote > 0:
            # Find the quote that starts this string
            start_quote = json_str.rfind('"', 0, last_complete_quote)
            if start_quote > 0:
                # Remove the incomplete string entry
                json_str = json_str[:start_quote]
                # Remove trailing comma if present
                json_str = json_str.rstrip(', ')
    
    # Handle incomplete strings at the end
    if not json_str.endswith(('"', ']', '}', ',')):
        # Find the last complete entry
        last_quote = json_str.rfind('"')
        if last_quote > 0:
            # Check if this quote is closing a string
            prev_quote = json_str.rfind('"', 0, last_quote)
            if prev_quote > 0:
                # Keep only up to the last complete entry
                json_str = json_str[:last_quote + 1]
    
    # Count brackets and braces to determine what needs to be closed
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')
    
    # Close any incomplete arrays
    for _ in range(open_brackets):
        json_str += ']'
    
    # Close any incomplete objects
    for _ in range(open_braces):
        json_str += '}'
    
    return json_str

def parse_llm_response(agent_output: str, parser: PydanticOutputParser[Any], extract_field: Optional[str] = None) -> Any:
    """
    Unified function to parse LLM responses with comprehensive error handling.
    
    Args:
        agent_output: Raw output from the LLM agent
        parser: Pydantic parser for the expected output format
        extract_field: Optional field to extract from parsed JSON (e.g., "images", "subject")
    
    Returns:
        Parsed data or error response
    """
    # Clean the output - remove any non-JSON text
    cleaned_output = agent_output.strip()
    if not cleaned_output.startswith('{'):
        # Try to find JSON in the output
        json_start = cleaned_output.find('{')
        if json_start != -1:
            cleaned_output = cleaned_output[json_start:]
    
    try:
        # Attempt to parse the full JSON output
        parsed_output = parser.parse(cleaned_output)
        if extract_field:
            return getattr(parsed_output, extract_field)
        return parsed_output
    except Exception as parse_error:
        try:
            # Try to repair truncated JSON
            repaired_json = repair_truncated_json(cleaned_output)
            direct_json = json.loads(repaired_json)
            if isinstance(direct_json, dict) and extract_field and extract_field in direct_json:
                result = direct_json[extract_field]
                # Handle special case for Google images that might be nested lists
                if extract_field == "images" and result and isinstance(result[0], list):
                    return [item for sublist in result for item in sublist]
                return result
            elif not extract_field:
                return direct_json
        except Exception:
            pass
        
        try:
            # If repair fails, try to parse partial JSON
            partial_json_output = parse_partial_json(cleaned_output)
            if isinstance(partial_json_output, dict) and extract_field and extract_field in partial_json_output:
                result = partial_json_output[extract_field]
                # Handle special case for Google images that might be nested lists
                if extract_field == "images" and result and isinstance(result[0], list):
                    return [item for sublist in result for item in sublist]
                return result
            elif not extract_field:
                return partial_json_output
        except Exception:
            pass
            
        return {
            "error": "Failed to parse agent output", 
            "details": str(parse_error), 
            "output": agent_output[:500] + "..." if len(agent_output) > 500 else agent_output
        }

IMG_URL_ROOT = "https://free-images.com/"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logging.info("Starting up application...")
    # Pre-warm the browser instance
    await get_browser()
    logging.info("Browser instance initialized")
    
    yield
    
    # Shutdown
    logging.info("Shutting down application...")
    await close_browser()
    _thread_pool.shutdown(wait=True)
    logging.info("Resources cleaned up")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:4200",
    "https://localhost",
    "https://localhost:3000",
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
    # Check cache first
    cache_key = "main_images"
    async with _cache_lock:
        if cache_key in _image_cache:
            cache_time, cached_data = _image_cache[cache_key]
            # Cache for 5 minutes
            if time.time() - cache_time < 300:
                return cached_data

    browser = await get_browser()
    page = await browser.new_page()
    
    try:
        await page.goto(IMG_URL_ROOT, wait_until="domcontentloaded")

        button_selectors = ["#dsws", "#dswb", "#dswc", "#dswa"]
        all_images = []

        # Process categories sequentially but with optimized waits
        for selector in button_selectors:
            try:
                await page.click(selector)
                # Wait for images to load with shorter timeout
                await page.wait_for_timeout(800)  # Reduced from 1000ms
                
                images = await page.query_selector_all("#piccont img, #spiccont img")
                image_urls = await asyncio.gather(*[img.get_attribute("src") for img in images])
                category_images = [url for url in image_urls if url]
                all_images.append(category_images)
            except Exception as e:
                logging.warning(f"Error processing category {selector}: {e}")
                all_images.append([])  # Add empty list to maintain structure
        
        # Ensure we always return 4 lists
        while len(all_images) < 4:
            all_images.append([])
        
        # Cache the result
        async with _cache_lock:
            _image_cache[cache_key] = (time.time(), all_images)
        
        return all_images
    finally:
        await page.close()

# Optimized scrolling function
async def scroll_to_bottom_optimized(page, max_scroll_time: int = 10):
    """
    Optimized scrolling that limits time spent scrolling.
    """
    start_time = time.time()
    previous_height = await page.evaluate("document.body.scrollHeight")
    
    while time.time() - start_time < max_scroll_time:
        # Scroll to the bottom of the page
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(0.8)  # Reduced wait time
        new_height = await page.evaluate("document.body.scrollHeight")
        if new_height == previous_height:
            break
        previous_height = new_height

# adapted from https://scrapingant.com/blog/how-to-scrape-google-images
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
    # Check cache first
    cache_key = f"google_images_{subject}_{max_images}"
    async with _cache_lock:
        if cache_key in _image_cache:
            cache_time, cached_data = _image_cache[cache_key]
            # Cache for 10 minutes
            if time.time() - cache_time < 600:
                return cached_data

    browser = await get_browser()
    page = await browser.new_page()
    
    try:
        # Navigate to Google Images with safe search on
        url = f"https://www.google.com/search?q={subject}&tbm=isch&safe=active"
        await page.goto(url, wait_until="domcontentloaded")

        # Optimized scrolling with time limit
        await scroll_to_bottom_optimized(page, max_scroll_time=8)
        
        # Wait for the image section to appear with shorter timeout
        try:
            await page.wait_for_selector('div[data-id="mosaic"]', timeout=5000)
        except:
            # If mosaic selector fails, try alternative
            await page.wait_for_timeout(2000)

        # Find all image elements on the page using original working selectors
        image_elements = await page.query_selector_all('div[data-attrid="images universal"]')
        if not image_elements:
            # Fallback to more general selectors
            image_elements = await page.query_selector_all('div[jscontroller] img')
        
        logging.info(f"Found {len(image_elements)} image elements on the page.")

        images_found = 0
        image_data_list = []

        # Process images with optimized approach
        for idx, image_element in enumerate(image_elements[:max_images * 2]):  # Limit elements to process
            if images_found >= max_images:
                break
            try:
                # Click on the image to get a full view with timeout
                await image_element.click(timeout=2000)
                
                # Wait for the full-size image with shorter timeout
                try:
                    await page.wait_for_selector("img.sFlh5c.FyHeAf.iPVvYb[jsaction]", timeout=3000)
                    img_tag = await page.query_selector("img.sFlh5c.FyHeAf.iPVvYb[jsaction]")
                except:
                    # Fallback selector
                    img_tag = await page.query_selector("img[jsaction]")
                
                if not img_tag:
                    continue

                # Get the image URL
                img_url = await img_tag.get_attribute("src")
                if img_url and not img_url.startswith('data:'):
                    image_data_list.append(img_url)
                    images_found += 1
                    
            except Exception as e:
                logging.debug(f"Error processing image {idx + 1}: {e}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for url in image_data_list:
            if url not in seen:
                seen.add(url)
                unique_images.append(url)
        
        result = unique_images[:max_images]
        
        # Cache the result
        async with _cache_lock:
            _image_cache[cache_key] = (time.time(), result)
        
        return result
        
    finally:
        await page.close()


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


# setup AI agent with optimized settings for faster responses
model = AzureChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    api_version=os.getenv("AZURE_API_VERSION", "2024-10-01-preview"),
    azure_ad_token_provider=get_ailab_bearer_token_provider(),
    azure_endpoint=os.getenv(
        "AZURE_ENDPOINT", "https://ct-enterprisechat-api.azure-api.net/"
    ),
    temperature=0.1,  # Lower temperature for more consistent, faster responses
    max_tokens=16000,  # Doubled to handle very large image lists
    timeout=60,  # Increased timeout for larger responses
    max_retries=2,  # Reduce retries for faster failure handling
)
tools = [get_main_image_array, get_images_from_google, get_subject_from_image]

# Create parsers for different outputs
main_image_parser = PydanticOutputParser(pydantic_object=ImageArray)
google_image_parser = PydanticOutputParser(pydantic_object=GoogleImageArray)
subject_parser = PydanticOutputParser(pydantic_object=ImageSubject)


template = """
You are an AI agent that retrieves images and identifies their subjects.

CRITICAL INSTRUCTIONS FOR TOOL USAGE:
- Use ONLY ONE tool per request - do not chain multiple tools together
- Do NOT analyze or get subjects of images unless explicitly asked to do so
- When retrieving Google images, ONLY return the image URLs - do not analyze them
- When retrieving main images, ONLY return the image arrays - do not analyze them
- Only use get_subject_from_image when explicitly asked to identify a subject

CRITICAL INSTRUCTIONS FOR JSON OUTPUT:
- You MUST respond with ONLY valid, complete JSON format
- Do NOT include any explanatory text, comments, or additional information before or after the JSON
- Do NOT truncate or abbreviate the JSON output - ensure it is COMPLETE
- Ensure all JSON brackets, quotes, and commas are properly closed
- Your response must be parseable by a JSON parser
- If the output is large, prioritize completeness over brevity
- NEVER cut off JSON mid-way through an array or object

If the user asks for "main images" or "images from free-images.com":
- Use ONLY the `get_main_image_array` tool
- Do NOT use any other tools
- Format the output as valid JSON following this exact structure:
{main_image_format_instructions}

If the user asks for images from "Google" or "Google Images" for a specific subject:
- Use ONLY the `get_images_from_google` tool
- Do NOT use get_subject_from_image or any other tools
- The `get_images_from_google` tool allows for two parameters: `subject` and `max_images`
- The `subject` parameter is required, and the `max_images` parameter is optional
- You must use the exact subject provided as the `subject` parameter
- If the user provides a max number of images: you must use that number as the `max_images` parameter
- Format the output as valid JSON following this exact structure:
{google_image_format_instructions}

If the user provides an image URL and asks for its subject:
- Use ONLY the `get_subject_from_image` tool
- Do NOT use any other tools
- Format the output as valid JSON following this exact structure:
{subject_format_instructions}

CRITICAL: Your final response must be ONLY valid, complete JSON with no additional text. Use only the requested tool and do not make additional tool calls.
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


# Generated docs endpoint
@app.get("/")
async def get_docs():
    return RedirectResponse("/docs")

@app.get("/images/main")
async def get_main_array() -> Any:
    result = await agent_executor.ainvoke(
        {"input": "Get the main array of images. Return ONLY valid JSON format with no additional text."}
    )
    return parse_llm_response(result["output"], main_image_parser, "images")


@app.get("/images/subject")
async def get_subject(image_url: str) -> Any:
    """Get the subject of an image."""
    # Fully qualify the image url
    if not image_url.startswith("http"):
        image_url = f"{IMG_URL_ROOT}{image_url}"
        
    result = await agent_executor.ainvoke(
        {"input": f"What is the subject of this image? {image_url}. Return ONLY valid JSON format with no additional text."}
    )
    return parse_llm_response(result["output"], subject_parser)


@app.get("/images/google")
async def get_google_images(subject: str, max_images: int = 10) -> Any:
    """Get images from Google Images for a given subject."""
    result = await agent_executor.ainvoke(
        {"input": f"Get images from Google for the subject: {subject}. Return a maximum of {max_images} images. Return ONLY valid JSON format with no additional text."}
    )
    return parse_llm_response(result["output"], google_image_parser, "images")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
