# llmcord.py - v1.3.1
from base64 import b64encode, b64decode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional
import io
import os
import uuid
import asyncio
from collections import deque

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml
import json
import bs4
import urllib.parse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("", "qwen3", "claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl", "ollama")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
EMBED_COLOR_ERROR = discord.Color.red()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

# Image generation queue and storage
image_queue = asyncio.Queue()
image_results = {}
IMAGE_STORAGE_FOLDER = "generated_images"
os.makedirs(IMAGE_STORAGE_FOLDER, exist_ok=True)

def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()

# Fix: Provide default values if keys are missing
if "models" not in config:
    # Create models structure from llm.model if it exists
    llm_model = config.get("llm", {}).get("model")
    if llm_model:
        config["models"] = {llm_model: {}}
    else:
        config["models"] = {}

# Create a proper permissions structure based on your existing config
if "permissions" not in config:
    config["permissions"] = {
        "users": {
            "admin_ids": config.get("admin_user_ids", []),
            "allowed_ids": [],
            "blocked_ids": []
        },
        "roles": {
            "allowed_ids": [],
            "blocked_ids": []
        },
        "channels": {
            "allowed_ids": config.get("allowed_channel_ids", []),
            "blocked_ids": []
        }
    }

# Track current provider and model - initialize properly at module level
current_provider = None
current_model = None
current_image_provider = "forge_1"  # Default image provider

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()

# Web search function - UPDATED VERSION
async def web_search(query: str) -> dict:
    """Perform web search using SearXNG endpoint"""
    base_url = config["llm"]["web_search"]["search_url"]
    max_results = config["llm"]["web_search"].get("max_results", 5)
    max_images_per_result = config["llm"]["web_search"].get("max_images_per_result", 2)
    timeout = config["llm"]["web_search"].get("timeout", 30)
    
    # Try JSON API first
    try:
        params = {
            "q": query,
            "format": "json",
        }
        
        logging.info(f"Trying JSON API at {base_url}")
        response = await httpx_client.get(base_url, params=params, timeout=timeout)
        logging.info(f"JSON Response status: {response.status_code}")
        
        # Check if we got JSON
        content_type = response.headers.get('content-type', '')
        if 'application/json' in content_type:
            data = response.json()
            logging.info(f"Successfully parsed JSON response")
            
            # Extract basic results
            results = []
            for result in data.get("results", [])[:max_results]:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                    "images": []  # Will be populated later
                })
                
            if results:
                logging.info(f"Found {len(results)} results from JSON API, now fetching images...")
                
                # Try to get images for each result
                for i, result in enumerate(results):
                    try:
                        # Extract domain from URL to try finding images
                        parsed_url = urllib.parse.urlparse(result["url"])
                        domain = parsed_url.netloc
                        
                        # Try different image search strategies
                        images_found = False
                        
                        # Strategy 1: Look for images on the page itself
                        if not images_found and max_images_per_result > 0:
                            try:
                                page_response = await httpx_client.get(result["url"], timeout=10)
                                if page_response.status_code == 200:
                                    from bs4 import BeautifulSoup
                                    soup = BeautifulSoup(page_response.text, 'html.parser')
                                    
                                    # Common image selectors
                                    image_selectors = [
                                        'img[src*="jpg"]',
                                        'img[src*="jpeg"]', 
                                        'img[src*="png"]',
                                        'img[src*="gif"]',
                                        'img[src*="webp"]',
                                        '.content img',
                                        '.post img',
                                        '.article img',
                                        'img'
                                    ]
                                    
                                    for selector in image_selectors:
                                        images = soup.select(selector)[:max_images_per_result]
                                        if images:
                                            result["images"] = []
                                            for img in images:
                                                img_url = img.get('src', '')
                                                if img_url and not img_url.startswith(('http://', 'https://')):
                                                    # Convert relative URL to absolute
                                                    if img_url.startswith('//'):
                                                        img_url = f"https:{img_url}"
                                                    elif img_url.startswith('/'):
                                                        base_url = f"{parsed_url.scheme}://{domain}"
                                                        img_url = f"{base_url}{img_url}"
                                                    else:
                                                        base_path = "/".join(parsed_url.path.split('/')[:-1])
                                                        img_url = f"{parsed_url.scheme}://{domain}{base_path}/{img_url}"
                                                
                                                if img_url and (img_url.startswith(('http://', 'https://'))):
                                                    alt_text = img.get('alt', '')
                                                    result["images"].append({
                                                        "url": img_url,
                                                        "alt": alt_text
                                                    })
                                            
                                            if result["images"]:
                                                images_found = True
                                                logging.info(f"Found {len(result['images'])} images on page: {result['title']}")
                                                break
                            except Exception as e:
                                logging.debug(f"Failed to fetch images from {result['url']}: {e}")
                        
                        # Strategy 2: Use SearXNG's image search API if available
                        if not images_found and max_images_per_result > 0:
                            try:
                                # Try to use the same base URL but with different parameters
                                image_params = {
                                    "q": query,
                                    "format": "json",
                                    "engines": ["bing", "duckduckgo-images"],  # Image-focused engines
                                }
                                
                                image_response = await httpx_client.get(base_url, params=image_params, timeout=timeout)
                                if image_response.status_code == 200 and 'application/json' in image_response.headers.get('content-type', ''): 
                                    image_data = image_response.json()
                                    if image_data.get("results"):
                                        result["images"] = []
                                        for img_result in image_data.get("results", [])[:max_images_per_result]:
                                            img_url = img_result.get("img_src", img_result.get("url", ""))
                                            if img_url and img_url.startswith(('http://', 'https://')):
                                                result["images"].append({
                                                    "url": img_url,
                                                    "alt": img_result.get("title", "")
                                                })
                                        
                                        if result["images"]:
                                            images_found = True
                                            logging.info(f"Found {len(result['images'])} images via image search API")
                                            break
                            except Exception as e:
                                logging.debug(f"Image search API failed: {e}")
                        
                        # If no images found, log it
                        if not images_found:
                            logging.debug(f"No images found for: {result['title']}")
                            
                    except Exception as e:
                        logging.warning(f"Error fetching images for result {i}: {e}")
                        continue
                
                return {"success": True, "results": results}
        
        # If not JSON, log what we got
        logging.warning(f"Got non-JSON response. Content-Type: {content_type}")
        logging.warning(f"Response preview: {response.text[:500]}")
        
    except Exception as e:
        logging.info(f"JSON API failed ({str(e)}), trying HTML parsing...")
    
    # Fallback to HTML parsing with image extraction
    try:
        from bs4 import BeautifulSoup
        
        # Make regular search request without format=json
        params = {
            "q": query,
        }
        
        logging.info(f"Trying HTML parsing at {base_url}")
        response = await httpx_client.get(base_url, params=params, timeout=timeout)
        logging.info(f"HTML Response status: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Save the HTML to a file for inspection
        with open('/tmp/searxng_response.html', 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        logging.info("Saved full HTML response to /tmp/searxng_response.html for debugging")
        
        results = []
        
        # SearXNG uses specific classes for search results
        result_selectors = [
            '.result',
            '.result-item', 
            '.search-result',
            '.engine_item'
        ]
        
        title_selectors = [
            '.title a',
            '.result h3 a',
            '.result-title a',
            '.result a',
            'h3 a'
        ]
        
        content_selectors = [
            '.content',
            '.result .description',
            '.result-content',
            '.snippet',
            '.result p'
        ]
        
        # Image selectors - look for images within each result
        image_selectors = [
            '.result img',
            '.result-item img',
            '.search-result img',
            '.engine_item img',
            '.thumbnail img',
            '.image img',
            'img',
            '.result picture img',
            '.result figure img',
            '.media img',
            '.result-image img',
            '.result-thumbnail img'
        ]
        
        # Find result containers
        result_containers = []
        for selector in result_selectors:
            found = soup.select(selector)
            if found:
                result_containers = found
                logging.info(f"Found {len(found)} results using selector: {selector}")
                break
                
        if not result_containers:
            logging.error("Could not find any search result containers")
            return {"success": False, "error": "No search results found"}
            
        # Extract results from containers
        total_images_found = 0
        for container in result_containers[:max_results]:
            # Get title
            title = ""
            url = ""
            for title_selector in title_selectors:
                title_elem = container.select_one(title_selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    break
                    
            # Get content
            content = ""
            for content_selector in content_selectors:
                content_elem = container.select_one(content_selector)
                if content_elem:
                    content = content_elem.get_text(strip=True)[:200] + "..." if len(content_elem.get_text(strip=True)) > 200 else content_elem.get_text(strip=True)
                    break
                    
            # Get images
            images = []
            for image_selector in image_selectors:
                image_elems = container.select(image_selector)
                if image_elems:
                    logging.debug(f"Found {len(image_elems)} images with selector: {image_selector}")
                    
                for img in image_elems[:max_images_per_result]:
                    img_url = img.get('src', '')
                    # Handle relative URLs
                    if img_url and not img_url.startswith(('http://', 'https://')):
                        parsed_base = urllib.parse.urlparse(base_url)
                        base = f"{parsed_base.scheme}://{parsed_base.netloc}"
                        img_url = f"{base}{img_url}" if img_url.startswith('/') else f"{base}/{img_url}"
                        
                    if img_url and img_url.startswith(('http://', 'https://')):
                        alt_text = img.get('alt', '')
                        images.append({
                            "url": img_url,
                            "alt": alt_text
                        })
                        
            total_images_found += len(images)
            
            if title and url:
                results.append({
                    "title": title,
                    "url": url,
                    "content": content,
                    "images": images
                })
                
        if results:
            logging.info(f"Successfully parsed {len(results)} results from HTML with {total_images_found} images")
            return {"success": True, "results": results}
        else:
            logging.error("Found containers but no results could be extracted")
            return {"success": False, "error": "Failed to extract search results from HTML"}
            
    except Exception as e:
        logging.exception(f"HTML parsing failed: {e}")
    
    # If both methods fail
    return {
        "success": False, 
        "error": "Web search endpoint is not accessible or does not return valid data."
    }

# Helper function to format search results
def format_search_results(search_data: dict) -> str:
    """Format web search results into a readable string."""
    if not search_data.get("success"):
        return f"Search failed: {search_data.get('error', 'Unknown error')}"
    
    results = search_data.get("results", [])
    if not results:
        return "No search results found."
    
    formatted = []
    for i, res in enumerate(results, 1):
        title = res.get("title", "Untitled")
        content = res.get("content", "No content available.")
        url = res.get("url", "")
        
        # Truncate content if too long
        if len(content) > 200:
            content = content[:200] + "..."
        
        # Format each result
        result_text = f"{i}. [{title}]({url})\n   {content}"
        
        # Add images if any
        images = res.get("images", [])
        if images:
            image_urls = [f"[{img.get('alt', f'Image {j+1}')}]( {img['url']} )" for j, img in enumerate(images)]
            result_text += "\n   🖼️ Images: " + ", ".join(image_urls)
        
        formatted.append(result_text)
    
    full_text = "Here are some search results:\n\n" + "\n\n".join(formatted)
    
    # Truncate the entire text if too long
    max_length = 2000
    if len(full_text) > max_length:
        full_text = full_text[:max_length-3] + "..."
    
    return full_text

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# Image generation worker task
async def image_generation_worker():
    """Process image generation requests from the queue"""
    while True:
        try:
            request_id, prompt, negative_prompt, provider_config, parameters = await image_queue.get()
            
            logging.info(f"Processing image generation request {request_id}")
            
            try:
                # Generate the image
                image_data = await generate_image(prompt, negative_prompt, provider_config, parameters)
                
                if image_data:
                    # Save image to file
                    filename = f"{uuid.uuid4()}.png"
                    filepath = os.path.join(IMAGE_STORAGE_FOLDER, filename)
                    
                    try:
                        # Handle base64 image data - remove the "data:image/png;base64," prefix if present
                        if "," in image_data:
                            _, image_data = image_data.split(",", 1)
                        
                        # Decode base64 image data
                        image_bytes = b64decode(image_data)
                        
                        # Save to file
                        with open(filepath, "wb") as f:
                            f.write(image_bytes)
                        
                        image_results[request_id] = {"status": "success", "filepath": filepath}
                        logging.info(f"Image saved successfully: {filepath}")
                    except Exception as e:
                        logging.exception(f"Error saving generated image: {e}")
                        image_results[request_id] = {"status": "error", "error": str(e)}
                else:
                    image_results[request_id] = {"status": "error", "error": "Failed to generate image"}
                    
            except Exception as e:
                logging.exception(f"Error processing image generation: {e}")
                image_results[request_id] = {"status": "error", "error": str(e)}
            
            finally:
                image_queue.task_done()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.exception(f"Unexpected error in image worker: {e}")


# Image generation function for local providers
async def generate_image(prompt: str, negative_prompt: str, provider_config: dict, parameters: dict = None) -> Optional[str]:
    """Generate an image using a local provider (like Stable Diffusion WebUI Forge)"""
    try:
        forge_url = provider_config.get("forge_url")
        if not forge_url:
            logging.error("Forge URL not configured for image generation")
            return None
            
        # Default parameters from config
        default_params = provider_config.get("default_params", {})
        default_model = provider_config.get("default_model", "novaMatureXL_v35")
        default_size = provider_config.get("default_size", "768x768")
        
        # Parse size
        width, height = map(int, default_size.split('x'))
        
        # Merge parameters with defaults
        final_params = default_params.copy()
        if parameters:
            final_params.update(parameters)
        
        # Apply overrides from function parameters
        if "negative_prompt" in final_params:
            final_params["negative_prompt"] = negative_prompt or final_params["negative_prompt"]
        else:
            final_params["negative_prompt"] = negative_prompt or ""
            
        # Set dimensions from parameters or defaults
        final_params["width"] = final_params.get("width", width)
        final_params["height"] = final_params.get("height", height)
        
        # Prepare the payload - corrected for WebUI Forge API
        payload = {
            "prompt": prompt,
            "negative_prompt": final_params["negative_prompt"],
            "steps": final_params.get("steps", 40),
            "cfg_scale": final_params.get("cfg_scale", 7.0),
            "sampler_name": final_params.get("sampler_name", "DPM++ 2M Karras"),
            "restore_faces": final_params.get("restore_faces", False),
            "width": final_params["width"],
            "height": final_params["height"],
            "override_settings": {
                "sd_model_checkpoint": default_model
            }
        }
        
        # Add any additional parameters from config
        for key, value in final_params.items():
            if key not in ["negative_prompt", "steps", "cfg_scale", "sampler_name", "restore_faces", "width", "height"]:
                payload[key] = value
        
        headers = {"Content-Type": "application/json"}
        
        logging.info(f"Sending image generation request to {forge_url}")
        logging.info(f"Payload: {payload}")
        
        async with httpx.AsyncClient(timeout=provider_config.get("timeout", 720.0)) as client:
            response = await client.post(
                f"{forge_url}/sdapi/v1/txt2img",
                json=payload,
                headers=headers
            )
            
            logging.info(f"Response status: {response.status_code}")
            logging.info(f"Response text (first 200 chars): {response.text[:200]}...")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logging.info(f"Response data keys: {list(data.keys())}")
                    if "images" in data and len(data["images"]) > 0:
                        # Return the first image (base64 encoded)
                        return data["images"][0]
                    else:
                        logging.error("No images returned from image generation")
                        logging.error(f"Full response: {data}")
                        return None
                except Exception as e:
                    logging.exception(f"Error parsing JSON response: {e}")
                    logging.error(f"Raw response: {response.text}")
                    return None
            else:
                logging.error(f"Image generation failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logging.exception(f"Error generating image: {e}")
        return None

@discord_bot.tree.command(name="diagnose_websearch", description="Diagnose web search configuration (Admin only)")
async def diagnose_websearch_command(interaction: discord.Interaction):
    """Diagnose the web search configuration"""
    
    # Check if user is admin
    config = await asyncio.to_thread(get_config)
    permissions = config.get("permissions", {
        "users": {
            "admin_ids": config.get("admin_user_ids", []),
            "allowed_ids": [],
            "blocked_ids": []
        }
    })
    
    user_is_admin = interaction.user.id in permissions["users"]["admin_ids"]
    
    if not user_is_admin:
        await interaction.response.send_message("Only administrators can use this command.", ephemeral=True)
        return
    
    # Test the web search
    result = await web_search("test query")
    
    embed = discord.Embed(
        title="Web Search Diagnosis",
        color=EMBED_COLOR_ERROR if not result["success"] else EMBED_COLOR_COMPLETE
    )
    
    if result["success"]:
        embed.description = f"✅ Web search is working!\nFound {len(result['results'])} results."
        embed.add_field(name="Sample Result", value=result['results'][0]['title'][:100] + "..." if result['results'] else "No results", inline=False)
    else:
        embed.description = f"❌ Web search failed: {result['error']}"
        
        # Add troubleshooting tips
        embed.add_field(
            name="Troubleshooting Tips",
            value=(
                "1. Verify the endpoint URL points to an API, not a web interface\n"
                "2. Try adding '/api' or '/search' to the base URL\n"
                "3. Ensure the server has API enabled\n"
                "4. Check if authentication is required"
            ),
            inline=False
        )
    
    await interaction.response.send_message(embed=embed, ephemeral=True)

@discord_bot.tree.command(name="test_websearch", description="Test the web search functionality (Admin only)")
async def test_websearch_command(interaction: discord.Interaction):
    """Test the web search functionality"""
    
    # Check if user is admin
    config = await asyncio.to_thread(get_config)
    permissions = config.get("permissions", {
        "users": {
            "admin_ids": config.get("admin_user_ids", []),
            "allowed_ids": [],
            "blocked_ids": []
        }
    })
    
    user_is_admin = interaction.user.id in permissions["users"]["admin_ids"]
    
    if not user_is_admin:
        await interaction.response.send_message("Only administrators can use this command.", ephemeral=True)
        return
    
    # Test the web search
    result = await web_search("test query")
    
    if result["success"]:
        await interaction.response.send_message(
            f"Web search test successful! Found {len(result['results'])} results.",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            f"Web search test failed: {result['error']}",
            ephemeral=True
        )

# Image generation command that uses queue system
@discord_bot.tree.command(name="image", description="Generate an image from a prompt")
async def image_command(interaction: discord.Interaction, 
                       prompt: str,
                       negative_prompt: Optional[str] = None) -> None:
    """Generate an image based on a text prompt"""
    
    # Defer response to show we're working
    await interaction.response.defer(ephemeral=False)
    
    # Check if image generation is configured
    llm_config = config.get("llm", {})
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        await interaction.followup.send("Image generation is not configured.", ephemeral=True)
        return
    
    # Get current image provider configuration
    provider_config = image_providers.get(current_image_provider, {})
    if not provider_config:
        await interaction.followup.send(f"Image provider '{current_image_provider}' not found.", ephemeral=True)
        return
    
    # Generate request ID and add to queue
    request_id = str(uuid.uuid4())
    await image_queue.put((request_id, prompt, negative_prompt or "", provider_config, {}))
    
    # Wait for result with timeout (20 minutes)
    start_time = datetime.now()
    timeout = 1200  # 20 minutes in seconds
    
    while (datetime.now() - start_time).seconds < timeout:
        if request_id in image_results:
            result = image_results.pop(request_id)
            
            if result["status"] == "success":
                try:
                    # Send the image file
                    filepath = result["filepath"]
                    file = discord.File(filepath, filename=os.path.basename(filepath))
                    
                    # Create embed
                    embed = discord.Embed(title=f"Generated Image for: {prompt[:50]}...", color=EMBED_COLOR_COMPLETE)
                    if negative_prompt:
                        embed.description = f"Negative prompt: {negative_prompt[:50]}..."
                    
                    await interaction.followup.send(embed=embed, file=file)
                    
                    # Clean up file after sending
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logging.warning(f"Could not delete temporary image file {filepath}: {e}")
                        
                except Exception as e:
                    logging.exception(f"Error sending generated image: {e}")
                    await interaction.followup.send("Failed to send generated image.", ephemeral=True)
            else:
                error_msg = result.get("error", "Unknown error")
                await interaction.followup.send(f"Failed to generate image: {error_msg}", ephemeral=True)
            
            return
        
        await asyncio.sleep(5)  # Check every 5 seconds
    
    # Timeout reached
    await interaction.followup.send("Image generation timed out after 20 minutes.", ephemeral=True)


# Alternative approach: Use a simple button-based workflow instead of modal
@discord_bot.tree.command(name="image_advanced", description="Generate an image with advanced parameters")
async def image_advanced_command(interaction: discord.Interaction, 
                                prompt: str,
                                negative_prompt: Optional[str] = None,
                                steps: Optional[int] = None,
                                cfg_scale: Optional[float] = None,
                                width: Optional[int] = None,
                                height: Optional[int] = None) -> None:
    """Generate an image with advanced options"""
    
    # Defer response to show we're working
    await interaction.response.defer(ephemeral=False)
    
    # Check if image generation is configured
    llm_config = config.get("llm", {})
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        await interaction.followup.send("Image generation is not configured.", ephemeral=True)
        return
    
    # Get current image provider configuration
    provider_config = image_providers.get(current_image_provider, {})
    if not provider_config:
        await interaction.followup.send(f"Image provider '{current_image_provider}' not found.", ephemeral=True)
        return
    
    # Build parameters
    parameters = {}
    if steps is not None:
        parameters["steps"] = steps
    if cfg_scale is not None:
        parameters["cfg_scale"] = cfg_scale
    if width is not None:
        parameters["width"] = width
    if height is not None:
        parameters["height"] = height
    
    # Generate request ID and add to queue
    request_id = str(uuid.uuid4())
    await image_queue.put((request_id, prompt, negative_prompt or "", provider_config, parameters))
    
    # Wait for result with timeout (20 minutes)
    start_time = datetime.now()
    timeout = 1200  # 20 minutes in seconds
    
    while (datetime.now() - start_time).seconds < timeout:
        if request_id in image_results:
            result = image_results.pop(request_id)
            
            if result["status"] == "success":
                try:
                    # Send the image file
                    filepath = result["filepath"]
                    file = discord.File(filepath, filename=os.path.basename(filepath))
                    
                    # Create embed
                    embed = discord.Embed(title=f"Generated Image for: {prompt[:50]}...", color=EMBED_COLOR_COMPLETE)
                    if negative_prompt:
                        embed.description = f"Negative prompt: {negative_prompt[:50]}..."
                    
                    await interaction.followup.send(embed=embed, file=file)
                    
                    # Clean up file after sending
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logging.warning(f"Could not delete temporary image file {filepath}: {e}")
                        
                except Exception as e:
                    logging.exception(f"Error sending generated image: {e}")
                    await interaction.followup.send("Failed to send generated image.", ephemeral=True)
            else:
                error_msg = result.get("error", "Unknown error")
                await interaction.followup.send(f"Failed to generate image: {error_msg}", ephemeral=True)
            
            return
        
        await asyncio.sleep(5)  # Check every 5 seconds
    
    # Timeout reached
    await interaction.followup.send("Image generation timed out after 20 minutes.", ephemeral=True)


# New /providers command to switch between different endpoints
@discord_bot.tree.command(name="providers", description="Switch between different providers/endpoints")
async def providers_command(interaction: discord.Interaction, provider: str) -> None:
    """Switch to a different provider"""
    global current_provider, current_model
    
    # Get available providers from config
    llm_config = config.get("llm", {})
    providers = llm_config.get("providers", {})
    
    if not providers:
        await interaction.response.send_message("No providers configured.", ephemeral=True)
        return
        
    if provider not in providers:
        available_providers = list(providers.keys())
        await interaction.response.send_message(
            f"Provider '{provider}' not found. Available providers: {', '.join(available_providers)}", 
            ephemeral=True
        )
        return
    
    # Set the new provider and model
    current_provider = provider
    provider_config = providers[provider]
    
    # Try to get the default model for this provider
    # For Ollama, we'll use the main model from config or set a default
    default_model = provider_config.get("model") or "default"
    
    # If no specific model is set, try to find one from the models section
    if default_model == "default":
        # Check if there's a model in the main llm section that matches this provider
        llm_model = config.get("llm", {}).get("model", "")
        if llm_model and provider in llm_model:
            default_model = llm_model.split("/", 1)[1] if "/" in llm_model else llm_model
        else:
            # Fallback to a reasonable default for ollama - let's make it more flexible
            # We'll just use a generic name since we don't know the actual model names
            default_model = "qwen3"  # This will be overridden by the actual model name
    
    current_model = default_model
    output = f"Switched to provider: `{provider}` with model: `{default_model}`"
        
    logging.info(output)
    await interaction.response.send_message(output, ephemeral=True)


# New /image_providers command to switch between different image generation endpoints
@discord_bot.tree.command(name="image_providers", description="Switch between different image generation providers")
async def image_providers_command(interaction: discord.Interaction, provider: str) -> None:
    """Switch to a different image generation provider"""
    global current_image_provider
    
    # Get available image providers from config
    llm_config = config.get("llm", {})
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        await interaction.response.send_message("No image providers configured.", ephemeral=True)
        return
        
    # Filter out non-provider keys (like forge_url, default_model, etc.)
    actual_providers = {k: v for k, v in image_providers.items() if isinstance(v, dict) and 'forge_url' in v}
    
    if not actual_providers:
        await interaction.response.send_message("No valid image providers configured.", ephemeral=True)
        return
    
    if provider not in actual_providers:
        available_providers = list(actual_providers.keys())
        await interaction.response.send_message(
            f"Image provider '{provider}' not found. Available providers: {', '.join(available_providers)}", 
            ephemeral=True
        )
        return
    
    # Set the new image provider
    current_image_provider = provider
    output = f"Switched to image provider: `{provider}`"
        
    logging.info(output)
    await interaction.response.send_message(output, ephemeral=True)


# New /tools command to toggle active tools
@discord_bot.tree.command(name="tools", description="Toggle active tools")
async def tools_command(interaction: discord.Interaction, tool: str, enabled: bool):
    """Enable/disable specific tools"""
    global config
    
    # Get current config
    current_config = await asyncio.to_thread(get_config)
    llm_config = current_config.get("llm", {})
    active_tools = llm_config.get("active_tools", [])
    
    # Define available tools
    available_tools = ["image_generator", "web_search"]
    
    if tool not in available_tools:
        await interaction.response.send_message(
            f"Tool '{tool}' not found. Available tools: {', '.join(available_tools)}", 
            ephemeral=True
        )
        return
    
    # Update the active_tools list
    if enabled and tool not in active_tools:
        active_tools.append(tool)
    elif not enabled and tool in active_tools:
        active_tools.remove(tool)
    
    # Update the config file
    try:
        with open("config.yaml", "w") as f:
            yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
        
        status = "enabled" if enabled else "disabled"
        message = f"Tool '{tool}' has been {status}."
        logging.info(message)
        await interaction.response.send_message(message, ephemeral=True)
        
    except Exception as e:
        error_msg = f"Failed to update tool settings: {e}"
        logging.error(error_msg)
        await interaction.response.send_message(error_msg, ephemeral=True)

@tools_command.autocomplete("tool")
async def tools_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    """Autocomplete for tool names"""
    available_tools = ["image_generator", "web_search"]
    filtered_tools = [t for t in available_tools if curr_str.lower() in t.lower()]
    
    return [Choice(name=t, value=t) for t in filtered_tools[:25]]


# New /allow_dm command for admins to toggle DM functionality
@discord_bot.tree.command(name="allow_dm", description="Toggle Direct Message functionality (Admin only)")
async def allow_dm_command(interaction: discord.Interaction, enabled: bool) -> None:
    """Toggle whether the bot accepts Direct Messages"""
    # Check if user is admin
    config = await asyncio.to_thread(get_config)
    permissions = config.get("permissions", {
        "users": {
            "admin_ids": config.get("admin_user_ids", []),
            "allowed_ids": [],
            "blocked_ids": []
        },
        "roles": {
            "allowed_ids": [],
            "blocked_ids": []
        },
        "channels": {
            "allowed_ids": config.get("allowed_channel_ids", []),
            "blocked_ids": []
        }
    })
    
    user_is_admin = interaction.user.id in permissions["users"]["admin_ids"]
    
    if not user_is_admin:
        await interaction.response.send_message("Only administrators can use this command.", ephemeral=True)
        return
    
    # Update the config file
    try:
        # Read current config
        with open("config.yaml", "r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f) or {}
        
        # Update allow_dms setting
        current_config["allow_dms"] = enabled
        
        # Write back to file
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
        
        status = "enabled" if enabled else "disabled"
        message = f"Direct Messages have been {status}."
        logging.info(message)
        await interaction.response.send_message(message, ephemeral=True)
        
    except Exception as e:
        error_msg = f"Failed to update DM settings: {e}"
        logging.error(error_msg)
        await interaction.response.send_message(error_msg, ephemeral=True)


@providers_command.autocomplete("provider")
async def provider_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    """Autocomplete for provider names"""
    llm_config = config.get("llm", {})
    providers = llm_config.get("providers", {})
    
    if not providers:
        return []
        
    # Filter providers based on search string
    filtered_providers = [p for p in providers.keys() if curr_str.lower() in p.lower()]
    
    # Return up to 25 choices
    choices = [Choice(name=p, value=p) for p in filtered_providers[:25]]
    return choices


@image_providers_command.autocomplete("provider")
async def image_provider_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    """Autocomplete for image provider names"""
    llm_config = config.get("llm", {})
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        return []
    
    # Filter out non-provider keys (like forge_url, default_model, etc.)
    actual_providers = [k for k, v in image_providers.items() if isinstance(v, dict) and 'forge_url' in v]
    
    if not actual_providers:
        return []
        
    # Filter providers based on search string
    filtered_providers = [p for p in actual_providers if curr_str.lower() in p.lower()]
    
    # Return up to 25 choices
    choices = [Choice(name=p, value=p) for p in filtered_providers[:25]]
    return choices


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    # Sync commands when bot is ready
    try:
        synced = await discord_bot.tree.sync()
        logging.info(f"Synced {len(synced)} command(s)")
        for cmd in synced:
            logging.info(f"  - {cmd.name}")
    except Exception as e:
        logging.error(f"Failed to sync commands: {e}")
    
    # Start image generation worker
    discord_bot.loop.create_task(image_generation_worker())


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time, current_provider, current_model

    is_dm = new_msg.channel.type == discord.ChannelType.private

    # Skip if it's a bot message
    if new_msg.author.bot:
        return

    # The key fix: Allow processing of all non-empty user messages
    should_process = False
    
    # Only process if:
    #   - It's a direct message (DM), OR
    #   - The bot is mentioned in the message
    should_process = is_dm or discord_bot.user.mention in new_msg.content

    if not should_process:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    # Fix: Safely access permissions with fallback
    permissions = config.get("permissions", {
        "users": {
            "admin_ids": config.get("admin_user_ids", []),
            "allowed_ids": [],
            "blocked_ids": []
        },
        "roles": {
            "allowed_ids": [],
            "blocked_ids": []
        },
        "channels": {
            "allowed_ids": config.get("allowed_channel_ids", []),
            "blocked_ids": []
        }
    })

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    # NEW: Check if DMs are disabled when this is a DM message
    if is_dm and not allow_dms:
        return  # Silently ignore DMs if they're disabled

    if is_bad_user or is_bad_channel:
        return

    # Use the current provider/model setup - FIXED: Properly handle global variables
    llm_config = config.get("llm", {})
    
    # If no provider set yet, try to get from config
    if current_provider is None:
        providers = llm_config.get("providers", {})
        if providers:
            # Get first available provider
            current_provider = list(providers.keys())[0]
            provider_config = providers[current_provider]
            
            # Try to get model from main llm config
            default_model = config.get("llm", {}).get("model")
            if default_model:
                # Extract model name if it's in format "provider/model"
                if "/" in default_model:
                    current_model = default_model.split("/", 1)[1]
                else:
                    current_model = default_model
            else:
                current_model = "qwen3"  # Default fallback
        else:
            logging.error("No providers configured!")
            return
    
    # Get provider configuration
    provider_config = llm_config.get("providers", {}).get(current_provider, {})
    
    if not provider_config:
        logging.error(f"Provider configuration not found for: {current_provider}")
        return

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # Fix: Handle models properly - use the llm.model value if needed
    model_parameters = {}
    # Only access models if they exist and are properly structured
    if "models" in config and current_model:
        model_key = f"{current_provider}/{current_model}"
        model_parameters = config["models"].get(model_key, {}) or {}

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in current_model.lower() for x in VISION_MODEL_TAGS) if current_model else False
    accept_usernames = any(current_provider.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = new_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in new_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in new_msg.embeds]
                    + [component.content for component in new_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]
                
                curr_node.role = "assistant" if new_msg.author == discord_bot.user else "user"

                curr_node.user_id = new_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(new_msg.attachments) > len(good_attachments)

                try:
                    if (
                        new_msg.reference == None
                        and discord_bot.user.mention not in new_msg.content
                        and (prev_msg_in_channel := ([m async for m in new_msg.channel.history(before=new_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if new_msg.channel.type == discord.ChannelType.private else new_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = new_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and new_msg.reference == None and new_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := new_msg.channel.id if parent_is_thread_start else getattr(new_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = new_msg.channel.starter_message or await new_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = new_msg.reference.cached_message or await new_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    # Web search integration - UPDATED VERSION
    if "web_search" in llm_config.get("active_tools", []):
        # Get trigger words from config with fallback defaults
        web_search_config = llm_config.get("web_search", {})
        trigger_words = web_search_config.get("trigger_words", [
            "search", "find", "google", "look up", "who is", 
            "find me", "double check", "check again", "what is", "doublecheck"
        ])
        
        if any(word in new_msg.content.lower() for word in trigger_words):
            logging.info(f"Trigger word found in message: {new_msg.content}")
            search_data = await web_search(new_msg.content)
            
            if search_data["success"] and search_data["results"]:
                formatted_search = format_search_results(search_data)
                
                # Insert as a new message in the conversation history
                search_message = {
                    "role": "user",
                    "content": formatted_search
                }
                messages.insert(1, search_message)  # After the original user message
                
                # Optionally send as separate message for verification
                try:
                    await new_msg.channel.send(
                        content=formatted_search,
                        suppress_embeds=True
                    )
                    logging.info("Sent search results as separate Discord message for verification")
                except Exception as e:
                    logging.warning(f"Failed to send search results as separate message: {e}")
            else:
                logging.warning(f"Web search failed: {search_data.get('error', 'Unknown error')}")

    # Add system prompt at the beginning of messages - FIXED VERSION
    system_prompt = llm_config.get("system_prompt") or ""
    
    if system_prompt:
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        
        if accept_usernames:
            system_prompt += "\n\nUser's names are their Discord IDs and should be typed as '<@ID>'."

        # IMPORTANT: Append system prompt to the end of the reverse-chronological list
        messages.append(dict(role="system", content=system_prompt))
    else:
        logging.info("No system prompt found in config")

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    # Use current model or fallback to first available
    model_to_use = current_model or "qwen3"
    
    openai_kwargs = dict(model=model_to_use, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception as e:
        logging.exception(f"Error while generating response: {e}")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
