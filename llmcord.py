# llmcord.py (with reasoning display options and vision capabilities preserved)
import os
import uuid
import logging
import asyncio
import httpx
import yaml
import urllib.parse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any, Literal, List
from base64 import b64decode, b64encode
import discord
from discord.app_commands import Choice as AppChoice
from discord.ext import commands
from openai import AsyncOpenAI
from collections import OrderedDict
import re

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
EMBED_COLOR_INCOMPLETE = 0x3498db
EMBED_COLOR_COMPLETE = 0x2ecc71
EMBED_COLOR_ERROR = 0xe74c3c
STREAMING_INDICATOR = "..."
EDIT_DELAY_SECONDS = 0.5
MAX_MESSAGE_NODES = 100
VISION_MODEL_TAGS = ["", "vision", "gpt-4", "claude"]
PROVIDERS_SUPPORTING_USERNAMES = ["", "openai", "anthropic", "gemini"]

# Global variables
image_queue = asyncio.Queue()
image_results = {}
image_queue_positions = {}
queue_lock = asyncio.Lock()
IMAGE_STORAGE_FOLDER = "generated_images"
os.makedirs(IMAGE_STORAGE_FOLDER, exist_ok=True)

# Config handling
def get_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

config = get_config()
if "llm" not in config:
    config["llm"] = {}
if "models" not in config:
    llm_model = config.get("llm", {}).get("model")
    if llm_model:
        config["models"] = {llm_model: {}}
    else:
        config["models"] = {}

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

current_provider = "Kobold_Server"
current_model = None
current_image_provider = "forge_1"
msg_nodes = {}
last_task_time = 0

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "Original: github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)
httpx_client = httpx.AsyncClient()

# Reasoning processing functions
def extract_reasoning(content: str, start_tag: str, end_tag: str) -> tuple[str, Optional[str]]:
    """Extract reasoning section from content using configured tags"""
    reasoning_start = content.find(start_tag)
    if reasoning_start == -1:
        return content, None
    
    reasoning_end = content.find(end_tag, reasoning_start)
    if reasoning_end == -1:
        return content, None
    
    # Extract the reasoning content (without the tags)
    reasoning_content = content[reasoning_start + len(start_tag):reasoning_end].strip()
    
    # Create clean content without reasoning section
    clean_content = (
        content[:reasoning_start].rstrip() + 
        content[reasoning_end + len(end_tag):].lstrip()
    )
    
    return clean_content, reasoning_content

def process_reasoning_content(
    content: str, 
    reasoning_format: str,
    start_tag: str,
    end_tag: str
) -> tuple[str, Optional[str], bool]:
    """
    Process content to handle reasoning sections based on configuration
    Returns: (processed_content, reasoning_content, has_reasoning)
    """
    clean_content, reasoning_content = extract_reasoning(content, start_tag, end_tag)
    
    if not reasoning_content:
        return content, None, False
    
    if reasoning_format == "spoiler":
        # Replace with spoiler format
        spoiler_content = f"> ||{reasoning_content}||\n\n"
        return clean_content, reasoning_content, True
    elif reasoning_format == "separate":
        # Just remove the reasoning section but keep it for separate message
        return clean_content, reasoning_content, True
    else:  # "none" - completely remove reasoning
        return clean_content, None, True

# Web search functions
async def web_search(query: str) -> dict:
    base_url = config["llm"]["web_search"]["search_url"]
    max_results = config["llm"]["web_search"].get("max_results", 5)
    max_images_per_result = config["llm"]["web_search"].get("max_images_per_result", 2)
    timeout = config["llm"]["web_search"].get("timeout", 30)
    
    try:
        params = {"q": query, "format": "json"}
        response = await httpx_client.get(base_url, params=params, timeout=timeout)
        content_type = response.headers.get('content-type', '')
        
        if 'application/json' in content_type:
            try:
                data = response.json()
                results = []
                for result in data.get("results", [])[:max_results]:
                    basic_result = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", "")[:200] + "..." if len(result.get("content", "")) > 200 else result.get("content", ""),
                        "images": [],
                        "source": result.get("source", ""),
                        "category": result.get("category", ""),
                        "published_date": result.get("published_date", "")
                    }
                    enhanced_content = await extract_enhanced_content(basic_result["url"], basic_result["content"])
                    basic_result["enhanced_content"] = enhanced_content or {}
                    results.append(basic_result)
                
                for i, result in enumerate(results):
                    try:
                        images = await extract_images_from_result(result["url"], max_images_per_result)
                        result["images"] = images
                        parsed_url = urllib.parse.urlparse(result["url"])
                        result["domain"] = parsed_url.netloc
                    except Exception as e:
                        logger.warning(f"Error extracting images for result {i}: {e}")
                        continue
                
                return {"success": True, "results": results}
            except Exception as e:
                logger.warning(f"JSON parsing failed: {e}")
                pass
        
        logger.warning(f"Got non-JSON response. Content-Type: {content_type}")
        logger.warning(f"Response preview: {response.text[:500]}")
        
    except Exception as e:
        logger.info(f"JSON API failed ({str(e)}), trying HTML parsing...")
    
    # HTML parsing fallback
    try:
        from bs4 import BeautifulSoup
        params = {"q": query}
        response = await httpx_client.get(base_url, params=params, timeout=timeout)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        with open('/tmp/searxng_response.html', 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        
        results = []
        result_selectors = ['.result', '.result-item', '.search-result', '.engine_item', '.result-group']
        title_selectors = ['.title a', '.result h3 a', '.result-title a', '.result a', 'h3 a', '.result-title', '.title']
        content_selectors = ['.content', '.result .description', '.result-content', '.snippet', '.result p', '.result-excerpt', '.result-text']
        image_selectors = ['.result img', '.result-item img', '.search-result img', '.engine_item img', '.thumbnail img', '.image img', 'img', '.result picture img', '.result figure img', '.media img', '.result-image img', '.result-thumbnail img', '.result-image-container img']
        
        result_containers = []
        for selector in result_selectors:
            found = soup.select(selector)
            if found:
                result_containers = found
                logger.info(f"Found {len(found)} results using selector: {selector}")
                break
                
        if not result_containers:
            logger.error("Could not find any search result containers")
            return {"success": False, "error": "No search results found"}
            
        total_images_found = 0
        for container in result_containers[:max_results]:
            title = ""
            url = ""
            for title_selector in title_selectors:
                title_elem = container.select_one(title_selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    if url and not url.startswith(('http://', 'https://')):
                        parsed_base = urllib.parse.urlparse(base_url)
                        base = f"{parsed_base.scheme}://{parsed_base.netloc}"
                        url = f"{base}{url}" if url.startswith('/') else f"{base}/{url}"
                    break
                    
            content = ""
            for content_selector in content_selectors:
                content_elem = container.select_one(content_selector)
                if content_elem:
                    content = content_elem.get_text(strip=True)[:200] + "..." if len(content_elem.get_text(strip=True)) > 200 else content_elem.get_text(strip=True)
                    break
                    
            source = ""
            category = ""
            published_date = ""
            
            source_selectors = ['.source', '.site', '.domain', '.meta-source']
            for sel in source_selectors:
                source_elem = container.select_one(sel)
                if source_elem:
                    source = source_elem.get_text(strip=True)
                    break
                    
            category_selectors = ['.category', '.type', '.result-category']
            for sel in category_selectors:
                cat_elem = container.select_one(sel)
                if cat_elem:
                    category = cat_elem.get_text(strip=True)
                    break
                    
            date_selectors = ['.date', '.published', '.timestamp', '.time']
            for sel in date_selectors:
                date_elem = container.select_one(sel)
                if date_elem:
                    published_date = date_elem.get_text(strip=True)
                    break
            
            enhanced_content = await extract_enhanced_content(url, content)
            
            images = []
            for image_selector in image_selectors:
                image_elems = container.select(image_selector)
                if image_elems:
                    logger.debug(f"Found {len(image_elems)} images with selector: {image_selector}")
                    break
            
            if not images:
                image_links = container.select('a[href*=".jpg"], a[href*=".jpeg"], a[href*=".png"], a[href*=".gif"]')
                for link in image_links[:max_images_per_result]:
                    img_url = link.get('href', '')
                    if img_url and img_url.startswith(('http://', 'https://')):
                        images.append({
                            "url": img_url,
                            "alt": link.get('title', link.get('alt', ''))
                        })
            
            total_images_found += len(images)
            
            if title and url:
                result = {
                    "title": title,
                    "url": url,
                    "content": content,
                    "enhanced_content": enhanced_content or {},
                    "images": images,
                    "source": source,
                    "category": category,
                    "published_date": published_date,
                    "domain": urllib.parse.urlparse(url).netloc if url else ""
                }
                results.append(result)
                
        if results:
            logger.info(f"Successfully parsed {len(results)} results from HTML with {total_images_found} images")
            return {"success": True, "results": results}
        else:
            logger.error("Found containers but no results could be extracted")
            return {"success": False, "error": "Failed to extract search results from HTML"}
            
    except Exception as e:
        logger.exception(f"HTML parsing failed: {e}")
    
    return {
        "success": False, 
        "error": "Web search endpoint is not accessible or does not return valid data."
    }

async def extract_enhanced_content(url: str, fallback_content: str) -> dict:
    if not url:
        return {
            "summary": fallback_content,
            "key_points": [],
            "main_content": "",
            "entities": [],
            "keywords": []
        }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for script in soup(["script", "style"]):
                    script.decompose()
                
                main_content = ""
                content_selectors = ['main', '.content', '.article', '.post', '.entry-content', '.post-content', '.main-content', '.body', '.text']
                
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        main_content = content_elem.get_text(strip=True)
                        if len(main_content) > 100:
                            break
                
                if not main_content:
                    body = soup.find('body')
                    if body:
                        main_content = body.get_text(strip=True)
                
                key_points = []
                if main_content:
                    paragraphs = soup.find_all('p')
                    for i, p in enumerate(paragraphs[:3]):
                        text = p.get_text(strip=True)
                        if text and len(text) > 50:
                            key_points.append(text[:200] + "..." if len(text) > 200 else text)
                
                entities = extract_entities(main_content)
                
                keywords = []
                title_elem = soup.find('title')
                if title_elem:
                    title_words = title_elem.get_text().split()
                    keywords.extend([w for w in title_words if len(w) > 3 and not w.lower() in ['the', 'and', 'for', 'are', 'but', 'not']])
                
                meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
                if meta_keywords:
                    meta_words = meta_keywords.get('content', '').split(',')
                    keywords.extend([w.strip() for w in meta_words if len(w.strip()) > 3])
                
                keywords = list(dict.fromkeys(keywords))[:20]
                
                return {
                    "summary": fallback_content[:300] + "..." if len(fallback_content) > 300 else fallback_content,
                    "key_points": key_points,
                    "main_content": main_content[:1000] + "..." if len(main_content) > 1000 else main_content,
                    "entities": entities[:10],
                    "keywords": keywords
                }
                
    except Exception as e:
        logger.warning(f"Failed to extract enhanced content from {url}: {e}")
        return {
            "summary": fallback_content,
            "key_points": [],
            "main_content": "",
            "entities": [],
            "keywords": []
        }

def extract_entities(text: str) -> list:
    import re
    
    entities = []
    
    names = re.findall(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b', text)
    entities.extend(names)
    
    orgs = re.findall(r'\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\s+(?:Inc|Corp|LLC|Ltd|Co)\b', text)
    entities.extend(orgs)
    
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b', text)
    entities.extend(dates)
    
    urls = re.findall(r'https?://[^\s]+', text)
    entities.extend(urls)
    
    return list(dict.fromkeys(entities))

async def extract_images_from_result(url: str, max_images: int = 2) -> list:
    if not url:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for script in soup(["script", "style"]):
                    script.decompose()
                
                images = []
                img_selectors = [
                    'img[src*=".jpg"], img[src*=".jpeg"], img[src*=".png"], img[src*=".gif"]',
                    'img',
                    '.image img',
                    '.gallery img',
                    '.photo img',
                    '.thumbnail img'
                ]
                
                for selector in img_selectors:
                    img_elements = soup.select(selector)
                    if img_elements:
                        break
                
                if not img_elements:
                    img_elements = soup.find_all('img')
                
                for img in img_elements[:max_images]:
                    img_url = img.get('src', '')
                    alt_text = img.get('alt', '')
                    
                    if img_url and not img_url.startswith(('http://', 'https://')):
                        try:
                            parsed_url = urllib.parse.urlparse(url)
                            base = f"{parsed_url.scheme}://{parsed_url.netloc}"
                            if img_url.startswith('/'):
                                img_url = f"{base}{img_url}"
                            else:
                                base_path = '/'.join(parsed_url.path.split('/')[:-1])
                                img_url = f"{base}{base_path}/{img_url}" if base_path else f"{base}/{img_url}"
                        except:
                            pass
                    
                    if img_url and img_url.startswith(('http://', 'https://')):
                        images.append({
                            "url": img_url,
                            "alt": alt_text[:100] if alt_text else "No description"
                        })
                
                return images[:max_images]
                
    except Exception as e:
        logger.warning(f"Failed to extract images from {url}: {e}")
        return []

def format_search_results(search_data: dict) -> str:
    if not search_data.get("success"):
        return f"Search failed: {search_data.get('error', 'Unknown error')}"
    
    results = search_data.get("results", [])
    if not results:
        return "No search results found."
    
    formatted = []
    for i, res in enumerate(results, 1):
        title = res.get("title", "Untitled")
        url = res.get("url", "")
        content = res.get("content", "No content available.")
        enhanced_content = res.get("enhanced_content", {}) or {}
        
        if not isinstance(enhanced_content, dict):
            enhanced_content = {}
        
        key_points = enhanced_content.get("key_points", [])
        entities = enhanced_content.get("entities", [])
        keywords = enhanced_content.get("keywords", [])
        
        if len(content) > 200:
            content = content[:200] + "..."
        
        result_text = f"{i}. [{title}]({url})\n"
        result_text += f"   {content}\n"
        
        if enhanced_content.get("main_content"):
            main_content_preview = enhanced_content["main_content"][:300] + "..." if len(enhanced_content["main_content"]) > 300 else enhanced_content["main_content"]
            result_text += f"   Main content preview: {main_content_preview}\n"
        
        if key_points:
            result_text += "   Key points:\n"
            for point in key_points[:2]:
                result_text += f"     • {point}\n"
        
        if entities:
            entities_preview = ", ".join(entities[:3])
            result_text += f"   Entities: {entities_preview}\n"
        
        if keywords:
            keywords_preview = ", ".join(keywords[:5])
            result_text += f"   Keywords: {keywords_preview}\n"
        
        images = res.get("images", [])
        if images:
            image_urls = [f"[Image {j+1}]({img['url']})" for j, img in enumerate(images)]
            result_text += f"   🖼️ Images: {', '.join(image_urls)}\n"
        
        formatted.append(result_text)
    
    full_text = "Here are some enhanced search results:\n\n" + "\n\n".join(formatted)
    
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

# Image generation functions
async def image_generation_worker():
    while True:
        try:
            request_id, prompt, negative_prompt, provider_config, parameters, user_id = await image_queue.get()
            
            async with queue_lock:
                queue_position = image_queue.qsize() + 1
                image_queue_positions[request_id] = queue_position
            
            try:
                image_data = await generate_image(prompt, negative_prompt, provider_config, parameters)
                
                if image_data:
                    filename = f"{uuid.uuid4()}.png"
                    filepath = os.path.join(IMAGE_STORAGE_FOLDER, filename)
                    
                    try:
                        if "," in image_data:
                            _, image_data = image_data.split(",", 1)
                        
                        image_bytes = b64decode(image_data)
                        
                        with open(filepath, "wb") as f:
                            f.write(image_bytes)
                        
                        image_results[request_id] = {"status": "success", "filepath": filepath}
                        logger.info(f"Image saved successfully: {filepath}")
                    except Exception as e:
                        logger.exception(f"Error saving generated image: {e}")
                        image_results[request_id] = {"status": "error", "error": str(e)}
                else:
                    image_results[request_id] = {"status": "error", "error": "Failed to generate image"}
                    
            except Exception as e:
                logger.exception(f"Error processing image generation: {e}")
                image_results[request_id] = {"status": "error", "error": str(e)}
            
            finally:
                async with queue_lock:
                    image_queue_positions.pop(request_id, None)
                image_queue.task_done()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(f"Unexpected error in image worker: {e}")

async def generate_image(prompt: str, negative_prompt: str, provider_config: dict, parameters: dict = None) -> Optional[str]:
    try:
        forge_url = provider_config.get("forge_url")
        if not forge_url:
            logger.error("Forge URL not configured for image generation")
            return None
            
        default_params = provider_config.get("default_params", {})
        default_model = provider_config.get("default_model", "novaMatureXL_v35")
        default_size = provider_config.get("default_size", "768x768")
        
        width, height = map(int, default_size.split('x'))
        
        final_params = default_params.copy()
        if parameters:
            final_params.update(parameters)
        
        if "negative_prompt" in final_params:
            final_params["negative_prompt"] = negative_prompt or final_params["negative_prompt"]
        else:
            final_params["negative_prompt"] = negative_prompt or ""
            
        final_params["width"] = final_params.get("width", width)
        final_params["height"] = final_params.get("height", height)
        
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
        
        for key, value in final_params.items():
            if key not in ["negative_prompt", "steps", "cfg_scale", "sampler_name", "restore_faces", "width", "height"]:
                payload[key] = value
        
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"Sending image generation request to {forge_url}")
        logger.info(f"Payload: {payload}")
        
        async with httpx.AsyncClient(timeout=provider_config.get("timeout", 720.0)) as client:
            response = await client.post(
                f"{forge_url}/sdapi/v1/txt2img",
                json=payload,
                headers=headers
            )
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response text (first 200 chars): {response.text[:200]}...")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"Response data keys: {list(data.keys())}")
                    if "images" in data and len(data["images"]) > 0:
                        return data["images"][0]
                    else:
                        logger.error("No images returned from image generation")
                        logger.error(f"Full response: {data}")
                        return None
                except Exception as e:
                    logger.exception(f"Error parsing JSON response: {e}")
                    logger.error(f"Raw response: {response.text}")
                    return None
            else:
                logger.error(f"Image generation failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.exception(f"Error generating image: {e}")
        return None

# Commands (without /tools)
@discord_bot.tree.command(name="diagnose_websearch", description="Diagnose web search configuration (Admin only)")
async def diagnose_websearch_command(interaction: discord.Interaction):
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
    
    await interaction.response.defer(ephemeral=True)
    
    result = await web_search("test query")
    
    embed = discord.Embed(
        title="Web Search Diagnosis",
        color=EMBED_COLOR_ERROR if not result["success"] else EMBED_COLOR_COMPLETE
    )
    
    if result["success"]:
        embed.description = f"✅ Web search is working!\nFound {len(result['results'])} results."
        if result['results']:
            embed.add_field(name="Sample Result", value=result['results'][0]['title'][:100] + "..." if result['results'] else "No results", inline=False)
    else:
        embed.description = f"❌ Web search failed: {result['error']}"
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
    
    await interaction.followup.send(embed=embed, ephemeral=True)

@discord_bot.tree.command(name="test_websearch", description="Test the web search functionality (Admin only)")
async def test_websearch_command(interaction: discord.Interaction):
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
    
    await interaction.response.defer(ephemeral=True)
    
    result = await web_search("test query")
    
    if result["success"]:
        await interaction.followup.send(
            f"Web search test successful! Found {len(result['results'])} results.",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            f"Web search test failed: {result['error']}",
            ephemeral=True
        )

@discord_bot.tree.command(name="image", description="Generate an image from a prompt")
async def image_command(interaction: discord.Interaction, 
                       prompt: str,
                       negative_prompt: Optional[str] = None) -> None:
    await interaction.response.defer(ephemeral=False)
    
    llm_config = config.get("llm", {})
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        await interaction.followup.send("Image generation is not configured.", ephemeral=True)
        return
    
    provider_config = image_providers.get(current_image_provider, {})
    if not provider_config:
        await interaction.followup.send(f"Image provider '{current_image_provider}' not found.", ephemeral=True)
        return
    
    request_id = str(uuid.uuid4())
    await image_queue.put((request_id, prompt, negative_prompt or "", provider_config, {}, interaction.user.id))
    
    async with queue_lock:
        queue_position = image_queue.qsize() + 1
    
    queue_msg = f"Your image generation request has been added to the queue. You are at position **#{queue_position}**."
    await interaction.followup.send(queue_msg, ephemeral=True)
    
    start_time = datetime.now()
    timeout = 1200
    
    while (datetime.now() - start_time).seconds < timeout:
        if request_id in image_results:
            result = image_results.pop(request_id)
            
            if result["status"] == "success":
                try:
                    filepath = result["filepath"]
                    file = discord.File(filepath, filename=os.path.basename(filepath))
                    
                    embed = discord.Embed(title=f"Generated Image for: {prompt[:50]}...", color=EMBED_COLOR_COMPLETE)
                    if negative_prompt:
                        embed.description = f"Negative prompt: {negative_prompt[:50]}..."
                    
                    await interaction.followup.send(embed=embed, file=file)
                    
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logger.warning(f"Could not delete temporary image file {filepath}: {e}")
                        
                except Exception as e:
                    logger.exception(f"Error sending generated image: {e}")
                    await interaction.followup.send("Failed to send generated image.", ephemeral=True)
            else:
                error_msg = result.get("error", "Unknown error")
                await interaction.followup.send(f"Failed to generate image: {error_msg}", ephemeral=True)
            
            return
        
        await asyncio.sleep(5)
    
    await interaction.followup.send("Image generation timed out after 20 minutes.", ephemeral=True)

@discord_bot.tree.command(name="image_advanced", description="Generate an image with advanced parameters")
async def image_advanced_command(interaction: discord.Interaction, 
                                prompt: str,
                                negative_prompt: Optional[str] = None,
                                steps: Optional[int] = None,
                                cfg_scale: Optional[float] = None,
                                width: Optional[int] = None,
                                height: Optional[int] = None) -> None:
    await interaction.response.defer(ephemeral=False)
    
    llm_config = config.get("llm", {})
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        await interaction.followup.send("Image generation is not configured.", ephemeral=True)
        return
    
    provider_config = image_providers.get(current_image_provider, {})
    if not provider_config:
        await interaction.followup.send(f"Image provider '{current_image_provider}' not found.", ephemeral=True)
        return
    
    parameters = {}
    if steps is not None:
        parameters["steps"] = steps
    if cfg_scale is not None:
        parameters["cfg_scale"] = cfg_scale
    if width is not None:
        parameters["width"] = width
    if height is not None:
        parameters["height"] = height
    
    request_id = str(uuid.uuid4())
    await image_queue.put((request_id, prompt, negative_prompt or "", provider_config, parameters, interaction.user.id))
    
    async with queue_lock:
        queue_position = image_queue.qsize() + 1
    
    queue_msg = f"Your image generation request has been added to the queue. You are at position **#{queue_position}**."
    await interaction.followup.send(queue_msg, ephemeral=True)
    
    start_time = datetime.now()
    timeout = 1200
    
    while (datetime.now() - start_time).seconds < timeout:
        if request_id in image_results:
            result = image_results.pop(request_id)
            
            if result["status"] == "success":
                try:
                    filepath = result["filepath"]
                    file = discord.File(filepath, filename=os.path.basename(filepath))
                    
                    embed = discord.Embed(title=f"Generated Image for: {prompt[:50]}...", color=EMBED_COLOR_COMPLETE)
                    if negative_prompt:
                        embed.description = f"Negative prompt: {negative_prompt[:50]}..."
                    
                    await interaction.followup.send(embed=embed, file=file)
                    
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        logger.warning(f"Could not delete temporary image file {filepath}: {e}")
                        
                except Exception as e:
                    logger.exception(f"Error sending generated image: {e}")
                    await interaction.followup.send("Failed to send generated image.", ephemeral=True)
            else:
                error_msg = result.get("error", "Unknown error")
                await interaction.followup.send(f"Failed to generate image: {error_msg}", ephemeral=True)
            
            return
        
        await asyncio.sleep(5)
    
    await interaction.followup.send("Image generation timed out after 20 minutes.", ephemeral=True)

@discord_bot.tree.command(name="providers", description="Switch between different providers/endpoints")
async def providers_command(interaction: discord.Interaction, provider: str) -> None:
    global current_provider, current_model
    
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
    
    current_provider = provider
    provider_config = providers[provider]
    
    default_model = provider_config.get("model") or "default"
    if default_model == "default":
        llm_model = config.get("llm", {}).get("model")
        if llm_model and provider in llm_model:
            default_model = llm_model.split("/", 1)[1] if "/" in llm_model else llm_model
        else:
            default_model = "qwen3"
    
    current_model = default_model
    output = f"Switched to provider: `{provider}` with model: `{default_model}`"
        
    logger.info(output)
    await interaction.response.send_message(output, ephemeral=True)

@discord_bot.tree.command(name="image_providers", description="Switch between different image generation providers")
async def image_providers_command(interaction: discord.Interaction, provider: str) -> None:
    global current_image_provider
    
    llm_config = config.get("llm", {})
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        await interaction.response.send_message("No image providers configured.", ephemeral=True)
        return
        
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
    
    current_image_provider = provider
    output = f"Switched to image provider: `{provider}`"
        
    logger.info(output)
    await interaction.response.send_message(output, ephemeral=True)

@discord_bot.tree.command(name="allow_dm", description="Toggle Direct Message functionality (Admin only)")
async def allow_dm_command(interaction: discord.Interaction, enabled: bool) -> None:
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
    
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f) or {}
        
        current_config["allow_dms"] = enabled
        
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
        
        status = "enabled" if enabled else "disabled"
        message = f"Direct Messages have been {status}."
        logger.info(message)
        await interaction.response.send_message(message, ephemeral=True)
        
    except Exception as e:
        error_msg = f"Failed to update DM settings: {e}"
        logger.error(error_msg)
        await interaction.response.send_message(error_msg, ephemeral=True)

@discord_bot.tree.command(name="image_triggers", description="Show configured image triggers")
async def image_triggers_command(interaction: discord.Interaction):
    try:
        config = await asyncio.to_thread(get_config)
        image_config = config.get("llm", {}).get("image_generator", {})
        image_triggers = image_config.get("image_triggers", [
            "make me an image",
            "generate image",
            "create an image",
            "draw this",
            "show me a picture of",
            "image of",
            "picture of"
        ])
        
        triggers_list = "\n".join([f"• {trigger}" for trigger in image_triggers])
        
        embed = discord.Embed(
            title="Image Generation Triggers",
            description=f"**Current image triggers:**\n{triggers_list}",
            color=discord.Color.blue()
        )
        embed.set_footer(text="These triggers will activate image generation")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    except Exception as e:
        logger.exception(f"Error showing image triggers: {e}")
        await interaction.response.send_message("Failed to retrieve image triggers.", ephemeral=True)

@discord_bot.tree.command(name="web_search_triggers", description="Show configured web search triggers")
async def web_search_triggers_command(interaction: discord.Interaction):
    try:
        config = await asyncio.to_thread(get_config)
        web_search_config = config.get("llm", {}).get("web_search", {})
        trigger_words = web_search_config.get("trigger_words", [
            "search", "find", "google", "look up", "who is", 
            "find me", "double check", "check again", "what is", "doublecheck"
        ])
        
        triggers_list = "\n".join([f"• {trigger}" for trigger in trigger_words])
        
        embed = discord.Embed(
            title="Web Search Triggers",
            description=f"**Current web search triggers:**\n{triggers_list}",
            color=discord.Color.green()
        )
        embed.set_footer(text="These triggers will activate web search")
        
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    except Exception as e:
        logger.exception(f"Error showing web search triggers: {e}")
        await interaction.response.send_message("Failed to retrieve web search triggers.", ephemeral=True)

@providers_command.autocomplete("provider")
async def provider_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[AppChoice[str]]:
    llm_config = config.get("llm", {})
    providers = llm_config.get("providers", {})
    
    if not providers:
        return []
        
    filtered_providers = [p for p in providers.keys() if curr_str.lower() in p.lower()]
    return [AppChoice(name=p, value=p) for p in filtered_providers[:25]]

@image_providers_command.autocomplete("provider")
async def image_provider_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[AppChoice[str]]:
    llm_config = config.get("llm", {})
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        return []
    
    actual_providers = [k for k, v in image_providers.items() if isinstance(v, dict) and 'forge_url' in v]
    
    if not actual_providers:
        return []
        
    filtered_providers = [p for p in actual_providers if curr_str.lower() in p.lower()]
    return [AppChoice(name=p, value=p) for p in filtered_providers[:25]]

@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logger.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    try:
        synced = await discord_bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s)")
        for cmd in synced:
            logger.info(f"  - {cmd.name}")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")
    
    discord_bot.loop.create_task(image_generation_worker())

@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time, current_provider, current_model

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if new_msg.author.bot:
        return

    should_process = False
    should_process = is_dm or discord_bot.user.mention in new_msg.content

    if not should_process:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

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

    if is_dm and not allow_dms:
        return

    if is_bad_user or is_bad_channel:
        return

    llm_config = config.get("llm", {})
    
    if current_provider is None:
        providers = llm_config.get("providers", {})
        if providers:
            current_provider = list(providers.keys())[0]
            provider_config = providers[current_provider]
            
            default_model = config.get("llm", {}).get("model")
            if default_model:
                if "/" in default_model:
                    current_model = default_model.split("/", 1)[1]
                else:
                    current_model = default_model
            else:
                current_model = "qwen3"
        else:
            logger.error("No providers configured!")
            return
    
    provider_config = llm_config.get("providers", {}).get(current_provider, {})
    
    if not provider_config:
        logger.error(f"Provider configuration not found for: {current_provider}")
        return

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = {}
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

    messages = []
    user_warnings = set()
    curr_msg = new_msg

    # Track if we have a user message in the conversation
    user_message_found = False
    user_display_name = None
    user_id_for_mention = None

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

                # --- FIXED VISION HANDLING ---
                # Determine if we should send images to the LLM based on:
                # 1. Current model being a vision model, OR
                # 2. Message having images AND being relevant to images
                has_image_attachments = any(att.content_type and att.content_type.startswith("image") for att in new_msg.attachments)
                image_relevant_prompt = any(word in new_msg.content.lower() for word in ["image", "picture", "photo", "what is in", "describe", "see", "looking at"])

                # Always enable image sending when there are images AND the prompt is image-related
                # Even if the model doesn't support vision natively
                if has_image_attachments and image_relevant_prompt:
                    # Override the accept_images flag to always include images
                    accept_images = True
                    max_images = 5  # Ensure we can send up to 5 images

                # Build image data for messages
                if has_image_attachments and image_relevant_prompt:
                    good_attachments = [att for att in new_msg.attachments if att.content_type and att.content_type.startswith("image")]
                    attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])
                    curr_node.images = [
                        dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                        for att, resp in zip(good_attachments, attachment_responses)
                        if att.content_type.startswith("image")
                    ]
                else:
                    curr_node.images = []

                # Build the final content for the LLM
                if curr_node.images[:max_images]:
                    content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
                else:
                    content = curr_node.text[:max_text]

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
                    logger.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            # Track user message information for later use
            if curr_node.role == "user" and curr_node.user_id:
                user_message_found = True
                user_id_for_mention = curr_node.user_id
                try:
                    user = await discord_bot.fetch_user(curr_node.user_id)
                    user_display_name = user.display_name
                except:
                    user_display_name = str(curr_node.user_id)

            # Include user display name in content for user messages when appropriate
            if content != "":
                # Add user mention to the content for user messages
                if curr_node.role == "user" and accept_usernames and curr_node.user_id:
                    try:
                        user = await discord_bot.fetch_user(curr_node.user_id)
                        display_name = user.display_name
                        # Add user mention to the start of content for user messages
                        if isinstance(content, list):  # For vision model content
                            # For vision models, we need to add it to the text part
                            if content[0].get('type') == 'text':
                                content[0]['text'] = f"@{display_name} {content[0]['text']}"
                        else:
                            # For regular text, prepend the mention
                            if not content.startswith(f"@{display_name}"):
                                content = f"@{display_name} {content}"
                    except Exception as e:
                        logger.warning(f"Could not fetch user {curr_node.user_id}: {e}")
                        # Fallback to user ID if we can't fetch the user
                        if isinstance(content, list):
                            if content[0].get('type') == 'text':
                                content[0]['text'] = f"@{curr_node.user_id} {content[0]['text']}"
                        else:
                            if not content.startswith(f"@{curr_node.user_id}"):
                                content = f"@{curr_node.user_id} {content}"
                
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    try:
                        user = await discord_bot.fetch_user(curr_node.user_id)
                        display_name = user.display_name
                        message["name"] = display_name
                    except:
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

    logger.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    # Image trigger detection
    image_generator_enabled = "image_generator" in llm_config.get("active_tools", [])
    image_providers = llm_config.get("image_generator", {})
    
    if image_generator_enabled and image_providers:
        image_config = config.get("llm", {}).get("image_generator", {})
        image_triggers = image_config.get("image_triggers", [
            "make me an image",
            "generate image",
            "create an image",
            "draw this",
            "show me a picture of",
            "image of",
            "picture of"
        ])
        
        content_lower = new_msg.content.lower()
        trigger_found = False
        prompt_text = ""
        
        for trigger in image_triggers:
            if content_lower.startswith(trigger) or content_lower.startswith(f"{trigger} "):
                trigger_found = True
                prompt_text = new_msg.content[len(trigger):].strip()
                break
            elif trigger in content_lower:
                trigger_found = True
                trigger_index = content_lower.find(trigger)
                prompt_text = new_msg.content[trigger_index + len(trigger):].strip()
                break
        
        if trigger_found and prompt_text:
            logger.info(f"Image trigger detected: {new_msg.content}")
            
            try:
                await new_msg.channel.trigger_typing()
            except:
                pass
            
            provider_config = image_providers.get(current_image_provider, {})
            if not provider_config:
                logger.error(f"Image provider '{current_image_provider}' not found for trigger")
                return
            
            request_id = str(uuid.uuid4())
            await image_queue.put((request_id, prompt_text, "", provider_config, {}, new_msg.author.id))
            
            async with queue_lock:
                queue_position = image_queue.qsize() + 1
            
            # Send a simple queue position message without the "Your image generation request has been added to the queue" text
            queue_msg = f"You are at position **#{queue_position}** in the image generation queue."
            await new_msg.reply(queue_msg, mention_author=False)
            
            start_time = datetime.now()
            timeout = 1200
            
            while (datetime.now() - start_time).seconds < timeout:
                if request_id in image_results:
                    result = image_results.pop(request_id)
                    
                    if result["status"] == "success":
                        try:
                            filepath = result["filepath"]
                            file = discord.File(filepath, filename=os.path.basename(filepath))
                            
                            embed = discord.Embed(title=f"Generated Image for: {prompt_text[:50]}...", color=EMBED_COLOR_COMPLETE)
                            
                            await new_msg.reply(embed=embed, file=file, mention_author=False)
                            
                            try:
                                os.remove(filepath)
                            except Exception as e:
                                logger.warning(f"Could not delete temporary image file {filepath}: {e}")
                                
                        except Exception as e:
                            logger.exception(f"Error sending generated image: {e}")
                            await new_msg.reply("Failed to send generated image.", mention_author=False)
                    else:
                        error_msg = result.get("error", "Unknown error")
                        await new_msg.reply(f"Failed to generate image: {error_msg}", mention_author=False)
                    
                    return
                
                await asyncio.sleep(5)
            
            await new_msg.reply("Image generation timed out after 20 minutes.", mention_author=False)
            return

    # Web search integration
    if "web_search" in llm_config.get("active_tools", []):
        web_search_config = llm_config.get("web_search", {})
        trigger_words = web_search_config.get("trigger_words", [
            "search", "find", "google", "look up", "who is", 
            "find me", "double check", "check again", "what is", "doublecheck"
        ])
        
        if any(word in new_msg.content.lower() for word in trigger_words):
            logger.info(f"Trigger word found in message: {new_msg.content}")
            
            search_data = await web_search(new_msg.content)
            
            if search_data["success"] and search_data["results"]:
                formatted_search = format_search_results(search_data)
                
                search_message = {
                    "role": "user",
                    "content": formatted_search
                }
                messages.insert(1, search_message)
                
                logger.info("Search results formatted and added to conversation for LLM processing")
            else:
                logger.warning(f"Web search failed: {search_data.get('error', 'Unknown error')}")

    # System prompt handling
    system_prompt = llm_config.get("system_prompt") or ""
    
    if system_prompt:
        now = datetime.now().astimezone()
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        
        if accept_usernames:
            system_prompt += "\n\nWhen responding to users in Discord, always mention the user by their display name (e.g., 'Hello @username!')."

        messages.append(dict(role="system", content=system_prompt))
    else:
        logger.info("No system prompt found in config")

    # Generate response
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    model_to_use = current_model or "qwen3"
    
    openai_kwargs = dict(model=model_to_use, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096
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
                        # Process reasoning sections based on configuration
                        reasoning_config = llm_config.get("reasoning", {})
                        show_reasoning = reasoning_config.get("show_reasoning", True)
                        reasoning_format = reasoning_config.get("reasoning_format", "spoiler")
                        start_tag = reasoning_config.get("reasoning_start_tag", "<THINKING>")
                        end_tag = reasoning_config.get("reasoning_end_tag", "</THINKING>")
                        
                        # Remove the STREAMING_INDICATOR from the displayed content
                        display_content = response_contents[-1]
                        if display_content.endswith(STREAMING_INDICATOR):
                            display_content = display_content[:-len(STREAMING_INDICATOR)]
                        
                        # Apply reasoning processing if enabled
                        if show_reasoning and display_content:
                            processed_content, reasoning_content, has_reasoning = process_reasoning_content(
                                display_content,
                                reasoning_format,
                                start_tag,
                                end_tag
                            )
                            
                            # Apply format-specific handling
                            if reasoning_format == "spoiler" and has_reasoning:
                                final_content = processed_content
                            elif reasoning_format == "separate" and has_reasoning:
                                final_content = processed_content
                                # Note: We'll handle the separate message view in final processing
                            else:
                                final_content = display_content
                        else:
                            # Completely remove reasoning sections if not showing
                            clean_content, _ = extract_reasoning(display_content, start_tag, end_tag)
                            final_content = clean_content
                        
                        # Truncate if needed
                        if len(final_content) > max_message_length:
                            final_content = final_content[:max_message_length-3] + "..."
                        
                        # Add user mention to the beginning of the response if it's a user message
                        if user_message_found and user_display_name and not final_content.strip().startswith(f"@{user_display_name}"):
                            # Check if response already mentions the user to avoid duplication
                            if f"@{user_display_name}" not in final_content:
                                final_content = f"@{user_display_name} {final_content}"
                        
                        embed.description = final_content if is_final_edit else final_content
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    # Add user mention to the beginning of the response if it's a user message
                    if user_message_found and user_display_name and not content.strip().startswith(f"@{user_display_name}"):
                        # Check if response already mentions the user to avoid duplication
                        if f"@{user_display_name}" not in content:
                            content = f"@{user_display_name} {content}"
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception as e:
        logger.exception(f"Error while generating response: {e}")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

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
