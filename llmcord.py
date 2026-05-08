import asyncio
import os
import uuid
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional
import re

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl", "qwen", "")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
EMBED_COLOR_ERROR = discord.Color.red()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500
IMAGE_STORAGE_FOLDER = "generated_images"
os.makedirs(IMAGE_STORAGE_FOLDER, exist_ok=True)


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


config = get_config()
llm_config = config.get("llm", {})

image_queue = asyncio.Queue()
image_results = {}
queue_lock = asyncio.Lock()

msg_nodes = {}
last_task_time = 0

current_provider = None
current_model = None
current_image_provider = None

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


def strip_jinja_templates(content: str) -> str:
    if not content:
        return content
    
    content = re.sub(r'\{\{\s*.*?\s*\}\}', '', content)
    content = re.sub(r'\{%\s*.*?\s*%\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\{#.*?#\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    return content.strip()


def parse_reasoning_effort(reasoning_effort: str, max_tokens: int = 4096) -> int:
    """
    Parse reasoning_effort string to reasoning_budget integer for koboldcpp API.
    
    Supported values: high, medium, low, minimal, none
    Returns: reasoning_budget as token count (-1 for unrestricted)
    """
    reasoning_effort = reasoning_effort.strip().lower() if reasoning_effort else ''
    
    if reasoning_effort == "none":
        return 0
    elif reasoning_effort == "minimal":
        return int(0.1 * max_tokens)
    elif reasoning_effort == "low":
        return int(0.25 * max_tokens)
    elif reasoning_effort == "medium":
        return int(0.5 * max_tokens)
    elif reasoning_effort == "high":
        return int(0.75 * max_tokens)
    else:
        return -1


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def image_generation_worker():
    while True:
        try:
            request_id, prompt, negative_prompt, provider_config, parameters, user_id = await image_queue.get()
            
            async with queue_lock:
                queue_position = image_queue.qsize() + 1
            
            try:
                image_data = await generate_image(prompt, negative_prompt, provider_config, parameters)
                
                if image_data:
                    filename = f"{uuid.uuid4()}.png"
                    filepath = os.path.join(IMAGE_STORAGE_FOLDER, filename)
                    
                    try:
                        if "," in image_data:
                            _, image_data = image_data.split(",", 1)
                        
                        import base64
                        image_bytes = base64.b64decode(image_data)
                        
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
                async with queue_lock:
                    pass
                image_queue.task_done()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logging.exception(f"Unexpected error in image worker: {e}")


async def generate_image(prompt: str, negative_prompt: str, provider_config: dict, parameters: dict = None) -> Optional[str]:
    try:
        forge_url = provider_config.get("forge_url")
        if not forge_url:
            logging.error("Forge URL not configured for image generation")
            return None
            
        default_params = provider_config.get("default_params", {})
        default_model = provider_config.get("default_model", "default")
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
        
        logging.info(f"Sending image generation request to {forge_url}")
        
        async with httpx.AsyncClient(timeout=provider_config.get("timeout", 720.0)) as client:
            response = await client.post(
                f"{forge_url}/sdapi/v1/txt2img",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "images" in data and len(data["images"]) > 0:
                        return data["images"][0]
                    else:
                        logging.error("No images returned from image generation")
                        return None
                except Exception as e:
                    logging.exception(f"Error parsing JSON response: {e}")
                    return None
            else:
                logging.error(f"Image generation failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logging.exception(f"Error generating image: {e}")
        return None


@discord_bot.tree.command(name="image", description="Generate an image from a prompt")
async def image_command(interaction: discord.Interaction, 
                       prompt: str,
                       negative_prompt: Optional[str] = None) -> None:
    await interaction.response.defer(ephemeral=False)
    
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        await interaction.followup.send("Image generation is not configured.", ephemeral=True)
        return
    
    actual_providers = {k: v for k, v in image_providers.items() if isinstance(v, dict) and 'forge_url' in v}
    
    if not actual_providers:
        await interaction.followup.send("No valid image providers configured.", ephemeral=True)
        return
    
    provider_config = actual_providers.get(current_image_provider or list(actual_providers.keys())[0], {})
    if not provider_config:
        await interaction.followup.send(f"Image provider not found.", ephemeral=True)
        return
    
    request_id = str(uuid.uuid4())
    
    async with queue_lock:
        queue_position = image_queue.qsize() + 1
    
    await image_queue.put((request_id, prompt, negative_prompt or "", provider_config, {}, interaction.user.id))
    
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
                        logging.warning(f"Could not delete temporary image file {filepath}: {e}")
                        
                except Exception as e:
                    logging.exception(f"Error sending generated image: {e}")
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
    
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        await interaction.followup.send("Image generation is not configured.", ephemeral=True)
        return
    
    actual_providers = {k: v for k, v in image_providers.items() if isinstance(v, dict) and 'forge_url' in v}
    
    if not actual_providers:
        await interaction.followup.send("No valid image providers configured.", ephemeral=True)
        return
    
    provider_config = actual_providers.get(current_image_provider or list(actual_providers.keys())[0], {})
    if not provider_config:
        await interaction.followup.send(f"Image provider not found.", ephemeral=True)
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
    
    async with queue_lock:
        queue_position = image_queue.qsize() + 1
    
    await image_queue.put((request_id, prompt, negative_prompt or "", provider_config, parameters, interaction.user.id))
    
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
                        logging.warning(f"Could not delete temporary image file {filepath}: {e}")
                        
                except Exception as e:
                    logging.exception(f"Error sending generated image: {e}")
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
        
    logging.info(output)
    await interaction.response.send_message(output, ephemeral=True)


@discord_bot.tree.command(name="image_providers", description="Switch between different image generation providers")
async def image_providers_command(interaction: discord.Interaction, provider: str) -> None:
    global current_image_provider
    
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
        
    logging.info(output)
    await interaction.response.send_message(output, ephemeral=True)


@discord_bot.tree.command(name="reset_memory", description="Reset your conversation memory/history")
async def reset_memory_command(interaction: discord.Interaction) -> None:
    user_id = interaction.user.id
    
    nodes_to_remove = [
        msg_id for msg_id, node in msg_nodes.items()
        if node.parent_msg and node.parent_msg.author.id == user_id
    ]
    
    for msg_id in nodes_to_remove:
        msg_nodes.pop(msg_id, None)
    
    logging.info(f"Reset conversation memory for user ID: {user_id}, removed {len(nodes_to_remove)} nodes")
    await interaction.response.send_message(f"✅ Your conversation memory has been reset. Removed {len(nodes_to_remove)} message nodes.", ephemeral=True)


@discord_bot.tree.command(name="allow_dm", description="Toggle Direct Message functionality (Admin only)")
async def allow_dm_command(interaction: discord.Interaction, enabled: bool) -> None:
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
    
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f) or {}
        
        current_config["allow_dms"] = enabled
        if "llm" not in current_config:
            current_config["llm"] = {}
        current_config["llm"]["allow_dms"] = enabled
        
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


@discord_bot.tree.command(name="toggle_jinja", description="Toggle Jinja template support for LLM models (Admin only)")
async def toggle_jinja_command(interaction: discord.Interaction, enabled: bool) -> None:
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
    
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f) or {}
        
        if "llm" not in current_config:
            current_config["llm"] = {}
        
        current_config["llm"]["enable_jinja"] = enabled
        
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
        
        status = "enabled" if enabled else "disabled"
        message = f"Jinja template support has been {status}."
        logging.info(message)
        await interaction.response.send_message(message, ephemeral=True)
        
    except Exception as e:
        error_msg = f"Failed to update Jinja settings: {e}"
        logging.error(error_msg)
        await interaction.response.send_message(error_msg, ephemeral=True)


@providers_command.autocomplete("provider")
async def provider_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    providers = llm_config.get("providers", {})
    
    if not providers:
        return []
        
    filtered_providers = [p for p in providers.keys() if curr_str.lower() in p.lower()]
    return [Choice(name=p, value=p) for p in filtered_providers[:25]]


@image_providers_command.autocomplete("provider")
async def image_provider_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    image_providers = llm_config.get("image_generator", {})
    
    if not image_providers:
        return []
    
    actual_providers = [k for k, v in image_providers.items() if isinstance(v, dict) and 'forge_url' in v]
    
    if not actual_providers:
        return []
        
    filtered_providers = [p for p in actual_providers if curr_str.lower() in p.lower()]
    return [Choice(name=p, value=p) for p in filtered_providers[:25]]


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()
    
    discord_bot.loop.create_task(image_generation_worker())


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time, current_provider, current_model, current_image_provider

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if new_msg.author.bot:
        return

    config = await asyncio.to_thread(get_config)
    llm_config = config.get("llm", {})

    should_process = False
    
    has_mention = discord_bot.user.mention in new_msg.content
    
    triggerword_enabled = llm_config.get("enable_triggerword", False)
    triggerword = llm_config.get("triggerword", "").lower().strip()
    has_triggerword = False
    
    if triggerword_enabled and triggerword:
        content_lower = new_msg.content.lower()
        if content_lower.startswith(triggerword) or content_lower.startswith(f"{triggerword} "):
            has_triggerword = True
        elif f" {triggerword} " in content_lower:
            has_triggerword = True
        elif f" {triggerword}" in content_lower and content_lower.endswith(triggerword):
            has_triggerword = True
    
    should_process = is_dm or has_mention or has_triggerword

    if not should_process:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

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
    
    if current_image_provider is None:
        image_providers = llm_config.get("image_generator", {})
        if image_providers:
            actual_providers = {k: v for k, v in image_providers.items() if isinstance(v, dict) and 'forge_url' in v}
            if actual_providers:
                current_image_provider = list(actual_providers.keys())[0]

    provider_config = llm_config.get("providers", {}).get(current_provider, {})
    
    if not provider_config:
        logging.error(f"Provider configuration not found for: {current_provider}")
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
    
    reasoning_config = llm_config.get("reasoning") or {}
    completely_disable_reasoning = reasoning_config.get("completely_disable_reasoning", False)
    reasoning_effort = reasoning_config.get("reasoning_effort", "")
    
    base_extra_body = provider_config.get("extra_body") or {}
    if model_parameters:
        base_extra_body = base_extra_body | model_parameters
    
    if completely_disable_reasoning:
        base_extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        base_extra_body["enable_thinking"] = False
        base_extra_body["no_thinking"] = True
        base_extra_body["disable_thinking"] = True
        base_extra_body["thinking"] = False
    
    if reasoning_effort:
        max_tokens = config.get("llm", {}).get("extra_api_parameters", {}).get("max_tokens", 4096)
        reasoning_budget = parse_reasoning_effort(reasoning_effort, max_tokens)
        if reasoning_budget >= 0:
            base_extra_body["reasoning_effort"] = reasoning_effort
            base_extra_body["reasoning_budget"] = reasoning_budget
            logging.info(f"Reasoning effort set to '{reasoning_effort}' with budget of {reasoning_budget} tokens")
    
    extra_body = base_extra_body if base_extra_body else None

    accept_images = any(x in current_model.lower() for x in VISION_MODEL_TAGS) if current_model else False

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None:
                cleaned_content = curr_msg.content
                
                if discord_bot.user.mention in cleaned_content:
                    cleaned_content = cleaned_content.removeprefix(discord_bot.user.mention).lstrip()
                
                triggerword_enabled = llm_config.get("enable_triggerword", False)
                triggerword = llm_config.get("triggerword", "").strip()
                if triggerword_enabled and triggerword:
                    content_lower = cleaned_content.lower()
                    if content_lower.startswith(triggerword):
                        cleaned_content = cleaned_content[len(triggerword):].lstrip()
                    elif content_lower.startswith(f"{triggerword} "):
                        cleaned_content = cleaned_content[len(triggerword) + 1:].lstrip()
                    elif f" {triggerword} " in content_lower:
                        cleaned_content = cleaned_content.replace(f" {triggerword} ", " ", 1).strip()
                    elif f" {triggerword}" in content_lower and content_lower.endswith(triggerword):
                        cleaned_content = cleaned_content[:content_lower.rfind(f" {triggerword}")].strip()
                
                cleaned_content = cleaned_content.lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference is None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference is None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = [dict(type="text", text=curr_node.text[:max_text])] + curr_node.images[:max_images] if curr_node.text and curr_node.text[:max_text] else curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text] if curr_node.text else ""

            if content != "":
                messages.append(dict(content=content, role=curr_node.role))

            if curr_node.text and len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if curr_node.images and len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

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
            logging.info(f"Image trigger detected: {new_msg.content}")
            
            try:
                await new_msg.channel.trigger_typing()
            except:
                pass
            
            actual_providers = {k: v for k, v in image_providers.items() if isinstance(v, dict) and 'forge_url' in v}
            provider_config = actual_providers.get(current_image_provider or list(actual_providers.keys())[0], {})
            if not provider_config:
                logging.error(f"Image provider not found for trigger")
                return
            
            request_id = str(uuid.uuid4())
            
            async with queue_lock:
                queue_position = image_queue.qsize() + 1
            
            await image_queue.put((request_id, prompt_text, "", provider_config, {}, new_msg.author.id))
            
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
                                logging.warning(f"Could not delete temporary image file {filepath}: {e}")
                                
                        except Exception as e:
                            logging.exception(f"Error sending generated image: {e}")
                            await new_msg.reply("Failed to send generated image.", mention_author=False)
                    else:
                        error_msg = result.get("error", "Unknown error")
                        await new_msg.reply(f"Failed to generate image: {error_msg}", mention_author=False)
                    
                    return
                
                await asyncio.sleep(5)
            
            await new_msg.reply("Image generation timed out after 20 minutes.", mention_author=False)
            return

    if "web_search" in llm_config.get("active_tools", []):
        web_search_config = llm_config.get("web_search", {})
        trigger_words = web_search_config.get("trigger_words", [
            "search", "find", "google", "look up", "who is", 
            "double check", "check again", "what is", "doublecheck"
        ])
        
        if any(word in new_msg.content.lower() for word in trigger_words):
            logging.info(f"Trigger word found in message: {new_msg.content}")
            
            base_url = web_search_config.get("search_url")
            max_results = web_search_config.get("max_results", 5)
            timeout = web_search_config.get("timeout", 30)
            
            try:
                params = {"q": new_msg.content, "format": "json"}
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
                            }
                            results.append(basic_result)
                        
                        if results:
                            formatted_search = "Here are some search results:\n\n" + "\n\n".join([f"{i+1}. [{r['title']}]({r['url']})\n   {r['content']}" for i, r in enumerate(results)])
                            
                            search_message = {
                                "role": "user",
                                "content": formatted_search
                            }
                            messages.insert(1, search_message)
                            
                            logging.info("Search results added to conversation")
                    except Exception as e:
                        logging.warning(f"JSON parsing failed: {e}")
            except Exception as e:
                logging.exception(f"Web search failed: {e}")

    system_prompt = llm_config.get("system_prompt") or ""
    
    enable_jinja = llm_config.get("enable_jinja", False)
    
    completely_disable_reasoning = reasoning_config.get("completely_disable_reasoning", False)
    
    if system_prompt:
        now = datetime.now().astimezone()
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        
        if not enable_jinja:
            system_prompt = strip_jinja_templates(system_prompt)
        
        messages.append(dict(role="system", content=system_prompt))

    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

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
                if finish_reason is not None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason is None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason is None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason is not None or msg_split_incoming
                    is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")

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

    except Exception:
        logging.exception("Error while generating response")

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
