from typing import Dict, Any, List, Optional
from lamb.lamb_classes import Assistant
import json
from lamb.logging_config import get_logger

logger = get_logger(__name__, component="MAIN")


def _has_vision_capability(assistant: Assistant) -> bool:
    """
    Check if the assistant has vision capabilities enabled.

    Args:
        assistant: Assistant object with metadata

    Returns:
        bool: True if vision is enabled, False otherwise
    """
    if not assistant:
        return False

    # Check if assistant has metadata (stored in api_callback column)
    metadata_str = getattr(assistant, 'metadata', None) or getattr(assistant, 'api_callback', None)
    if not metadata_str:
        return False

    try:
        metadata = json.loads(metadata_str)
        capabilities = metadata.get('capabilities', {})
        return capabilities.get('vision', False)
    except (json.JSONDecodeError, AttributeError):
        return False

def _has_image_generation_capability(assistant: Assistant) -> bool:
    """
    Check if the assistant has image generation capabilities enabled.

    Args:
        assistant: Assistant object with metadata

    Returns:
        bool: True if image generation is enabled, False otherwise
    """
    if not assistant:
        return False

    # Check if assistant has metadata (stored in api_callback column)
    metadata_str = getattr(assistant, 'metadata', None) or getattr(assistant, 'api_callback', None)
    if not metadata_str:
        return False

    try:
        metadata = json.loads(metadata_str)
        capabilities = metadata.get('capabilities', {})
        return capabilities.get('image_generation', False)
    except (json.JSONDecodeError, AttributeError):
        return False
def prompt_processor(
    request: Dict[str, Any],
    assistant: Optional[Assistant] = None,
    rag_context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """
    Simple augment prompt processor that:
    1. Uses system prompt from assistant if available
    2. Replaces last message with prompt template, substituting:
       - {user_input} with the original message
       - {context} with RAG context if available
    """
    messages = request.get('messages', [])
    if not messages:
        return messages

    # Get the last user message
    last_message = messages[-1]['content']

    # Create new messages list
    processed_messages = []

    if assistant:
        # Add system message from assistant if available
        if assistant.system_prompt:
            processed_messages.append({
                "role": "system",
                "content": assistant.system_prompt
            })
        
        # Add previous messages except the last one
        processed_messages.extend(messages[:-1])
        
        # Process the last message using the prompt template
        if assistant.prompt_template:
            # Check if assistant has vision capabilities
            has_vision = _has_vision_capability(assistant)

            if isinstance(last_message, list) and has_vision:
                # TODO: Add multi-context support for vision path (currently only {context} is replaced)
                # Multimodal content with vision-enabled assistant
                # Preserve images while applying template augmentations
                augmented_content = []

                # Extract text parts for {user_input} substitution
                text_parts = []
                for item in last_message:
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))

                user_input_text = ' '.join(text_parts)

                # Create augmented text content with template
                logger.debug(f"User message: {user_input_text}")
                augmented_text = assistant.prompt_template.replace("{user_input}", "\n\n" + user_input_text + "\n\n")

                # Add RAG context if available
                if rag_context:
                    context = rag_context.get("context", "") if isinstance(rag_context, dict) else str(rag_context)
                    
                    # Format sources if available
                    sources_text = ""
                    if isinstance(rag_context, dict) and "sources" in rag_context:
                        sources = rag_context["sources"]
                        if sources:
                            sources_text = "\n\n## Available Sources\n\n"
                            for i, source in enumerate(sources, 1):
                                title = source.get("title", "Unknown")
                                url = source.get("url", "")
                                similarity = source.get("similarity", 0)
                                sources_text += f"{i}. [{title}]({url}) (similarity: {similarity:.3f})\n"
                    
                    # Combine context with sources
                    full_context = context + sources_text
                    augmented_text = augmented_text.replace("{context}", "\n\n" + full_context + "\n\n")
                else:
                    augmented_text = augmented_text.replace("{context}", "")

                # Add the augmented text as first element
                augmented_content.append({
                    "type": "text",
                    "text": augmented_text
                })

                # Preserve all non-text elements (images, etc.)
                for item in last_message:
                    if item.get('type') != 'text':
                        augmented_content.append(item)

                # Add processed multimodal message
                processed_messages.append({
                    "role": messages[-1]['role'],
                    "content": augmented_content
                })

            else:
                # Text-only processing (legacy format or vision-disabled assistant)
                if isinstance(last_message, list):
                    # Extract text parts only (strips images for security)
                    text_parts = []
                    for item in last_message:
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                    user_input_text = ' '.join(text_parts)
                else:
                    # Legacy string format
                    user_input_text = str(last_message)

                # Replace placeholders in template
                logger.debug(f"User message: {user_input_text}")
                prompt = assistant.prompt_template.replace("{user_input}", "\n\n" + user_input_text + "\n\n")

                # Add RAG contexts if available (supports multi-context)
                if rag_context and isinstance(rag_context, dict):
                    is_multicontext = any(
                        isinstance(rag_context.get(k), dict)
                        for k in ("context", "context2")
                    )

                    if is_multicontext:
                        for context_key in ["context", "context2", "context3", "context4", "context5"]:
                            tool_result = rag_context.get(context_key, {})
                            if isinstance(tool_result, dict):
                                context_text = tool_result.get("context", "")
                            else:
                                context_text = str(tool_result) if tool_result else ""

                            if context_text:
                                prompt = prompt.replace("{" + context_key + "}", "\n\n" + context_text + "\n\n")
                            else:
                                prompt = prompt.replace("{" + context_key + "}", "")

                            if context_key == "context" and isinstance(tool_result, dict):
                                sources = tool_result.get("sources", [])
                                if sources:
                                    sources_text = "\n\n## Available Sources\n\n"
                                    for i, source in enumerate(sources, 1):
                                        title = source.get("title", "Unknown")
                                        url = source.get("url", "")
                                        similarity = source.get("similarity", 0)
                                        sources_text += f"{i}. [{title}]({url}) (similarity: {similarity:.3f})\n"
                                    prompt += sources_text
                    else:
                        context = rag_context.get("context", "")
                        sources_text = ""
                        if "sources" in rag_context:
                            sources = rag_context["sources"]
                            if sources:
                                sources_text = "\n\n## Available Sources\n\n"
                                for i, source in enumerate(sources, 1):
                                    title = source.get("title", "Unknown")
                                    url = source.get("url", "")
                                    similarity = source.get("similarity", 0)
                                    sources_text += f"{i}. [{title}]({url}) (similarity: {similarity:.3f})\n"
                        full_context = context + sources_text
                        prompt = prompt.replace("{context}", "\n\n" + full_context + "\n\n")
                        for extra_key in ["context2", "context3", "context4", "context5"]:
                            prompt = prompt.replace("{" + extra_key + "}", "")
                else:
                    for context_key in ["context", "context2", "context3", "context4", "context5"]:
                        prompt = prompt.replace("{" + context_key + "}", "")

                # Add processed text message
                processed_messages.append({
                    "role": messages[-1]['role'],
                    "content": prompt
                })
        else:
            # If no template, use original message
            processed_messages.append(messages[-1])
            
        return processed_messages
    
    # If no assistant provided, return original messages
    return messages 