from typing import Dict, Any, List, Optional, Union
from lamb.lamb_classes import Assistant
import json
from lamb.logging_config import get_logger

logger = get_logger(__name__, component="MAIN")

CONTEXT_KEYS = ["context", "context2", "context3", "context4", "context5"]


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


def _format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format RAG sources as markdown citation list."""
    if not sources:
        return ""
    sources_text = "\n\n## Available Sources\n\n"
    for i, source in enumerate(sources, 1):
        title = source.get("title", "Unknown")
        url = source.get("url", "")
        similarity = source.get("similarity", 0)
        sources_text += f"{i}. [{title}]({url}) (similarity: {similarity:.3f})\n"
    return sources_text


def _extract_user_input_text(last_message: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract user text from a string or multimodal content list."""
    if isinstance(last_message, list):
        text_parts = [
            item.get("text", "")
            for item in last_message
            if item.get("type") == "text"
        ]
        return " ".join(text_parts)
    return str(last_message)


def _apply_template_with_rag_contexts(
    template: str,
    user_input: str,
    rag_context: Optional[Dict[str, Any]],
) -> str:
    """Replace {user_input} and context placeholders in the prompt template."""
    prompt = template.replace("{user_input}", "\n\n" + user_input + "\n\n")

    if rag_context and isinstance(rag_context, dict):
        is_multicontext = any(
            isinstance(rag_context.get(k), dict)
            for k in ("context", "context2")
        )

        if is_multicontext:
            for context_key in CONTEXT_KEYS:
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
                        prompt += _format_sources(sources)
        else:
            context = rag_context.get("context", "")
            sources_text = _format_sources(rag_context.get("sources", []))
            full_context = context + sources_text
            prompt = prompt.replace("{context}", "\n\n" + full_context + "\n\n")
            for extra_key in CONTEXT_KEYS[1:]:
                prompt = prompt.replace("{" + extra_key + "}", "")
    else:
        for context_key in CONTEXT_KEYS:
            prompt = prompt.replace("{" + context_key + "}", "")

    return prompt


def _build_multimodal_content(augmented_text: str, last_message: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build multimodal content preserving non-text items (images, etc.)."""
    augmented_content = [{"type": "text", "text": augmented_text}]
    for item in last_message:
        if item.get("type") != "text":
            augmented_content.append(item)
    return augmented_content


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
       - {context}, {context2}, ... with RAG context if available
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
            has_vision = _has_vision_capability(assistant)
            user_input_text = _extract_user_input_text(last_message)
            logger.debug(f"User message: {user_input_text}")

            augmented_text = _apply_template_with_rag_contexts(
                assistant.prompt_template,
                user_input_text,
                rag_context,
            )

            if isinstance(last_message, list) and has_vision:
                content = _build_multimodal_content(augmented_text, last_message)
            else:
                content = augmented_text

            processed_messages.append({
                "role": messages[-1]['role'],
                "content": content
            })
        else:
            # If no template, use original message
            processed_messages.append(messages[-1])

        return processed_messages

    # If no assistant provided, return original messages
    return messages
