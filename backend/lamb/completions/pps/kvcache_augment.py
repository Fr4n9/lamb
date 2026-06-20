import json
import os
from typing import Any, Dict, List, Optional

from lamb.lamb_classes import Assistant
from lamb.logging_config import get_logger

logger = get_logger(__name__, component="MAIN")

# system (default): document in system prompt (Dual Tool, cacheable prefix)
# user_template (load-test baseline): document prepended to {context} in user message
_DOC_PLACEMENT = os.getenv("LAMB_KVCACHE_DOCUMENT_PLACEMENT", "system").strip().lower()

COMPATIBLE_RAG = [
    "library_file_rag",
    "knowledge_store_rag",
    "query_rewriting_ks_rag",
    "rubric_rag",
    "no_rag",
]

DEFAULT_RAG_PROMPT_TEMPLATE = (
    "Use the following context to answer the question. "
    "If the context does not contain the answer, say you do not know.\n\n"
    "Context:\n{context}\n\nQuestion: {user_input}"
)


def _document_in_user_template() -> bool:
    return _DOC_PLACEMENT in ("user_template", "user", "template")


def _format_reference_document(doc_text: str) -> str:
    return (
        "\n\n## REFERENCE DOCUMENT\n\n"
        "This document has been selected by the assistant creator as a reference "
        "that will likely be useful for many queries, as it is generally a helpful "
        "document. Use it as context when answering questions.\n\n"
        f"{doc_text}"
        "\n\nIMPORTANT: The reference document above is available for this entire "
        "conversation. Always consider it alongside any retrieved context when "
        "answering questions. If the user's question relates to the document's "
        "content, use it."
    )


def _labeled_document_from_context(
    document_context: Optional[Dict[str, Any]],
) -> str:
    if not document_context or not isinstance(document_context, dict):
        return ""
    doc_text = document_context.get("context", "")
    if not doc_text:
        return ""
    return _format_reference_document(doc_text)


def _build_rag_context_block(rag_context: Optional[Dict[str, Any]]) -> str:
    if not rag_context:
        return ""
    context = (
        rag_context.get("context", "")
        if isinstance(rag_context, dict)
        else str(rag_context)
    )
    sources_text = ""
    if isinstance(rag_context, dict) and rag_context.get("sources"):
        sources = rag_context["sources"]
        if sources:
            sources_text = "\n\n## Available Sources\n\n"
            for i, source in enumerate(sources, 1):
                title = source.get("title", "Unknown")
                url = source.get("url", "")
                similarity = source.get("similarity", 0)
                sources_text += f"{i}. [{title}]({url}) (similarity: {similarity:.3f})\n"
    return context + sources_text


def _context_for_template(
    rag_context: Optional[Dict[str, Any]],
    document_context: Optional[Dict[str, Any]],
) -> str:
    rag_block = _build_rag_context_block(rag_context)
    if _document_in_user_template():
        labeled_doc = _labeled_document_from_context(document_context)
        if labeled_doc:
            return labeled_doc + ("\n\n" + rag_block if rag_block else "")
    return rag_block


def _system_content_with_document(
    assistant: Assistant,
    document_context: Optional[Dict[str, Any]],
) -> str:
    system_content = assistant.system_prompt or ""
    if _document_in_user_template():
        return system_content
    labeled_doc = _labeled_document_from_context(document_context)
    if labeled_doc:
        system_content = (
            (labeled_doc + "\n\n" + system_content) if system_content else labeled_doc
        )
    return system_content


def _has_vision_capability(assistant: Assistant) -> bool:
    if not assistant:
        return False
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
    if not assistant:
        return False
    metadata_str = getattr(assistant, 'metadata', None) or getattr(assistant, 'api_callback', None)
    if not metadata_str:
        return False
    try:
        metadata = json.loads(metadata_str)
        capabilities = metadata.get('capabilities', {})
        return capabilities.get('image_generation', False)
    except (json.JSONDecodeError, AttributeError):
        return False


def _replace_context_placeholder(template: str, context_block: str) -> str:
    if context_block:
        return template.replace("{context}", "\n\n" + context_block + "\n\n")
    return template.replace("{context}", "")


def prompt_processor(
    request: Dict[str, Any],
    assistant: Optional[Assistant] = None,
    rag_context: Optional[Dict[str, Any]] = None,
    document_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    messages = request.get('messages', [])
    if not messages:
        return messages

    last_message = messages[-1]['content']
    processed_messages = []

    if assistant:
        system_content = _system_content_with_document(assistant, document_context)
        if system_content:
            processed_messages.append({
                "role": "system",
                "content": system_content
            })

        processed_messages.extend(messages[:-1])

        context_block = _context_for_template(rag_context, document_context)

        if assistant.prompt_template:
            has_vision = _has_vision_capability(assistant)

            if isinstance(last_message, list) and has_vision:
                text_parts = []
                for item in last_message:
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                user_input_text = ' '.join(text_parts)

                logger.debug(f"User message: {user_input_text}")
                augmented_text = assistant.prompt_template.replace(
                    "{user_input}", "\n\n" + user_input_text + "\n\n"
                )
                augmented_text = _replace_context_placeholder(augmented_text, context_block)

                augmented_content = [{"type": "text", "text": augmented_text}]
                for item in last_message:
                    if item.get('type') != 'text':
                        augmented_content.append(item)

                processed_messages.append({
                    "role": messages[-1]['role'],
                    "content": augmented_content
                })
            else:
                if isinstance(last_message, list):
                    text_parts = []
                    for item in last_message:
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                    user_input_text = ' '.join(text_parts)
                else:
                    user_input_text = str(last_message)

                logger.debug(f"User message: {user_input_text}")
                prompt = assistant.prompt_template.replace(
                    "{user_input}", "\n\n" + user_input_text + "\n\n"
                )
                prompt = _replace_context_placeholder(prompt, context_block)

                processed_messages.append({
                    "role": messages[-1]['role'],
                    "content": prompt
                })
        else:
            effective_template = None
            if rag_context or (
                _document_in_user_template() and _labeled_document_from_context(document_context)
            ):
                rag_text = _build_rag_context_block(rag_context)
                if rag_text or _document_in_user_template():
                    effective_template = DEFAULT_RAG_PROMPT_TEMPLATE

            if effective_template:
                if isinstance(last_message, list):
                    text_parts = []
                    for item in last_message:
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                    user_input_text = ' '.join(text_parts)
                else:
                    user_input_text = str(last_message)

                prompt = effective_template.replace(
                    "{user_input}", "\n\n" + user_input_text + "\n\n"
                )
                prompt = _replace_context_placeholder(prompt, context_block)

                processed_messages.append({
                    "role": messages[-1]['role'],
                    "content": prompt
                })
            else:
                processed_messages.append(messages[-1])

        return processed_messages

    return messages
