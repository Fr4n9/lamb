// assistantFormSubmit.js
/**
 * Pure functions for AssistantForm submission logic.
 * Extracted from AssistantForm.svelte to enable isolated testing.
 */

import { isKbBasedRag, isSingleFileRag, isRubricRag } from '$lib/utils/ragProcessorHelpers.js';
import { buildToolsPayload, validateTools, syncActiveToolFromForm } from './multitoolState.svelte.js';

/**
 * Validates form data before submission.
 * @param {Record<string, any>} form
 * @returns {string | null} Error message or null if valid
 */
export function validateSubmission(form) {
	if (!form.name?.trim()) return 'Assistant Name is required.';

	if (form.tools?.length) {
		syncActiveToolFromForm(form);
		const toolsError = validateTools(form.tools);
		if (toolsError) return toolsError;
		return null;
	}

	if (isRubricRag(form.selectedRagProcessor) && !form.selectedRubricId) {
		return 'Please select a rubric when using Rubric RAG.';
	}

	return null;
}

/**
 * Builds the API payload from form state.
 * @param {Record<string, any>} form
 * @returns {Record<string, any>}
 */
export function buildAssistantPayload(form) {
	/** @type {Record<string, any>} */
	let toolsMeta;

	if (form.tools?.length) {
		syncActiveToolFromForm(form);
		toolsMeta = buildToolsPayload(form.tools);
	} else {
		toolsMeta = {
			rag_processor: form.selectedRagProcessor,
			RAG_Top_k: Number(form.RAG_Top_k) || 3,
			RAG_collections: isKbBasedRag(form.selectedRagProcessor)
				? form.selectedKnowledgeBases.join(',')
				: '',
			file_path: isSingleFileRag(form.selectedRagProcessor) ? form.selectedFilePath : '',
			rubric_id: form.selectedRubricId,
			rubric_format: form.rubricFormat,
			multitools: false
		};
	}

	/** @type {Record<string, any>} */
	const metadataObj = {
		prompt_processor: form.selectedPromptProcessor,
		connector: form.selectedConnector,
		llm: form.selectedLlm,
		rag_processor: toolsMeta.rag_processor,
		capabilities: {
			vision: form.visionEnabled,
			image_generation: form.imageGenerationEnabled
		}
	};

	if (toolsMeta.multitools) {
		metadataObj.multitools = true;
		metadataObj.tools = toolsMeta.tools;
	}

	if (isSingleFileRag(toolsMeta.rag_processor) && toolsMeta.file_path) {
		metadataObj.file_path = toolsMeta.file_path;
	}

	if (isRubricRag(toolsMeta.rag_processor)) {
		metadataObj.rubric_id = toolsMeta.rubric_id;
		metadataObj.rubric_format = toolsMeta.rubric_format;
	}

	return {
		name: form.name.trim(),
		description: form.description,
		system_prompt: form.system_prompt,
		prompt_template: form.prompt_template,
		RAG_Top_k: Number(toolsMeta.RAG_Top_k) || 3,
		RAG_collections: toolsMeta.RAG_collections || '',
		metadata: JSON.stringify(metadataObj),
		pre_retrieval_endpoint: '',
		post_retrieval_endpoint: '',
		RAG_endpoint: ''
	};
}
