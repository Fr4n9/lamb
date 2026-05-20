/**
 * Multitool context source state management.
 *
 * Pure functions for managing per-assistant tool configurations.
 * Tool 0 maps to top-level assistant fields; tools 1-4 live in metadata.tools[].
 */

import {
	isKbBasedRag,
	isRubricRag,
	isSingleFileRag
} from '$lib/utils/ragProcessorHelpers.js';

export const MAX_TOOLS = 5;

/** @type {readonly string[]} */
export const TOOL_COLORS = Object.freeze([
	'#3B82F6',
	'#22C55E',
	'#F97316',
	'#8B5CF6',
	'#14B8A6'
]);

/**
 * @typedef {Object} ToolConfig
 * @property {number} index
 * @property {string} contextKey
 * @property {string} ragProcessor
 * @property {number} RAG_Top_k
 * @property {string[]} knowledgeBases
 * @property {string} filePath
 * @property {string} rubricId
 * @property {string} rubricFormat
 */

/**
 * @param {number} index
 * @returns {ToolConfig}
 */
export function createToolConfig(index) {
	return {
		index,
		contextKey: index === 0 ? 'context' : `context${index + 1}`,
		ragProcessor: '',
		RAG_Top_k: 3,
		knowledgeBases: [],
		filePath: '',
		rubricId: '',
		rubricFormat: 'markdown'
	};
}

/**
 * @param {ToolConfig[]} tools
 * @returns {ToolConfig[] | null}
 */
export function addTool(tools) {
	if (tools.length >= MAX_TOOLS) {
		return null;
	}
	return [...tools, createToolConfig(tools.length)];
}

/**
 * @param {ToolConfig[]} tools
 * @param {number} index
 * @returns {ToolConfig[]}
 */
export function removeTool(tools, index) {
	if (index <= 0 || index >= tools.length) {
		return tools;
	}
	const remaining = tools.filter((_, i) => i !== index);
	return remaining.map((tool, i) => ({
		...tool,
		index: i,
		contextKey: i === 0 ? 'context' : `context${i + 1}`
	}));
}

/**
 * Re-index placeholders in prompt template after a tool is removed.
 *
 * @param {string} template
 * @param {number} removedIndex - 1-based index of removed tool (1 = context2)
 * @param {number} oldToolCount - total tools before removal
 * @returns {string}
 */
export function reindexPlaceholders(template, removedIndex, oldToolCount) {
	if (!template) return template;
	let result = template;

	// Use temp tokens to avoid collisions when shifting placeholders down
	for (let i = removedIndex + 1; i < oldToolCount; i++) {
		const oldKey = i === 0 ? 'context' : `context${i + 1}`;
		result = result.replaceAll(`{${oldKey}}`, `__TEMP_${i}__`);
	}

	const deletedKey = removedIndex === 0 ? 'context' : `context${removedIndex + 1}`;
	result = result.replaceAll(`{${deletedKey}}`, '');

	for (let i = removedIndex + 1; i < oldToolCount; i++) {
		const newKey = i - 1 === 0 ? 'context' : `context${i}`;
		result = result.replaceAll(`__TEMP_${i}__`, `{${newKey}}`);
	}

	return result;
}

/**
 * @param {ToolConfig} tool
 * @returns {string | null}
 */
function validateSingleTool(tool) {
	if (!tool.ragProcessor) {
		return `Context Source ${tool.index + 1} is not fully configured`;
	}
	if (isKbBasedRag(tool.ragProcessor) && tool.knowledgeBases.length === 0) {
		return `Context Source ${tool.index + 1} is not fully configured`;
	}
	if (isSingleFileRag(tool.ragProcessor) && !tool.filePath) {
		return `Context Source ${tool.index + 1} is not fully configured`;
	}
	if (isRubricRag(tool.ragProcessor) && !tool.rubricId) {
		return `Context Source ${tool.index + 1} is not fully configured`;
	}
	return null;
}

/**
 * @param {ToolConfig[]} tools
 * @returns {string | null}
 */
export function validateTools(tools) {
	for (const tool of tools) {
		const error = validateSingleTool(tool);
		if (error) return error;
	}
	return null;
}

/**
 * @param {ToolConfig} tool
 * @returns {Record<string, any>}
 */
function toolToMetadataEntry(tool) {
	/** @type {Record<string, any>} */
	const entry = {
		rag_processor: tool.ragProcessor
	};
	if (isKbBasedRag(tool.ragProcessor)) {
		entry.RAG_collections = tool.knowledgeBases.join(',');
		if (tool.RAG_Top_k !== 3) {
			entry.RAG_Top_k = tool.RAG_Top_k;
		}
	}
	if (isSingleFileRag(tool.ragProcessor)) {
		entry.file_path = tool.filePath;
	}
	if (isRubricRag(tool.ragProcessor)) {
		entry.rubric_id = tool.rubricId;
		entry.rubric_format = tool.rubricFormat;
	}
	return entry;
}

/**
 * Build metadata fields for tool 0 and optional tools[] array.
 *
 * @param {ToolConfig[]} tools
 * @returns {Record<string, any>}
 */
export function buildToolsPayload(tools) {
	const tool0 = tools[0];
	/** @type {Record<string, any>} */
	const result = {
		rag_processor: tool0.ragProcessor,
		RAG_Top_k: tool0.RAG_Top_k,
		RAG_collections: isKbBasedRag(tool0.ragProcessor) ? tool0.knowledgeBases.join(',') : ''
	};

	if (isSingleFileRag(tool0.ragProcessor)) {
		result.file_path = tool0.filePath;
	}
	if (isRubricRag(tool0.ragProcessor)) {
		result.rubric_id = tool0.rubricId;
		result.rubric_format = tool0.rubricFormat;
	}

	if (tools.length <= 1) {
		result.multitools = false;
		return result;
	}

	result.multitools = true;
	result.tools = tools.slice(1).map(toolToMetadataEntry);
	return result;
}

/**
 * @param {Record<string, any>} assistant
 * @param {Record<string, any>} metadata
 * @returns {ToolConfig[]}
 */
export function toolsFromMetadata(assistant, metadata) {
	/** @type {ToolConfig[]} */
	const tools = [];

	const tool0 = createToolConfig(0);
	tool0.ragProcessor = metadata.rag_processor || '';
	tool0.RAG_Top_k = assistant.RAG_Top_k ?? 3;
	tool0.knowledgeBases = assistant.RAG_collections?.split(',').filter(Boolean) || [];
	tool0.filePath = metadata.file_path || '';
	tool0.rubricId = metadata.rubric_id || '';
	tool0.rubricFormat = metadata.rubric_format || 'markdown';
	tools.push(tool0);

	if (metadata.multitools && Array.isArray(metadata.tools)) {
		for (let i = 0; i < metadata.tools.length; i++) {
			const entry = metadata.tools[i];
			const tool = createToolConfig(i + 1);
			tool.ragProcessor = entry.rag_processor || '';
			tool.RAG_Top_k = entry.RAG_Top_k ?? 3;
			tool.knowledgeBases = entry.RAG_collections?.split(',').filter(Boolean) || [];
			tool.filePath = entry.file_path || '';
			tool.rubricId = entry.rubric_id || '';
			tool.rubricFormat = entry.rubric_format || 'markdown';
			tools.push(tool);
		}
	}

	return tools;
}

/**
 * @param {ToolConfig[]} tools
 * @param {string} promptTemplate
 * @returns {Array<{ index: number, contextKey: string }>}
 */
export function getMissingPlaceholders(tools, promptTemplate) {
	if (tools.length <= 1) return [];
	const template = promptTemplate || '';
	return tools
		.filter((tool) => !template.includes(`{${tool.contextKey}}`))
		.map((tool) => ({ index: tool.index, contextKey: tool.contextKey }));
}

/**
 * Build placeholder button labels for the prompt template UI.
 *
 * @param {ToolConfig[]} tools
 * @returns {Array<{ label: string, placeholder: string, color: string }>}
 */
export function buildContextPlaceholderButtons(tools) {
	return tools.map((tool) => ({
		label: `Context ${tool.index + 1}`,
		placeholder: `{${tool.contextKey}}`,
		color: TOOL_COLORS[tool.index] ?? TOOL_COLORS[0]
	}));
}
