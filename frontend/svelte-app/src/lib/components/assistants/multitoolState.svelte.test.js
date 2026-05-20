import { describe, test, expect } from 'vitest';
import {
	createToolConfig,
	addTool,
	removeTool,
	reindexPlaceholders,
	buildToolsPayload,
	validateTools,
	toolsFromMetadata,
	getMissingPlaceholders,
	MAX_TOOLS,
	TOOL_COLORS
} from './logic/multitoolState.svelte.js';

describe('createToolConfig', () => {
	test('creates a default tool config with correct index', () => {
		const tool = createToolConfig(0);
		expect(tool.index).toBe(0);
		expect(tool.ragProcessor).toBe('');
		expect(tool.RAG_Top_k).toBe(3);
		expect(tool.knowledgeBases).toEqual([]);
		expect(tool.contextKey).toBe('context');
	});

	test('creates tool with correct context key for index > 0', () => {
		const tool = createToolConfig(2);
		expect(tool.contextKey).toBe('context3');
	});
});

describe('addTool', () => {
	test('adds a tool to the array', () => {
		const tools = [createToolConfig(0)];
		const result = addTool(tools);
		expect(result).toHaveLength(2);
		expect(result[1].index).toBe(1);
		expect(result[1].contextKey).toBe('context2');
	});

	test('returns null when at max tools', () => {
		const tools = Array.from({ length: MAX_TOOLS }, (_, i) => createToolConfig(i));
		expect(addTool(tools)).toBeNull();
	});
});

describe('removeTool', () => {
	test('removes a tool and reindexes', () => {
		const tools = [createToolConfig(0), createToolConfig(1), createToolConfig(2)];
		tools[1].ragProcessor = 'simple_rag';
		tools[2].ragProcessor = 'context_aware_rag';
		const result = removeTool(tools, 1);
		expect(result).toHaveLength(2);
		expect(result[0].index).toBe(0);
		expect(result[1].index).toBe(1);
		expect(result[1].ragProcessor).toBe('context_aware_rag');
		expect(result[1].contextKey).toBe('context2');
	});

	test('cannot remove tool 0', () => {
		const tools = [createToolConfig(0), createToolConfig(1)];
		const result = removeTool(tools, 0);
		expect(result).toHaveLength(2);
	});
});

describe('reindexPlaceholders', () => {
	test('renames placeholders after tool deletion', () => {
		const template = 'Use {context} and {context2} and {context3} to answer {user_input}';
		const result = reindexPlaceholders(template, 1, 3);
		expect(result).toBe('Use {context} and  and {context2} to answer {user_input}');
	});

	test('no change when no orphaned placeholders', () => {
		const template = '{context} {context2} {user_input}';
		const result = reindexPlaceholders(template, 2, 2);
		expect(result).toBe('{context} {context2} {user_input}');
	});
});

describe('buildToolsPayload', () => {
	test('single tool returns no multitools fields', () => {
		const tools = [createToolConfig(0)];
		tools[0].ragProcessor = 'simple_rag';
		tools[0].knowledgeBases = ['col1'];
		tools[0].RAG_Top_k = 5;
		const result = buildToolsPayload(tools);
		expect(result.multitools).toBe(false);
		expect(result.tools).toBeUndefined();
		expect(result.rag_processor).toBe('simple_rag');
		expect(result.RAG_collections).toBe('col1');
		expect(result.RAG_Top_k).toBe(5);
	});

	test('multiple tools returns multitools payload', () => {
		const tools = [createToolConfig(0), createToolConfig(1)];
		tools[0].ragProcessor = 'simple_rag';
		tools[0].knowledgeBases = ['col1'];
		tools[0].RAG_Top_k = 3;
		tools[1].ragProcessor = 'context_aware_rag';
		tools[1].knowledgeBases = ['col2'];
		tools[1].RAG_Top_k = 7;
		const result = buildToolsPayload(tools);
		expect(result.multitools).toBe(true);
		expect(result.tools).toHaveLength(1);
		expect(result.tools[0].rag_processor).toBe('context_aware_rag');
		expect(result.tools[0].RAG_Top_k).toBe(7);
		expect(result.RAG_Top_k).toBe(3);
	});
});

describe('validateTools', () => {
	test('valid single tool passes', () => {
		const tools = [createToolConfig(0)];
		tools[0].ragProcessor = 'simple_rag';
		tools[0].knowledgeBases = ['col1'];
		expect(validateTools(tools)).toBeNull();
	});

	test('unconfigured additional tool fails', () => {
		const tools = [createToolConfig(0), createToolConfig(1)];
		tools[0].ragProcessor = 'simple_rag';
		tools[0].knowledgeBases = ['col1'];
		const error = validateTools(tools);
		expect(error).not.toBeNull();
	});

	test('KB-based rag with no KBs fails', () => {
		const tools = [createToolConfig(0), createToolConfig(1)];
		tools[0].ragProcessor = 'simple_rag';
		tools[0].knowledgeBases = ['col1'];
		tools[1].ragProcessor = 'simple_rag';
		tools[1].knowledgeBases = [];
		const error = validateTools(tools);
		expect(error).not.toBeNull();
	});
});

describe('toolsFromMetadata', () => {
	test('creates tools array from legacy metadata', () => {
		const assistant = {
			RAG_Top_k: 5,
			RAG_collections: 'col1,col2'
		};
		const metadata = {
			rag_processor: 'simple_rag',
			rubric_id: ''
		};
		const tools = toolsFromMetadata(assistant, metadata);
		expect(tools).toHaveLength(1);
		expect(tools[0].ragProcessor).toBe('simple_rag');
		expect(tools[0].knowledgeBases).toEqual(['col1', 'col2']);
		expect(tools[0].RAG_Top_k).toBe(5);
	});

	test('creates tools array from multitools metadata', () => {
		const assistant = {
			RAG_Top_k: 3,
			RAG_collections: 'col1'
		};
		const metadata = {
			rag_processor: 'simple_rag',
			multitools: true,
			tools: [
				{ rag_processor: 'context_aware_rag', RAG_collections: 'col2,col3', RAG_Top_k: 7 },
				{ rag_processor: 'rubric_rag', rubric_id: 'rubric-1', rubric_format: 'json' }
			]
		};
		const tools = toolsFromMetadata(assistant, metadata);
		expect(tools).toHaveLength(3);
		expect(tools[0].ragProcessor).toBe('simple_rag');
		expect(tools[1].ragProcessor).toBe('context_aware_rag');
		expect(tools[1].RAG_Top_k).toBe(7);
		expect(tools[2].ragProcessor).toBe('rubric_rag');
		expect(tools[2].rubricId).toBe('rubric-1');
	});
});

describe('getMissingPlaceholders', () => {
	test('returns tools without matching placeholder', () => {
		const tools = [createToolConfig(0), createToolConfig(1)];
		const missing = getMissingPlaceholders(tools, '{context} {user_input}');
		expect(missing).toHaveLength(1);
		expect(missing[0].contextKey).toBe('context2');
	});
});

describe('TOOL_COLORS', () => {
	test('has 5 colors', () => {
		expect(TOOL_COLORS).toHaveLength(5);
	});
});
