import { describe, test, expect, vi } from 'vitest';

vi.mock('$lib/utils/ragProcessorHelpers.js', () => ({
	isKbBasedRag: (p) => ['simple_rag', 'context_aware_rag', 'hierarchical_rag'].includes(p),
	isSingleFileRag: (p) => p === 'single_file_rag',
	isRubricRag: (p) => p === 'rubric_rag'
}));

import { validateSubmission, buildAssistantPayload } from './logic/assistantFormSubmit.js';
import { createToolConfig } from './logic/multitoolState.svelte.js';

function mockTool(index, ragProcessor, knowledgeBases = [], topK = 3) {
	const tool = createToolConfig(index);
	tool.ragProcessor = ragProcessor;
	tool.knowledgeBases = knowledgeBases;
	tool.RAG_Top_k = topK;
	return tool;
}

function createMockForm(overrides = {}) {
	return {
		name: 'test',
		description: '',
		system_prompt: '',
		prompt_template: '',
		RAG_Top_k: 3,
		selectedPromptProcessor: 'default',
		selectedConnector: 'openai',
		selectedLlm: 'gpt-4',
		selectedRagProcessor: 'no_rag',
		selectedFilePath: '',
		visionEnabled: false,
		imageGenerationEnabled: false,
		selectedKnowledgeBases: [],
		selectedRubricId: '',
		rubricFormat: 'markdown',
		activeToolIndex: 0,
		tools: [mockTool(0, 'no_rag')],
		...overrides
	};
}

describe('validateSubmission', () => {
	test('returns error when name is empty', () => {
		const result = validateSubmission({ name: '', selectedRagProcessor: 'no_rag', selectedRubricId: '' });
		expect(result).toContain('Name');
	});

	test('returns error when rubric_rag selected without rubric', () => {
		const result = validateSubmission({ name: 'test', selectedRagProcessor: 'rubric_rag', selectedRubricId: '' });
		expect(result).toContain('rubric');
	});

	test('returns null when valid', () => {
		const result = validateSubmission(createMockForm({ tools: [mockTool(0, 'no_rag')] }));
		expect(result).toBeNull();
	});

	test('unconfigured tool returns error', () => {
		const form = createMockForm({
			tools: [mockTool(0, 'simple_rag', ['col1']), mockTool(1, '', [])]
		});
		const error = validateSubmission(form);
		expect(error).not.toBeNull();
	});
});

describe('buildAssistantPayload', () => {
	test('builds payload with metadata', () => {
		const payload = buildAssistantPayload(createMockForm({
			name: ' test ',
			description: 'desc',
			system_prompt: 'sys',
			prompt_template: 'tmpl',
			tools: [mockTool(0, 'no_rag')]
		}));
		expect(payload.name).toBe('test');
		expect(JSON.parse(payload.metadata).connector).toBe('openai');
	});

	test('includes rubric fields when rubric_rag is selected', () => {
		const payload = buildAssistantPayload(createMockForm({
			selectedRagProcessor: 'rubric_rag',
			selectedRubricId: 'rubric-123',
			rubricFormat: 'json',
			tools: [(() => {
				const t = mockTool(0, 'rubric_rag');
				t.rubricId = 'rubric-123';
				t.rubricFormat = 'json';
				return t;
			})()]
		}));
		const metadata = JSON.parse(payload.metadata);
		expect(metadata.rubric_id).toBe('rubric-123');
		expect(metadata.rubric_format).toBe('json');
	});

	test('includes KB collections when kb-based RAG is selected', () => {
		const payload = buildAssistantPayload(createMockForm({
			RAG_Top_k: 5,
			selectedRagProcessor: 'simple_rag',
			visionEnabled: true,
			selectedKnowledgeBases: ['kb1', 'kb2'],
			tools: [mockTool(0, 'simple_rag', ['kb1', 'kb2'], 5)]
		}));
		expect(payload.RAG_collections).toBe('kb1,kb2');
		expect(payload.RAG_Top_k).toBe(5);
		const metadata = JSON.parse(payload.metadata);
		expect(metadata.capabilities.vision).toBe(true);
	});

	test('multiple tools produces multitools payload', () => {
		const payload = buildAssistantPayload(createMockForm({
			tools: [
				mockTool(0, 'simple_rag', ['col1'], 3),
				mockTool(1, 'context_aware_rag', ['col2'], 7)
			]
		}));
		const metadata = JSON.parse(payload.metadata);
		expect(metadata.multitools).toBe(true);
		expect(metadata.tools).toHaveLength(1);
		expect(metadata.tools[0].rag_processor).toBe('context_aware_rag');
		expect(metadata.tools[0].RAG_Top_k).toBe(7);
		expect(payload.RAG_Top_k).toBe(3);
	});
});
