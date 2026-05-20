// importAssistantValidator.test.js — Edge-case tests for modelExtractor and validation
import { describe, test, expect } from 'vitest';
import { validateImportedAssistant } from './logic/importAssistantValidator.js';
import { extractModelsFromConnectorData } from './logic/assistantFormUtils.svelte.js';

describe('modelExtractor edge cases', () => {
	const capabilities = {
		prompt_processors: ['default'],
		connectors: {
			openai: { models: ['gpt-4'] },
			empty_connector: null
		},
		rag_processors: ['no_rag']
	};

	test('handles modelExtractor returning empty for undefined connector data', () => {
		const json = JSON.stringify({
			name: 'Test',
			system_prompt: 'prompt',
			metadata: JSON.stringify({
				prompt_processor: 'default',
				connector: 'openai',
				llm: 'gpt-4',
				rag_processor: 'no_rag'
			})
		});
		const result = validateImportedAssistant(json, capabilities, extractModelsFromConnectorData);
		expect(result.hasErrors).toBe(false);
	});

	test('handles modelExtractor with null connector data gracefully', () => {
		const caps = {
			prompt_processors: ['default'],
			connectors: {
				openai: null
			},
			rag_processors: ['no_rag']
		};
		const json = JSON.stringify({
			name: 'Test',
			system_prompt: 'prompt',
			metadata: JSON.stringify({
				prompt_processor: 'default',
				connector: 'openai',
				llm: 'gpt-4',
				rag_processor: 'no_rag'
			})
		});
		const result = validateImportedAssistant(json, caps, extractModelsFromConnectorData);
		// null connector value is treated as invalid connector (not as retrieval failure)
		expect(result.validationLog.some(log => log.includes('Invalid connector'))).toBe(true);
	});

	test('handles modelExtractor with empty models array', () => {
		const caps = {
			prompt_processors: ['default'],
			connectors: {
				openai: { models: [] }
			},
			rag_processors: ['no_rag']
		};
		const json = JSON.stringify({
			name: 'Test',
			system_prompt: 'prompt',
			metadata: JSON.stringify({
				prompt_processor: 'default',
				connector: 'openai',
				llm: 'gpt-4',
				rag_processor: 'no_rag'
			})
		});
		const result = validateImportedAssistant(json, caps, extractModelsFromConnectorData);
		expect(result.hasErrors).toBe(false);
	});

	test('modelExtractor handles undefined', () => {
		expect(extractModelsFromConnectorData(undefined)).toEqual([]);
	});

	test('modelExtractor handles null', () => {
		expect(extractModelsFromConnectorData(null)).toEqual([]);
	});
});

describe('multitools import validation', () => {
	const mockCapabilities = {
		prompt_processors: ['simple_augment'],
		connectors: { openai: { models: ['gpt-4'] } },
		rag_processors: ['simple_rag', 'context_aware_rag', 'no_rag']
	};

	test('validates multitools metadata with tools array', () => {
		const json = JSON.stringify({
			name: 'test',
			system_prompt: 'hi',
			prompt_template: '{context} {context2} {user_input}',
			RAG_Top_k: 3,
			RAG_collections: 'col1',
			metadata: JSON.stringify({
				prompt_processor: 'simple_augment',
				connector: 'openai',
				llm: 'gpt-4',
				rag_processor: 'simple_rag',
				multitools: true,
				tools: [{ rag_processor: 'context_aware_rag', RAG_collections: 'col2', RAG_Top_k: 5 }]
			})
		});
		const result = validateImportedAssistant(json, mockCapabilities, extractModelsFromConnectorData);
		expect(result.hasErrors).toBe(false);
	});

	test('warns on invalid rag_processor in tools array', () => {
		const json = JSON.stringify({
			name: 'test',
			system_prompt: 'hi',
			metadata: JSON.stringify({
				prompt_processor: 'simple_augment',
				connector: 'openai',
				llm: 'gpt-4',
				rag_processor: 'simple_rag',
				multitools: true,
				tools: [{ rag_processor: 'nonexistent_rag' }]
			})
		});
		const result = validateImportedAssistant(json, mockCapabilities, extractModelsFromConnectorData);
		expect(result.validationLog.some((l) => l.includes('nonexistent_rag'))).toBe(true);
	});

	test('legacy format without multitools still works', () => {
		const json = JSON.stringify({
			name: 'test',
			system_prompt: 'hi',
			RAG_Top_k: 3,
			RAG_collections: 'col1',
			metadata: JSON.stringify({
				prompt_processor: 'simple_augment',
				connector: 'openai',
				llm: 'gpt-4',
				rag_processor: 'simple_rag'
			})
		});
		const result = validateImportedAssistant(json, mockCapabilities, extractModelsFromConnectorData);
		expect(result.hasErrors).toBe(false);
	});
});
