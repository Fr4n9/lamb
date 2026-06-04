import { describe, it, expect, vi, beforeEach } from 'vitest';
import '@testing-library/jest-dom/vitest';
import { render, waitFor } from '@testing-library/svelte';
import AssistantUsageBreakdown from './AssistantUsageBreakdown.svelte';

vi.mock('$lib/services/adminService', () => ({
	fetchAssistantUsageByModel: vi.fn()
}));
vi.mock('$lib/stores/userStore', () => ({
	user: { subscribe: vi.fn((fn) => { fn({ isLoggedIn: true, token: 'tok' }); return vi.fn(); }) }
}));
vi.mock('$lib/i18n', async () => {
	const { readable } = await import('svelte/store');
	return { _: readable((k, o) => o?.default || k), locale: readable('en'), waitLocale: vi.fn(), setupI18n: vi.fn() };
});

import { fetchAssistantUsageByModel } from '$lib/services/adminService';

describe('AssistantUsageBreakdown', () => {
	beforeEach(() => vi.clearAllMocks());

	it('fetches and displays per-model breakdown', async () => {
		fetchAssistantUsageByModel.mockResolvedValueOnce({
			assistant_id: 1,
			breakdown: [
				{
					provider: 'openai', model_name: 'gpt-4o',
					prompt_tokens: 12000, cached_prompt_tokens: 9000,
					non_cached_prompt_tokens: 3000, completion_tokens: 8000,
					total_tokens: 20000, cost_usd: 0.42, request_count: 85,
					pricing: { input_per_1m: 2.5, cached_input_per_1m: 1.25, output_per_1m: 10.0 }
				}
			]
		});

		const { getByText } = render(AssistantUsageBreakdown, { props: { assistantId: 1 } });
		await waitFor(() => {
			expect(getByText('gpt-4o')).toBeInTheDocument();
			expect(getByText('85')).toBeInTheDocument();
		});
	});

	it('shows error state on fetch failure', async () => {
		fetchAssistantUsageByModel.mockRejectedValueOnce(new Error('Server error'));
		const { getByText } = render(AssistantUsageBreakdown, { props: { assistantId: 1 } });
		await waitFor(() => {
			expect(getByText(/Server error/)).toBeInTheDocument();
		});
	});
});
