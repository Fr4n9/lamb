<script>
	import { onMount } from 'svelte';
	import { _ } from '$lib/i18n';
	import { Modal, Button, IconButton } from '$lib/components/ui';
	import { Plus, Trash2, Pencil, Save, X } from '$lib/components/ui/icons';
	import ConfirmationModal from '$lib/components/modals/ConfirmationModal.svelte';
	import { toast } from '$lib/stores/toast';
	import { user } from '$lib/stores/userStore';
	import {
		fetchModelPricing,
		createModelPricing,
		updateModelPricing,
		deleteModelPricing
	} from '$lib/services/adminService';

	let { onClose } = $props();

	let pricingList = $state([]);
	let isLoading = $state(true);
	let error = $state(null);

	let newProvider = $state('openai');
	let newModel = $state('');
	let newInput = $state('');
	let newCachedInput = $state('');
	let newOutput = $state('');
	let isSaving = $state(false);
	let deleteTarget = $state(null);

	let editingId = $state(null);
	let editInput = $state('');
	let editCachedInput = $state('');
	let editOutput = $state('');

	onMount(async () => {
		await loadPricing();
	});

	async function loadPricing() {
		isLoading = true;
		error = null;
		try {
			const token = $user?.token;
			const data = await fetchModelPricing(token);
			pricingList = data.pricing || [];
		} catch (e) {
			error = e?.message || 'Unknown error';
		} finally {
			isLoading = false;
		}
	}

	async function handleAdd() {
		if (!newModel.trim() || !newInput || !newOutput) return;
		isSaving = true;
		try {
			const token = $user?.token;
			await createModelPricing(token, {
				provider: newProvider,
				model_name: newModel.trim(),
				input_per_1m: parseFloat(newInput),
				cached_input_per_1m: newCachedInput ? parseFloat(newCachedInput) : null,
				output_per_1m: parseFloat(newOutput)
			});
			toast.success('Model pricing added');
			newModel = '';
			newInput = '';
			newCachedInput = '';
			newOutput = '';
			await loadPricing();
		} catch (e) {
			toast.error(e?.message || 'Failed to add pricing');
		} finally {
			isSaving = false;
		}
	}

	function startEdit(row) {
		editingId = row.id;
		editInput = String(row.input_per_1m);
		editCachedInput = row.cached_input_per_1m != null ? String(row.cached_input_per_1m) : '';
		editOutput = String(row.output_per_1m);
	}

	function cancelEdit() {
		editingId = null;
	}

	async function saveEdit() {
		if (!editingId || !editInput || !editOutput) return;
		try {
			const token = $user?.token;
			await updateModelPricing(token, editingId, {
				input_per_1m: parseFloat(editInput),
				cached_input_per_1m: editCachedInput ? parseFloat(editCachedInput) : null,
				output_per_1m: parseFloat(editOutput)
			});
			toast.success('Pricing updated');
			editingId = null;
			await loadPricing();
		} catch (e) {
			toast.error(e?.message || 'Failed to update pricing');
		}
	}

	async function confirmDelete() {
		if (!deleteTarget) return;
		try {
			const token = $user?.token;
			await deleteModelPricing(token, deleteTarget);
			toast.success('Pricing deleted');
			deleteTarget = null;
			await loadPricing();
		} catch (e) {
			toast.error(e?.message || 'Failed to delete pricing');
		}
	}
</script>

<Modal open={true} onclose={onClose} title={$_('admin.costManagement.pricing.title', { default: 'Manage model pricing' })} size="xl">
	<div class="flex-1 overflow-y-auto p-5">
		{#if isLoading}
			<p class="text-sm text-muted">{$_('admin.costManagement.pricing.loading', { default: 'Loading pricing...' })}</p>
		{:else if error}
			<p class="text-sm text-danger">{error}</p>
		{:else}
			<div class="text-xs text-muted mb-3 space-y-1">
				<p>{$_('admin.costManagement.pricing.helperText', { default: 'Model name must match the assistant LLM config. Changing prices affects new usage only.' })}</p>
				<p>{$_('admin.costManagement.pricing.cachedHelper', { default: 'Cached $/1M is the rate for prompt cache hits (OpenAI). Leave empty to bill all prompt tokens at the input rate.' })}</p>
				<p>{$_('admin.costManagement.pricing.providerHelper', { default: 'Billing provider (usually openai for OpenAI connector).' })}</p>
			</div>
			<table class="w-full text-sm mb-6">
				<thead>
					<tr class="text-left text-muted border-b border-border">
						<th class="pb-2 pr-2">Provider</th>
						<th class="pb-2 pr-2">Model</th>
						<th class="pb-2 pr-2 text-right">Input $/1M</th>
						<th class="pb-2 pr-2 text-right">Cached $/1M</th>
						<th class="pb-2 pr-2 text-right">Output $/1M</th>
						<th class="pb-2"></th>
					</tr>
				</thead>
				<tbody>
					{#each pricingList as row}
						{#if editingId === row.id}
							<tr class="border-b border-border/50 bg-surface-hover">
								<td class="py-1.5 pr-2">{row.provider}</td>
								<td class="py-1.5 pr-2 font-medium">{row.model_name}</td>
								<td class="py-1.5 pr-2">
									<input bind:value={editInput} type="number" step="0.001" class="w-20 border border-border rounded px-1.5 py-1 text-sm text-right focus:ring-2 focus:ring-brand focus:outline-none" />
								</td>
								<td class="py-1.5 pr-2">
									<input bind:value={editCachedInput} type="number" step="0.001" class="w-20 border border-border rounded px-1.5 py-1 text-sm text-right focus:ring-2 focus:ring-brand focus:outline-none" placeholder="—" />
								</td>
								<td class="py-1.5 pr-2">
									<input bind:value={editOutput} type="number" step="0.001" class="w-20 border border-border rounded px-1.5 py-1 text-sm text-right focus:ring-2 focus:ring-brand focus:outline-none" />
								</td>
								<td class="py-1.5 flex gap-1">
									<IconButton icon={Save} variant="ghost" size="sm" ariaLabel="Save" onclick={saveEdit} />
									<IconButton icon={X} variant="ghost" size="sm" ariaLabel="Cancel" onclick={cancelEdit} />
								</td>
							</tr>
						{:else}
							<tr class="border-b border-border/50">
								<td class="py-1.5 pr-2">{row.provider}</td>
								<td class="py-1.5 pr-2 font-medium">{row.model_name}</td>
								<td class="py-1.5 pr-2 text-right">{row.input_per_1m}</td>
								<td class="py-1.5 pr-2 text-right">{row.cached_input_per_1m ?? '—'}</td>
								<td class="py-1.5 pr-2 text-right">{row.output_per_1m}</td>
								<td class="py-1.5 flex gap-1">
									<IconButton icon={Pencil} variant="ghost" size="sm" ariaLabel="Edit" onclick={() => startEdit(row)} />
									<IconButton icon={Trash2} variant="danger-ghost" size="sm" ariaLabel="Delete" onclick={() => deleteTarget = row.id} />
								</td>
							</tr>
						{/if}
					{/each}
				</tbody>
			</table>
		{/if}

		<div class="border-t border-border pt-4">
			<h3 class="text-sm font-semibold mb-3">{$_('admin.costManagement.pricing.addModel', { default: 'Add model' })}</h3>
			<div class="grid grid-cols-5 gap-2">
				<input bind:value={newProvider} placeholder="Provider" class="border border-border rounded-lg px-2 py-1.5 text-sm focus:ring-2 focus:ring-brand focus:outline-none" />
				<input bind:value={newModel} placeholder="Model name" class="border border-border rounded-lg px-2 py-1.5 text-sm focus:ring-2 focus:ring-brand focus:outline-none" />
				<input bind:value={newInput} placeholder="Input $/1M" type="number" step="0.001" class="border border-border rounded-lg px-2 py-1.5 text-sm focus:ring-2 focus:ring-brand focus:outline-none" />
				<input bind:value={newCachedInput} placeholder="Cached $/1M" type="number" step="0.001" class="border border-border rounded-lg px-2 py-1.5 text-sm focus:ring-2 focus:ring-brand focus:outline-none" />
				<input bind:value={newOutput} placeholder="Output $/1M" type="number" step="0.001" class="border border-border rounded-lg px-2 py-1.5 text-sm focus:ring-2 focus:ring-brand focus:outline-none" />
			</div>
			<Button variant="primary" onclick={handleAdd} disabled={isSaving || !newModel.trim() || !newInput || !newOutput} iconLeftComponent={Plus} class="mt-3">
				{isSaving ? '...' : $_('admin.costManagement.pricing.addButton', { default: 'Add pricing' })}
			</Button>
		</div>
	</div>
</Modal>

<ConfirmationModal
	isOpen={deleteTarget !== null}
	variant="danger"
	title={$_('admin.costManagement.pricing.deleteTitle', { default: 'Delete pricing row' })}
	message={$_('admin.costManagement.pricing.deleteMessage', { default: 'Are you sure you want to delete this pricing row?' })}
	onconfirm={confirmDelete}
	oncancel={() => deleteTarget = null}
/>
