<!-- src/lib/components/assistants/components/ContextSourceTabs.svelte -->
<script>
	import { _ } from '$lib/i18n';
	import { MAX_TOOLS, TOOL_COLORS } from '../logic/multitoolState.svelte.js';

	let {
		tools = [],
		activeToolIndex = $bindable(0),
		formState = 'create',
		onAddTool,
		onRemoveTool
	} = $props();

	let confirmingDelete = $state(-1);
	let canModifyTools = $derived(formState === 'create');

	function handleTabClick(index) {
		activeToolIndex = index;
		confirmingDelete = -1;
	}

	function handleAddClick() {
		if (!canModifyTools || tools.length >= MAX_TOOLS) return;
		onAddTool?.();
	}

	function handleDeleteClick(index, event) {
		event.stopPropagation();
		if (!canModifyTools || index === 0) return;
		confirmingDelete = index;
	}

	function confirmDelete(index) {
		onRemoveTool?.(index);
		confirmingDelete = -1;
		if (activeToolIndex >= tools.length - 1) {
			activeToolIndex = Math.max(0, tools.length - 2);
		}
	}

	function cancelDelete() {
		confirmingDelete = -1;
	}
</script>

<div class="mb-4">
	<h4 class="text-md font-medium text-gray-700 mb-2">
		{$_('assistants.form.contextSources.title', { default: 'Context Sources' })}
	</h4>

	{#if confirmingDelete >= 0}
		<div class="p-3 bg-amber-50 border border-amber-200 rounded-md mb-2">
			<p class="text-sm text-amber-800 mb-2">
				{$_('assistants.form.contextSources.deleteConfirm', {
					values: { number: confirmingDelete + 1 },
					default: `Remove Context Source ${confirmingDelete + 1}? Placeholders in the prompt template will be updated automatically.`
				})}
			</p>
			<div class="flex gap-2">
				<button
					type="button"
					class="px-3 py-1 text-xs font-medium text-white bg-red-600 rounded hover:bg-red-700"
					onclick={() => confirmDelete(confirmingDelete)}
				>
					{$_('assistants.form.contextSources.deleteConfirmButton', { default: 'Remove' })}
				</button>
				<button
					type="button"
					class="px-3 py-1 text-xs font-medium text-gray-700 bg-white border border-gray-300 rounded hover:bg-gray-50"
					onclick={cancelDelete}
				>
					{$_('assistants.form.contextSources.deleteCancel', { default: 'Cancel' })}
				</button>
			</div>
		</div>
	{/if}

	<div class="flex items-center gap-1 flex-wrap">
		{#each tools as tool, index (tool.index)}
			<div class="relative flex items-center">
				<button
					type="button"
					class="px-3 py-1.5 text-sm font-medium rounded-l-md border transition-colors"
					class:rounded-r-md={index === 0 || !canModifyTools}
					style:background-color={activeToolIndex === index ? TOOL_COLORS[index] : `${TOOL_COLORS[index]}20`}
					style:color={activeToolIndex === index ? '#ffffff' : TOOL_COLORS[index]}
					style:border-color={TOOL_COLORS[index]}
					onclick={() => handleTabClick(index)}
				>
					{$_('assistants.form.contextSources.tab', {
						values: { number: index + 1 },
						default: `Context ${index + 1}`
					})}
				</button>
				{#if canModifyTools && index > 0}
					<button
						type="button"
						class="px-1.5 py-1.5 text-xs rounded-r-md border border-l-0 hover:bg-red-50 text-gray-500 hover:text-red-600"
						style:border-color={TOOL_COLORS[index]}
						title={$_('assistants.form.contextSources.deleteTool', { default: 'Remove' })}
						onclick={(e) => handleDeleteClick(index, e)}
					>
						&times;
					</button>
				{/if}
			</div>
		{/each}

		{#if canModifyTools}
			<button
				type="button"
				class="px-2 py-1.5 text-lg font-medium text-gray-500 border border-dashed border-gray-300 rounded-md hover:bg-gray-50 hover:text-brand disabled:opacity-40 disabled:cursor-not-allowed"
				disabled={tools.length >= MAX_TOOLS}
				title={tools.length >= MAX_TOOLS
					? $_('assistants.form.contextSources.maxReached', {
							values: { max: MAX_TOOLS },
							default: `Maximum of ${MAX_TOOLS} context sources reached`
						})
					: $_('assistants.form.contextSources.addTool', { default: 'Add Context Source' })}
				onclick={handleAddClick}
			>
				+
			</button>
		{:else if formState === 'edit'}
			<p class="text-xs text-gray-500 italic ml-1">
				{$_('assistants.form.contextSources.editModeNoChanges', {
					default: 'Adding or removing context sources is not available in edit mode'
				})}
			</p>
		{/if}
	</div>
</div>
