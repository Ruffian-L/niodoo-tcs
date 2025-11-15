<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { Plus, Archive, Calendar, Download, Upload, Trash2 } from '@lucide/svelte';
	import { createConversation, activeConversation, chatStore } from '$lib/stores/chat.svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import Button from '$lib/components/ui/button/button.svelte';
	import ArchiveDialog from './ArchiveDialog.svelte';
	import ExportImportDialog from './ExportImportDialog.svelte';
	import { toast } from 'svelte-sonner';

	let showArchiveDialog = $state(false);
	let showExportImportDialog = $state(false);
	let showClearDialog = $state(false);
	let currentConversationId = $derived(activeConversation()?.id || page.params.id || null);

	async function handleNewChat() {
		const id = await createConversation();
		await goto(`/chat/${id}`);
	}

	function handleOpenArchive() {
		showArchiveDialog = true;
	}

	function handleOpenExportImport() {
		showExportImportDialog = true;
	}

	async function handleClearAll() {
		try {
			const response = await fetch('/api/chats/clear', {
				method: 'POST'
			});

			if (!response.ok) {
				throw new Error('Failed to clear conversations');
			}

			const result = await response.json();
			toast.success(result.message || 'All conversations cleared');
			await chatStore.loadConversations();
			await goto('/');
			showClearDialog = false;
		} catch (error) {
			console.error('Clear error:', error);
			toast.error('Failed to clear conversations');
		}
	}
</script>

<div class="flex items-center gap-2 border-b border-sidebar-border bg-sidebar px-4 py-2">
	<Button variant="ghost" size="sm" onclick={handleNewChat} class="gap-2">
		<Plus class="h-4 w-4" />
		New Chat
	</Button>

	<Button variant="ghost" size="sm" onclick={handleOpenArchive} class="gap-2">
		<Archive class="h-4 w-4" />
		Archive
	</Button>

	<Button variant="ghost" size="sm" onclick={handleOpenExportImport} class="gap-2">
		<Download class="h-4 w-4" />
		Save/Load
	</Button>

	<Button variant="ghost" size="sm" onclick={() => showClearDialog = true} class="gap-2 text-destructive">
		<Trash2 class="h-4 w-4" />
		Clear All
	</Button>
</div>

<Dialog.Root bind:open={showArchiveDialog}>
	<ArchiveDialog bind:open={showArchiveDialog} />
</Dialog.Root>

<Dialog.Root bind:open={showExportImportDialog}>
	<ExportImportDialog bind:open={showExportImportDialog} bind:conversationId={currentConversationId} />
</Dialog.Root>

<Dialog.Root bind:open={showClearDialog}>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Clear All Conversations</Dialog.Title>
			<Dialog.Description>
				This will permanently delete all non-archived conversations. This action cannot be undone.
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer class="gap-2">
			<Dialog.Close asChild let:builder>
				<Button builders={[builder]} variant="outline">Cancel</Button>
			</Dialog.Close>
			<Button onclick={handleClearAll} variant="destructive">
				<Trash2 class="h-4 w-4" />
				Clear All
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

