<script lang="ts">
	import { Download, Upload, FileText, X } from '@lucide/svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import Button from '$lib/components/ui/button/button.svelte';
	import { toast } from 'svelte-sonner';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { goto } from '$app/navigation';

	let { open = $bindable(false), conversationId = $bindable<string | null>(null) } = $props();

	let isImporting = $state(false);
	let isExporting = $state(false);
	let importFile: File | null = $state(null);
	let fileInput: HTMLInputElement | null = $state(null);

	async function handleExport() {
		if (!conversationId) {
			toast.error('No conversation selected');
			return;
		}

		isExporting = true;
		try {
			const response = await fetch(`/api/chats/${conversationId}/export`);
			if (!response.ok) {
				const error = await response.json().catch(() => ({ error: 'Export failed' }));
				throw new Error(error.error || 'Export failed');
			}

			// Get filename from Content-Disposition header or generate one
			const contentDisposition = response.headers.get('Content-Disposition');
			let filename = `chat_${conversationId}_${Date.now()}.json`;
			if (contentDisposition) {
				const filenameMatch = contentDisposition.match(/filename="(.+)"/);
				if (filenameMatch) {
					filename = filenameMatch[1];
				}
			}

			const blob = await response.blob();
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = filename;
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);

			toast.success(`Chat saved as ${filename}`);
			open = false;
		} catch (error) {
			console.error('Export error:', error);
			toast.error(error instanceof Error ? error.message : 'Failed to export conversation');
		} finally {
			isExporting = false;
		}
	}

	function handleFileSelect(event: Event) {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files.length > 0) {
			importFile = target.files[0];
		}
	}

	async function handleImport() {
		if (!importFile) {
			toast.error('Please select a file to import');
			return;
		}

		isImporting = true;
		try {
			const text = await importFile.text();
			const importData = JSON.parse(text);

			const response = await fetch('/api/chats/import', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(importData)
			});

			if (!response.ok) {
				const error = await response.json();
				throw new Error(error.error || 'Import failed');
			}

			const result = await response.json();
			toast.success(`Imported ${result.importedMessages} messages`);

			// Refresh conversations and navigate to the new one
			await chatStore.loadConversations();
			await goto(`/chat/${result.conversation.id}`);
			open = false;
			importFile = null;
			if (fileInput) fileInput.value = '';
		} catch (error) {
			console.error('Import error:', error);
			toast.error(error instanceof Error ? error.message : 'Failed to import conversation');
		} finally {
			isImporting = false;
		}
	}

	function handleClearFile() {
		importFile = null;
		if (fileInput) fileInput.value = '';
	}
</script>

<Dialog.Content class="max-w-md">
	<Dialog.Header>
		<Dialog.Title class="flex items-center gap-2">
			<FileText class="h-5 w-5" />
			Save / Load Chat
		</Dialog.Title>
		<Dialog.Description>
			Save conversations to JSON files or load previously saved chats.
		</Dialog.Description>
	</Dialog.Header>

	<div class="space-y-4 py-4">
		<!-- Save Chat Section -->
		<div class="space-y-3">
			<h3 class="font-semibold flex items-center gap-2">
				<Download class="h-4 w-4" />
				Save Chat
			</h3>
			<p class="text-sm text-muted-foreground">
				Save the current conversation to a JSON file on your computer.
			</p>
			<Button
				class="w-full"
				onclick={handleExport}
				disabled={!conversationId || isExporting}
				variant="default"
			>
				<Download class="h-4 w-4" />
				{isExporting ? 'Saving...' : 'Save Chat'}
			</Button>
		</div>

		<div class="border-t" />

		<!-- Load Chat Section -->
		<div class="space-y-3">
			<h3 class="font-semibold flex items-center gap-2">
				<Upload class="h-4 w-4" />
				Load Chat
			</h3>
			<p class="text-sm text-muted-foreground">
				Load a previously saved conversation from a JSON file.
			</p>

			<div class="space-y-2">
				<input
					bind:ref={fileInput}
					type="file"
					accept=".json"
					onchange={handleFileSelect}
					class="hidden"
					id="import-file-input"
				/>
				<label for="import-file-input">
					<Button variant="outline" class="w-full" onclick={() => fileInput?.click()}>
						<Upload class="h-4 w-4" />
						Choose File
					</Button>
				</label>

				{#if importFile}
					<div class="flex items-center justify-between rounded-lg border p-3 bg-accent">
						<div class="flex items-center gap-2 flex-1 min-w-0">
							<FileText class="h-4 w-4 flex-shrink-0" />
							<span class="text-sm truncate">{importFile.name}</span>
							<span class="text-xs text-muted-foreground">
								({(importFile.size / 1024).toFixed(1)} KB)
							</span>
						</div>
						<Button variant="ghost" size="sm" onclick={handleClearFile}>
							<X class="h-4 w-4" />
						</Button>
					</div>
				{/if}

				<Button
					class="w-full"
					onclick={handleImport}
					disabled={!importFile || isImporting}
					variant="default"
				>
					{isImporting ? 'Loading...' : 'Load Chat'}
				</Button>
			</div>
		</div>
	</div>

	<Dialog.Footer>
		<Dialog.Close asChild let:builder>
			<Button builders={[builder]} variant="outline">Close</Button>
		</Dialog.Close>
	</Dialog.Footer>
</Dialog.Content>

