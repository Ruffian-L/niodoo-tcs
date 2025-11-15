<script lang="ts">
	import { DatabaseStore } from '$lib/stores/database';
	import { goto } from '$app/navigation';
	import { Archive, Calendar, X } from '@lucide/svelte';
	import * as Dialog from '$lib/components/ui/dialog';
	import Button from '$lib/components/ui/button/button.svelte';
	import Input from '$lib/components/ui/input/input.svelte';
	import Label from '$lib/components/ui/label/label.svelte';
	import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
	import { toast } from 'svelte-sonner';
	let { open = $bindable(false) } = $props();

	let archivedConversations = $state<Array<DatabaseConversation & { archivedAt: number }>>([]);
	let isLoading = $state(false);
	let startDate = $state('');
	let endDate = $state('');

	async function loadArchived() {
		isLoading = true;
		try {
			const start = startDate ? new Date(startDate).getTime() : undefined;
			const end = endDate ? new Date(endDate).getTime() : undefined;
			archivedConversations = await DatabaseStore.getArchivedConversations(start, end);
		} catch (error) {
			console.error('Failed to load archived conversations:', error);
			toast.error('Failed to load archived conversations');
		} finally {
			isLoading = false;
		}
	}

	async function handleUnarchive(id: string) {
		try {
			await DatabaseStore.unarchiveConversation(id);
			toast.success('Conversation unarchived');
			await loadArchived();
		} catch (error) {
			console.error('Failed to unarchive conversation:', error);
			toast.error('Failed to unarchive conversation');
		}
	}

	async function handleDelete(id: string) {
		try {
			await DatabaseStore.deleteConversation(id);
			toast.success('Conversation deleted');
			await loadArchived();
		} catch (error) {
			console.error('Failed to delete conversation:', error);
			toast.error('Failed to delete conversation');
		}
	}

	async function handleOpen(id: string) {
		await DatabaseStore.unarchiveConversation(id);
		await goto(`/chat/${id}`);
	}

	function formatDate(timestamp: number): string {
		return new Date(timestamp).toLocaleString();
	}

	function clearFilters() {
		startDate = '';
		endDate = '';
		loadArchived();
	}

	$effect(() => {
		if (open) {
			loadArchived();
		}
	});
</script>

<Dialog.Content class="max-w-2xl">
	<Dialog.Header>
		<Dialog.Title class="flex items-center gap-2">
			<Archive class="h-5 w-5" />
			Archived Conversations
		</Dialog.Title>
		<Dialog.Description>
			View and manage archived conversations. Filter by date range to find specific conversations.
		</Dialog.Description>
	</Dialog.Header>

	<div class="space-y-4 py-4">
		<!-- Date Filters -->
		<div class="grid grid-cols-2 gap-4">
			<div class="space-y-2">
				<Label for="startDate">Start Date</Label>
				<Input
					id="startDate"
					type="date"
					bind:value={startDate}
					onchange={loadArchived}
					placeholder="Start date"
				/>
			</div>
			<div class="space-y-2">
				<Label for="endDate">End Date</Label>
				<Input
					id="endDate"
					type="date"
					bind:value={endDate}
					onchange={loadArchived}
					placeholder="End date"
				/>
			</div>
		</div>

		<div class="flex gap-2">
			<Button variant="outline" size="sm" onclick={clearFilters}>
				<X class="h-4 w-4" />
				Clear Filters
			</Button>
		</div>

		<!-- Archived Conversations List -->
		<ScrollArea class="h-[400px]">
			{@if isLoading}
				<div class="flex items-center justify-center py-8 text-muted-foreground">
					Loading archived conversations...
				</div>
			{:else if archivedConversations.length === 0}
				<div class="flex flex-col items-center justify-center py-8 text-center text-muted-foreground">
					<Archive class="h-12 w-12 mb-4 opacity-50" />
					<p>No archived conversations found</p>
					{#if startDate || endDate}
						<p class="text-sm mt-2">Try adjusting your date filters</p>
					{/if}
				</div>
			{:else}
				<div class="space-y-2">
					{#each archivedConversations as conversation}
						<div
							class="flex items-center justify-between rounded-lg border p-3 hover:bg-accent"
						>
							<div class="flex-1 min-w-0">
								<h4 class="font-medium truncate">{conversation.name}</h4>
								<p class="text-sm text-muted-foreground flex items-center gap-1">
									<Calendar class="h-3 w-3" />
									Archived: {formatDate(conversation.archivedAt)}
								</p>
							</div>
							<div class="flex gap-2 ml-4">
								<Button variant="ghost" size="sm" onclick={() => handleOpen(conversation.id)}>
									Open
								</Button>
								<Button
									variant="ghost"
									size="sm"
									onclick={() => handleUnarchive(conversation.id)}
								>
									Unarchive
								</Button>
								<Button
									variant="ghost"
									size="sm"
									onclick={() => handleDelete(conversation.id)}
									class="text-destructive"
								>
									Delete
								</Button>
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</ScrollArea>
	</div>

	<Dialog.Footer>
		<Dialog.Close asChild let:builder>
			<Button builders={[builder]} variant="outline">Close</Button>
		</Dialog.Close>
	</Dialog.Footer>
</Dialog.Content>

