import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getSQLiteDatabase } from '$lib/database/sqlite';

/**
 * POST /api/chats/import - Import a conversation from JSON
 * Body: { conversation: {...}, messages: [...] }
 */
export const POST: RequestHandler = async ({ request }) => {
	try {
		const importData = await request.json();

		if (!importData.conversation || !importData.messages) {
			return json({ error: 'Invalid import format. Expected {conversation, messages}' }, { status: 400 });
		}

		const db = getSQLiteDatabase();

		// Create new conversation (use imported name or generate one)
		const conversationName = importData.conversation.name || `Imported Chat ${new Date().toLocaleString()}`;
		const newConversation = db.createConversation(conversationName);

		// Import messages in order
		const messageMap = new Map<string, string>(); // old_id -> new_id
		const rootMessageId = db.createRootMessage(newConversation.id);
		messageMap.set('root', rootMessageId);

		// Build a map of old message IDs to their data for proper parent resolution
		const messageDataMap = new Map<string, typeof importData.messages[0]>();
		for (const msg of importData.messages) {
			messageDataMap.set(msg.id, msg);
		}

		// Sort messages by timestamp to maintain order
		const sortedMessages = [...importData.messages]
			.filter((msg) => !(msg.role === 'system' && msg.type === 'root'))
			.sort((a, b) => a.timestamp - b.timestamp);

		// Import messages, maintaining parent relationships
		for (const msg of sortedMessages) {
			// Determine parent - if original parent exists in map, use it, otherwise use root
			let parentId: string | null = rootMessageId;
			if (msg.parent) {
				// Check if parent was already imported
				if (messageMap.has(msg.parent)) {
					parentId = messageMap.get(msg.parent)!;
				} else {
					// Parent not yet imported, use root for now (will be fixed in second pass if needed)
					parentId = rootMessageId;
				}
			}

			const newMessage = db.createMessageBranch(
				{
					convId: newConversation.id,
					type: msg.type || 'text',
					role: msg.role,
					content: msg.content,
					thinking: msg.thinking || '',
					timestamp: msg.timestamp || Date.now(),
					parent: parentId || '',
					children: [],
					model: msg.model,
					extra: msg.extra,
					timings: msg.timings
				},
				parentId
			);

			messageMap.set(msg.id, newMessage.id);
		}

		return json({
			success: true,
			conversation: newConversation,
			importedMessages: messageMap.size - 1 // Exclude root
		});
	} catch (error) {
		console.error('Error importing conversation:', error);
		return json({ error: 'Failed to import conversation' }, { status: 500 });
	}
};

