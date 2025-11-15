import type { RequestHandler } from './$types';
import { getSQLiteDatabase } from '$lib/database/sqlite';

/**
 * GET /api/chats/[id]/export - Export a conversation to JSON file
 */
export const GET: RequestHandler = async ({ params }) => {
	try {
		const db = getSQLiteDatabase();
		const conversation = db.getConversation(params.id);

		if (!conversation) {
			return new Response(JSON.stringify({ error: 'Conversation not found' }), {
				status: 404,
				headers: { 'Content-Type': 'application/json' }
			});
		}

		const messages = db.getConversationMessages(params.id);

		const exportData = {
			conversation: {
				id: conversation.id,
				name: conversation.name,
				lastModified: conversation.lastModified,
				exportedAt: Date.now()
			},
			messages: messages.map((msg) => ({
				id: msg.id,
				type: msg.type,
				role: msg.role,
				content: msg.content,
				thinking: msg.thinking || '',
				timestamp: msg.timestamp,
				parent: msg.parent || null,
				model: msg.model,
				extra: msg.extra,
				timings: msg.timings
			}))
		};

		const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
		const filename = `chat_${conversation.id}_${timestamp}.json`;

		return new Response(JSON.stringify(exportData, null, 2), {
			headers: {
				'Content-Type': 'application/json',
				'Content-Disposition': `attachment; filename="${filename}"`
			}
		});
	} catch (error) {
		console.error('Error exporting conversation:', error);
		return new Response(JSON.stringify({ error: 'Failed to export conversation' }), {
			status: 500,
			headers: { 'Content-Type': 'application/json' }
		});
	}
};

