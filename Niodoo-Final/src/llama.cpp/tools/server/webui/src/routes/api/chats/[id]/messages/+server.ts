import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getSQLiteDatabase } from '$lib/database/sqlite';

/**
 * GET /api/chats/[id]/messages - Get all messages for a conversation
 */
export const GET: RequestHandler = async ({ params }) => {
	try {
		const db = getSQLiteDatabase();
		const messages = db.getConversationMessages(params.id);
		return json(messages);
	} catch (error) {
		console.error('Error fetching messages:', error);
		return json({ error: 'Failed to fetch messages' }, { status: 500 });
	}
};

/**
 * POST /api/chats/[id]/messages - Create a new message
 * Body: { type, role, content, thinking, timestamp, parent, model, extra, timings }
 */
export const POST: RequestHandler = async ({ params, request }) => {
	try {
		const messageData = await request.json();
		const db = getSQLiteDatabase();

		// If parentId is provided, create a branch
		if (messageData.parentId !== undefined && messageData.parentId !== null) {
			const message = db.createMessageBranch(
				{
					convId: params.id,
					type: messageData.type,
					role: messageData.role,
					content: messageData.content,
					thinking: messageData.thinking || '',
					timestamp: messageData.timestamp || Date.now(),
					parent: messageData.parent || '',
					children: [],
					model: messageData.model,
					extra: messageData.extra,
					timings: messageData.timings
				},
				messageData.parentId
			);
			return json(message);
		} else {
			const message = db.createMessage({
				convId: params.id,
				type: messageData.type,
				role: messageData.role,
				content: messageData.content,
				thinking: messageData.thinking || '',
				timestamp: messageData.timestamp || Date.now(),
				parent: messageData.parent || '',
				children: [],
				model: messageData.model,
				extra: messageData.extra,
				timings: messageData.timings
			});
			return json(message);
		}
	} catch (error) {
		console.error('Error creating message:', error);
		return json({ error: 'Failed to create message' }, { status: 500 });
	}
};




