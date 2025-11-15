import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getSQLiteDatabase } from '$lib/database/sqlite';

/**
 * GET /api/chats/[id] - Get a conversation by ID
 */
export const GET: RequestHandler = async ({ params }) => {
	try {
		const db = getSQLiteDatabase();
		const conversation = db.getConversation(params.id);

		if (!conversation) {
			return json({ error: 'Conversation not found' }, { status: 404 });
		}

		return json(conversation);
	} catch (error) {
		console.error('Error fetching conversation:', error);
		return json({ error: 'Failed to fetch conversation' }, { status: 500 });
	}
};

/**
 * PATCH /api/chats/[id] - Update a conversation
 * Body: { name?: string, currNode?: string }
 */
export const PATCH: RequestHandler = async ({ params, request }) => {
	try {
		const updates = await request.json();
		const db = getSQLiteDatabase();
		db.updateConversation(params.id, updates);
		const conversation = db.getConversation(params.id);
		return json(conversation);
	} catch (error) {
		console.error('Error updating conversation:', error);
		return json({ error: 'Failed to update conversation' }, { status: 500 });
	}
};

/**
 * DELETE /api/chats/[id] - Delete a conversation
 */
export const DELETE: RequestHandler = async ({ params }) => {
	try {
		const db = getSQLiteDatabase();
		db.deleteConversation(params.id);
		return json({ success: true });
	} catch (error) {
		console.error('Error deleting conversation:', error);
		return json({ error: 'Failed to delete conversation' }, { status: 500 });
	}
};




