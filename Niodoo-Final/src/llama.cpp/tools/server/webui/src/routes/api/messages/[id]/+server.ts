import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getSQLiteDatabase } from '$lib/database/sqlite';

/**
 * PATCH /api/messages/[id] - Update a message
 * Body: { content?, thinking?, timings? }
 */
export const PATCH: RequestHandler = async ({ params, request }) => {
	try {
		const updates = await request.json();
		const db = getSQLiteDatabase();
		db.updateMessage(params.id, updates);
		return json({ success: true });
	} catch (error) {
		console.error('Error updating message:', error);
		return json({ error: 'Failed to update message' }, { status: 500 });
	}
};

/**
 * DELETE /api/messages/[id] - Delete a message
 * Query params:
 *   - conversationId: string (required for cascading delete)
 *   - cascade: boolean (default: false) - Delete all descendants
 */
export const DELETE: RequestHandler = async ({ params, url }) => {
	try {
		const db = getSQLiteDatabase();
		const cascade = url.searchParams.get('cascade') === 'true';
		const conversationId = url.searchParams.get('conversationId');

		if (cascade && conversationId) {
			const deletedIds = db.deleteMessageCascading(conversationId, params.id);
			return json({ success: true, deletedIds });
		} else {
			db.deleteMessage(params.id);
			return json({ success: true });
		}
	} catch (error) {
		console.error('Error deleting message:', error);
		return json({ error: 'Failed to delete message' }, { status: 500 });
	}
};




