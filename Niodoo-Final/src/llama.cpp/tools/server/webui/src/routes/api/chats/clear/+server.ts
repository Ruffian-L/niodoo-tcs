import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getSQLiteDatabase } from '$lib/database/sqlite';

/**
 * POST /api/chats/clear - Clear all non-archived conversations
 * Query params:
 *   - archive: boolean (default: false) - If true, archive instead of delete
 */
export const POST: RequestHandler = async ({ url }) => {
	try {
		const db = getSQLiteDatabase();
		const archive = url.searchParams.get('archive') === 'true';

		const conversations = db.getAllConversations(false); // Get non-archived only

		if (archive) {
			// Archive all conversations
			for (const conv of conversations) {
				db.archiveConversation(conv.id);
			}
			return json({
				success: true,
				archived: conversations.length,
				message: `Archived ${conversations.length} conversation(s)`
			});
		} else {
			// Delete all conversations
			for (const conv of conversations) {
				db.deleteConversation(conv.id);
			}
			return json({
				success: true,
				deleted: conversations.length,
				message: `Deleted ${conversations.length} conversation(s)`
			});
		}
	} catch (error) {
		console.error('Error clearing conversations:', error);
		return json({ error: 'Failed to clear conversations' }, { status: 500 });
	}
};




