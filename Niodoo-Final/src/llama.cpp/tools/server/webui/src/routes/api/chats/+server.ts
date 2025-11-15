import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getSQLiteDatabase } from '$lib/database/sqlite';

/**
 * GET /api/chats - Get all conversations
 * Query params:
 *   - archived: boolean (default: false) - Include archived conversations
 *   - startDate: number (optional) - Filter archived by start date
 *   - endDate: number (optional) - Filter archived by end date
 */
export const GET: RequestHandler = async ({ url }) => {
	try {
		const db = getSQLiteDatabase();
		const includeArchived = url.searchParams.get('archived') === 'true';
		const startDate = url.searchParams.get('startDate')
			? parseInt(url.searchParams.get('startDate')!)
			: undefined;
		const endDate = url.searchParams.get('endDate')
			? parseInt(url.searchParams.get('endDate')!)
			: undefined;

		if (includeArchived && (startDate || endDate)) {
			const archived = db.getArchivedConversations(startDate, endDate);
			return json(archived);
		}

		const conversations = db.getAllConversations(includeArchived);
		return json(conversations);
	} catch (error) {
		console.error('Error fetching conversations:', error);
		return json({ error: 'Failed to fetch conversations' }, { status: 500 });
	}
};

/**
 * POST /api/chats - Create a new conversation
 * Body: { name: string }
 */
export const POST: RequestHandler = async ({ request }) => {
	try {
		const { name } = await request.json();
		if (!name || typeof name !== 'string') {
			return json({ error: 'Name is required' }, { status: 400 });
		}

		const db = getSQLiteDatabase();
		const conversation = db.createConversation(name);
		return json(conversation);
	} catch (error) {
		console.error('Error creating conversation:', error);
		return json({ error: 'Failed to create conversation' }, { status: 500 });
	}
};




