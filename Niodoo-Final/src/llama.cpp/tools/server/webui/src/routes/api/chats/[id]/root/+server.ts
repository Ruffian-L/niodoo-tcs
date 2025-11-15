import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { getSQLiteDatabase } from '$lib/database/sqlite';

/**
 * POST /api/chats/[id]/root - Create a root message for a conversation
 */
export const POST: RequestHandler = async ({ params }) => {
	try {
		const db = getSQLiteDatabase();
		const rootId = db.createRootMessage(params.id);
		return json({ id: rootId });
	} catch (error) {
		console.error('Error creating root message:', error);
		return json({ error: 'Failed to create root message' }, { status: 500 });
	}
};




