import Database from 'better-sqlite3';
import { join } from 'path';
import { existsSync, mkdirSync } from 'fs';
import type {
	DatabaseConversation,
	DatabaseMessage,
	DatabaseMessageExtra
} from '$lib/types/database';
import type { ChatMessageType, ChatRole } from '$lib/types/chat';
import { v4 as uuid } from 'uuid';

// Use .svelte-kit directory for database storage in development
// In production, this should be configured via environment variable
const DB_DIR = process.env.CHAT_DB_DIR || join(process.cwd(), 'data', 'chats');
const DB_PATH = join(DB_DIR, 'chats.db');

// Ensure data directory exists
if (!existsSync(DB_DIR)) {
	mkdirSync(DB_DIR, { recursive: true });
}

let dbInstance: Database.Database | null = null;

/**
 * Get or create SQLite database instance
 * Singleton pattern to ensure single connection
 * Only works in server context (not in browser)
 * Note: better-sqlite3 is a native module that only works in Node.js
 */
function getDatabase(): Database.Database {
	if (!dbInstance) {
		dbInstance = new Database(DB_PATH);
		dbInstance.pragma('journal_mode = WAL'); // Write-Ahead Logging for better concurrency
		dbInstance.pragma('foreign_keys = ON'); // Enable foreign key constraints
		initializeSchema(dbInstance);
	}
	return dbInstance;
}

/**
 * Initialize database schema
 * Creates tables for conversations, messages, and archives
 */
function initializeSchema(db: Database.Database): void {
	// Conversations table
	db.exec(`
		CREATE TABLE IF NOT EXISTS conversations (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL,
			lastModified INTEGER NOT NULL,
			currNode TEXT,
			archived INTEGER DEFAULT 0,
			archivedAt INTEGER,
			createdAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000)
		)
	`);

	// Messages table
	db.exec(`
		CREATE TABLE IF NOT EXISTS messages (
			id TEXT PRIMARY KEY,
			convId TEXT NOT NULL,
			type TEXT NOT NULL,
			role TEXT NOT NULL,
			content TEXT NOT NULL,
			thinking TEXT DEFAULT '',
			timestamp INTEGER NOT NULL,
			parent TEXT,
			model TEXT,
			extra TEXT,
			timings TEXT,
			FOREIGN KEY (convId) REFERENCES conversations(id) ON DELETE CASCADE,
			FOREIGN KEY (parent) REFERENCES messages(id) ON DELETE SET NULL
		)
	`);

	// Message children relationship table (for branching)
	db.exec(`
		CREATE TABLE IF NOT EXISTS message_children (
			parentId TEXT NOT NULL,
			childId TEXT NOT NULL,
			PRIMARY KEY (parentId, childId),
			FOREIGN KEY (parentId) REFERENCES messages(id) ON DELETE CASCADE,
			FOREIGN KEY (childId) REFERENCES messages(id) ON DELETE CASCADE
		)
	`);

	// Indexes for performance
	db.exec(`
		CREATE INDEX IF NOT EXISTS idx_messages_convId ON messages(convId);
		CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent);
		CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
		CREATE INDEX IF NOT EXISTS idx_conversations_lastModified ON conversations(lastModified);
		CREATE INDEX IF NOT EXISTS idx_conversations_archived ON conversations(archived);
	`);
}

/**
 * SQLite Database Service
 * Provides persistent storage using SQLite instead of IndexedDB
 * This allows for better server-side management and avoids browser storage limits
 */
export class SQLiteDatabase {
	private db: Database.Database;

	constructor() {
		this.db = getDatabase();
	}

	/**
	 * Create a new conversation
	 */
	createConversation(name: string): DatabaseConversation {
		const conversation: DatabaseConversation = {
			id: uuid(),
			name,
			lastModified: Date.now(),
			currNode: null
		};

		const stmt = this.db.prepare(`
			INSERT INTO conversations (id, name, lastModified, currNode, archived, createdAt)
			VALUES (?, ?, ?, ?, 0, ?)
		`);

		stmt.run(
			conversation.id,
			conversation.name,
			conversation.lastModified,
			conversation.currNode || '',
			Date.now()
		);

		return conversation;
	}

	/**
	 * Get all conversations, optionally filtered by archived status
	 */
	getAllConversations(includeArchived: boolean = false): DatabaseConversation[] {
		const query = includeArchived
			? `SELECT id, name, lastModified, currNode FROM conversations ORDER BY lastModified DESC`
			: `SELECT id, name, lastModified, currNode FROM conversations WHERE archived = 0 ORDER BY lastModified DESC`;

		const rows = this.db.prepare(query).all() as Array<{
			id: string;
			name: string;
			lastModified: number;
			currNode: string | null;
		}>;

		return rows.map((row) => ({
			id: row.id,
			name: row.name,
			lastModified: row.lastModified,
			currNode: row.currNode || null
		}));
	}

	/**
	 * Get archived conversations with date filtering
	 */
	getArchivedConversations(
		startDate?: number,
		endDate?: number
	): Array<DatabaseConversation & { archivedAt: number }> {
		let query = `
			SELECT id, name, lastModified, currNode, archivedAt
			FROM conversations
			WHERE archived = 1
		`;

		const params: number[] = [];

		if (startDate) {
			query += ` AND archivedAt >= ?`;
			params.push(startDate);
		}

		if (endDate) {
			query += ` AND archivedAt <= ?`;
			params.push(endDate);
		}

		query += ` ORDER BY archivedAt DESC`;

		const rows = this.db.prepare(query).all(...params) as Array<{
			id: string;
			name: string;
			lastModified: number;
			currNode: string | null;
			archivedAt: number;
		}>;

		return rows.map((row) => ({
			id: row.id,
			name: row.name,
			lastModified: row.lastModified,
			currNode: row.currNode || null,
			archivedAt: row.archivedAt
		}));
	}

	/**
	 * Get a conversation by ID
	 */
	getConversation(id: string): DatabaseConversation | undefined {
		const row = this.db
			.prepare(`SELECT id, name, lastModified, currNode FROM conversations WHERE id = ?`)
			.get(id) as
			| {
					id: string;
					name: string;
					lastModified: number;
					currNode: string | null;
			  }
			| undefined;

		if (!row) return undefined;

		return {
			id: row.id,
			name: row.name,
			lastModified: row.lastModified,
			currNode: row.currNode || null
		};
	}

	/**
	 * Update a conversation
	 */
	updateConversation(
		id: string,
		updates: Partial<Omit<DatabaseConversation, 'id'>>
	): void {
		const fields: string[] = [];
		const values: unknown[] = [];

		if (updates.name !== undefined) {
			fields.push('name = ?');
			values.push(updates.name);
		}

		if (updates.currNode !== undefined) {
			fields.push('currNode = ?');
			values.push(updates.currNode || '');
		}

		fields.push('lastModified = ?');
		values.push(Date.now());

		if (fields.length === 0) return;

		values.push(id);

		const stmt = this.db.prepare(
			`UPDATE conversations SET ${fields.join(', ')} WHERE id = ?`
		);
		stmt.run(...values);
	}

	/**
	 * Archive a conversation
	 */
	archiveConversation(id: string): void {
		const stmt = this.db.prepare(
			`UPDATE conversations SET archived = 1, archivedAt = ? WHERE id = ?`
		);
		stmt.run(Date.now(), id);
	}

	/**
	 * Unarchive a conversation
	 */
	unarchiveConversation(id: string): void {
		const stmt = this.db.prepare(
			`UPDATE conversations SET archived = 0, archivedAt = NULL WHERE id = ?`
		);
		stmt.run(id);
	}

	/**
	 * Delete a conversation and all its messages (cascading)
	 */
	deleteConversation(id: string): void {
		const stmt = this.db.prepare(`DELETE FROM conversations WHERE id = ?`);
		stmt.run(id);
	}

	/**
	 * Create a message
	 */
	createMessage(message: Omit<DatabaseMessage, 'id'>): DatabaseMessage {
		const newMessage: DatabaseMessage = {
			...message,
			id: uuid()
		};

		const stmt = this.db.prepare(`
			INSERT INTO messages (id, convId, type, role, content, thinking, timestamp, parent, model, extra, timings)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`);

		stmt.run(
			newMessage.id,
			newMessage.convId,
			newMessage.type,
			newMessage.role,
			newMessage.content,
			newMessage.thinking || '',
			newMessage.timestamp,
			newMessage.parent || null,
			newMessage.model || null,
			newMessage.extra ? JSON.stringify(newMessage.extra) : null,
			newMessage.timings ? JSON.stringify(newMessage.timings) : null
		);

		// Update parent's children if parent exists
		if (newMessage.parent) {
			const childStmt = this.db.prepare(
				`INSERT OR IGNORE INTO message_children (parentId, childId) VALUES (?, ?)`
			);
			childStmt.run(newMessage.parent, newMessage.id);
		}

		return newMessage;
	}

	/**
	 * Create a message branch (with parent relationship)
	 */
	createMessageBranch(
		message: Omit<DatabaseMessage, 'id'>,
		parentId: string | null
	): DatabaseMessage {
		const newMessage = this.createMessage({
			...message,
			parent: parentId || null
		});

		// Update conversation's current node
		if (message.convId) {
			this.updateConversation(message.convId, { currNode: newMessage.id });
		}

		return newMessage;
	}

	/**
	 * Get all messages for a conversation
	 */
	getConversationMessages(convId: string): DatabaseMessage[] {
		const rows = this.db
			.prepare(`SELECT * FROM messages WHERE convId = ? ORDER BY timestamp ASC`)
			.all(convId) as Array<{
			id: string;
			convId: string;
			type: string;
			role: string;
			content: string;
			thinking: string;
			timestamp: number;
			parent: string | null;
			model: string | null;
			extra: string | null;
			timings: string | null;
		}>;

		// Get children for each message
		const childrenMap = new Map<string, string[]>();
		const childRows = this.db
			.prepare(`SELECT parentId, childId FROM message_children WHERE parentId IN (SELECT id FROM messages WHERE convId = ?)`)
			.all(convId) as Array<{ parentId: string; childId: string }>;

		for (const row of childRows) {
			if (!childrenMap.has(row.parentId)) {
				childrenMap.set(row.parentId, []);
			}
			childrenMap.get(row.parentId)!.push(row.childId);
		}

		return rows.map((row) => ({
			id: row.id,
			convId: row.convId,
			type: row.type as ChatMessageType,
			role: row.role as ChatRole,
			content: row.content,
			thinking: row.thinking || '',
			timestamp: row.timestamp,
			parent: row.parent || '',
			children: childrenMap.get(row.id) || [],
			model: row.model || undefined,
			extra: row.extra ? (JSON.parse(row.extra) as DatabaseMessageExtra[]) : undefined,
			timings: row.timings ? JSON.parse(row.timings) : undefined
		}));
	}

	/**
	 * Update a message
	 */
	updateMessage(id: string, updates: Partial<Omit<DatabaseMessage, 'id'>>): void {
		const fields: string[] = [];
		const values: unknown[] = [];

		if (updates.content !== undefined) {
			fields.push('content = ?');
			values.push(updates.content);
		}

		if (updates.thinking !== undefined) {
			fields.push('thinking = ?');
			values.push(updates.thinking);
		}

		if (updates.timings !== undefined) {
			fields.push('timings = ?');
			values.push(JSON.stringify(updates.timings));
		}

		if (fields.length === 0) return;

		values.push(id);

		const stmt = this.db.prepare(`UPDATE messages SET ${fields.join(', ')} WHERE id = ?`);
		stmt.run(...values);
	}

	/**
	 * Delete a message and all its descendants (cascading)
	 */
	deleteMessageCascading(conversationId: string, messageId: string): string[] {
		const deletedIds: string[] = [];

		// Recursive CTE to find all descendants
		const descendants = this.db
			.prepare(`
			WITH RECURSIVE descendants AS (
				SELECT id FROM messages WHERE id = ?
				UNION ALL
				SELECT m.id FROM messages m
				INNER JOIN message_children mc ON m.id = mc.childId
				INNER JOIN descendants d ON mc.parentId = d.id
			)
			SELECT id FROM descendants
		`)
			.all(messageId) as Array<{ id: string }>;

		const idsToDelete = descendants.map((d) => d.id);
		deletedIds.push(...idsToDelete);

		// Delete from message_children first
		if (idsToDelete.length > 0) {
			const placeholders = idsToDelete.map(() => '?').join(',');
			this.db.prepare(`DELETE FROM message_children WHERE parentId IN (${placeholders}) OR childId IN (${placeholders})`).run(...idsToDelete, ...idsToDelete);
		}

		// Delete messages
		if (idsToDelete.length > 0) {
			const placeholders = idsToDelete.map(() => '?').join(',');
			this.db.prepare(`DELETE FROM messages WHERE id IN (${placeholders})`).run(...idsToDelete);
		}

		return deletedIds;
	}

	/**
	 * Delete a single message
	 */
	deleteMessage(messageId: string): void {
		// Remove from message_children
		this.db.prepare(`DELETE FROM message_children WHERE parentId = ? OR childId = ?`).run(messageId, messageId);

		// Delete the message
		this.db.prepare(`DELETE FROM messages WHERE id = ?`).run(messageId);
	}

	/**
	 * Create root message for a conversation
	 */
	createRootMessage(convId: string): string {
		const rootMessage: DatabaseMessage = {
			id: uuid(),
			convId,
			type: 'root',
			timestamp: Date.now(),
			role: 'system',
			content: '',
			parent: '',
			thinking: '',
			children: []
		};

		this.createMessage(rootMessage);
		return rootMessage.id;
	}

	/**
	 * Close database connection
	 */
	close(): void {
		if (dbInstance) {
			dbInstance.close();
			dbInstance = null;
		}
	}
}

// Singleton instance
let sqliteInstance: SQLiteDatabase | null = null;

export function getSQLiteDatabase(): SQLiteDatabase {
	if (!sqliteInstance) {
		sqliteInstance = new SQLiteDatabase();
	}
	return sqliteInstance;
}

