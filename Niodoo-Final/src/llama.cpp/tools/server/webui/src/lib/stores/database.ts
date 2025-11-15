import { filterByLeafNodeId, findDescendantMessages } from '$lib/utils/branching';

/**
 * DatabaseStore - Persistent data layer for conversation and message management
 *
 * This service provides a comprehensive data access layer built on SQLite via API.
 * It handles all persistent storage operations for conversations, messages, and application settings
 * with support for complex conversation branching and message threading.
 *
 * **Architecture & Relationships:**
 * - **DatabaseStore** (this class): Stateless data persistence layer
 *   - Manages SQLite operations through REST API
 *   - Handles conversation and message CRUD operations
 *   - Supports complex branching with parent-child relationships
 *   - Provides transaction safety for multi-table operations
 *
 * - **ChatStore**: Primary consumer for conversation state management
 *   - Uses DatabaseStore for all persistence operations
 *   - Coordinates UI state with database state
 *   - Handles conversation lifecycle and message branching
 *
 * **Key Features:**
 * - **Conversation Management**: Create, read, update, delete conversations
 * - **Message Branching**: Support for tree-like conversation structures
 * - **Transaction Safety**: Atomic operations for data consistency
 * - **Path Resolution**: Navigate conversation branches and find leaf nodes
 * - **Cascading Deletion**: Remove entire conversation branches
 * - **Archive Support**: Archive conversations by date/time
 *
 * **Database Schema:**
 * - `conversations`: Conversation metadata with current node tracking and archive status
 * - `messages`: Individual messages with parent-child relationships
 *
 * **Branching Model:**
 * Messages form a tree structure where each message can have multiple children,
 * enabling conversation branching and alternative response paths. The conversation's
 * `currNode` tracks the currently active branch endpoint.
 */

export class DatabaseStore {
	/**
	 * Adds a new message to the database.
	 *
	 * @param message - Message to add (without id)
	 * @returns The created message
	 */
	static async addMessage(message: Omit<DatabaseMessage, 'id'>): Promise<DatabaseMessage> {
		const response = await fetch(`/api/chats/${message.convId}/messages`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(message)
		});

		if (!response.ok) {
			throw new Error('Failed to add message');
		}

		return await response.json();
	}

	/**
	 * Creates a new conversation.
	 *
	 * @param name - Name of the conversation
	 * @returns The created conversation
	 */
	static async createConversation(name: string): Promise<DatabaseConversation> {
		const response = await fetch('/api/chats', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ name })
		});

		if (!response.ok) {
			throw new Error('Failed to create conversation');
		}

		return await response.json();
	}

	/**
	 * Creates a new message branch by adding a message and updating parent/child relationships.
	 * Also updates the conversation's currNode to point to the new message.
	 *
	 * @param message - Message to add (without id)
	 * @param parentId - Parent message ID to attach to
	 * @returns The created message
	 */
	static async createMessageBranch(
		message: Omit<DatabaseMessage, 'id'>,
		parentId: string | null
	): Promise<DatabaseMessage> {
		const response = await fetch(`/api/chats/${message.convId}/messages`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				...message,
				parentId: parentId
			})
		});

		if (!response.ok) {
			throw new Error('Failed to create message branch');
		}

		return await response.json();
	}

	/**
	 * Creates a root message for a new conversation.
	 * Root messages are not displayed but serve as the tree root for branching.
	 *
	 * @param convId - Conversation ID
	 * @returns The created root message ID
	 */
	static async createRootMessage(convId: string): Promise<string> {
		const response = await fetch(`/api/chats/${convId}/root`, {
			method: 'POST'
		});

		if (!response.ok) {
			throw new Error('Failed to create root message');
		}

		const data = await response.json();
		return data.id;
	}

	/**
	 * Deletes a conversation and all its messages.
	 *
	 * @param id - Conversation ID
	 */
	static async deleteConversation(id: string): Promise<void> {
		const response = await fetch(`/api/chats/${id}`, {
			method: 'DELETE'
		});

		if (!response.ok) {
			throw new Error('Failed to delete conversation');
		}
	}

	/**
	 * Deletes a message and removes it from its parent's children array.
	 *
	 * @param messageId - ID of the message to delete
	 */
	static async deleteMessage(messageId: string): Promise<void> {
		const response = await fetch(`/api/messages/${messageId}`, {
			method: 'DELETE'
		});

		if (!response.ok) {
			throw new Error('Failed to delete message');
		}
	}

	/**
	 * Deletes a message and all its descendant messages (cascading deletion).
	 * This removes the entire branch starting from the specified message.
	 *
	 * @param conversationId - ID of the conversation containing the message
	 * @param messageId - ID of the root message to delete (along with all descendants)
	 * @returns Array of all deleted message IDs
	 */
	static async deleteMessageCascading(
		conversationId: string,
		messageId: string
	): Promise<string[]> {
		const response = await fetch(
			`/api/messages/${messageId}?conversationId=${conversationId}&cascade=true`,
			{
				method: 'DELETE'
			}
		);

		if (!response.ok) {
			throw new Error('Failed to delete message cascading');
		}

		const data = await response.json();
		return data.deletedIds || [];
	}

	/**
	 * Gets all conversations, sorted by last modified time (newest first).
	 *
	 * @param includeArchived - Whether to include archived conversations (default: false)
	 * @returns Array of conversations
	 */
	static async getAllConversations(includeArchived: boolean = false): Promise<DatabaseConversation[]> {
		const url = includeArchived ? '/api/chats?archived=true' : '/api/chats';
		const response = await fetch(url);

		if (!response.ok) {
			throw new Error('Failed to fetch conversations');
		}

		return await response.json();
	}

	/**
	 * Gets a conversation by ID.
	 *
	 * @param id - Conversation ID
	 * @returns The conversation if found, otherwise undefined
	 */
	static async getConversation(id: string): Promise<DatabaseConversation | undefined> {
		const response = await fetch(`/api/chats/${id}`);

		if (response.status === 404) {
			return undefined;
		}

		if (!response.ok) {
			throw new Error('Failed to fetch conversation');
		}

		return await response.json();
	}

	/**
	 * Gets all leaf nodes (messages with no children) in a conversation.
	 * Useful for finding all possible conversation endpoints.
	 *
	 * @param convId - Conversation ID
	 * @returns Array of leaf node message IDs
	 */
	static async getConversationLeafNodes(convId: string): Promise<string[]> {
		const allMessages = await this.getConversationMessages(convId);
		return allMessages.filter((msg) => msg.children.length === 0).map((msg) => msg.id);
	}

	/**
	 * Gets all messages in a conversation, sorted by timestamp (oldest first).
	 *
	 * @param convId - Conversation ID
	 * @returns Array of messages in the conversation
	 */
	static async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		const response = await fetch(`/api/chats/${convId}/messages`);

		if (!response.ok) {
			throw new Error('Failed to fetch messages');
		}

		return await response.json();
	}

	/**
	 * Gets the conversation path from root to the current leaf node.
	 * Uses the conversation's currNode to determine the active branch.
	 *
	 * @param convId - Conversation ID
	 * @returns Array of messages in the current conversation path
	 */
	static async getConversationPath(convId: string): Promise<DatabaseMessage[]> {
		const conversation = await this.getConversation(convId);

		if (!conversation) {
			return [];
		}

		const allMessages = await this.getConversationMessages(convId);

		if (allMessages.length === 0) {
			return [];
		}

		// If no currNode is set, use the latest message as leaf
		const leafNodeId =
			conversation.currNode ||
			allMessages.reduce((latest, msg) => (msg.timestamp > latest.timestamp ? msg : latest)).id;

		return filterByLeafNodeId(allMessages, leafNodeId, false) as DatabaseMessage[];
	}

	/**
	 * Updates a conversation.
	 *
	 * @param id - Conversation ID
	 * @param updates - Partial updates to apply
	 * @returns Promise that resolves when the conversation is updated
	 */
	static async updateConversation(
		id: string,
		updates: Partial<Omit<DatabaseConversation, 'id'>>
	): Promise<void> {
		const response = await fetch(`/api/chats/${id}`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(updates)
		});

		if (!response.ok) {
			throw new Error('Failed to update conversation');
		}
	}

	/**
	 * Updates the conversation's current node (active branch).
	 * This determines which conversation path is currently being viewed.
	 *
	 * @param convId - Conversation ID
	 * @param nodeId - Message ID to set as current node
	 */
	static async updateCurrentNode(convId: string, nodeId: string): Promise<void> {
		await this.updateConversation(convId, {
			currNode: nodeId
		});
	}

	/**
	 * Updates a message.
	 *
	 * @param id - Message ID
	 * @param updates - Partial updates to apply
	 * @returns Promise that resolves when the message is updated
	 */
	static async updateMessage(
		id: string,
		updates: Partial<Omit<DatabaseMessage, 'id'>>
	): Promise<void> {
		const response = await fetch(`/api/messages/${id}`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(updates)
		});

		if (!response.ok) {
			throw new Error('Failed to update message');
		}
	}

	/**
	 * Archives a conversation.
	 *
	 * @param id - Conversation ID
	 */
	static async archiveConversation(id: string): Promise<void> {
		const response = await fetch(`/api/chats/${id}/archive`, {
			method: 'POST'
		});

		if (!response.ok) {
			throw new Error('Failed to archive conversation');
		}
	}

	/**
	 * Unarchives a conversation.
	 *
	 * @param id - Conversation ID
	 */
	static async unarchiveConversation(id: string): Promise<void> {
		const response = await fetch(`/api/chats/${id}/archive`, {
			method: 'DELETE'
		});

		if (!response.ok) {
			throw new Error('Failed to unarchive conversation');
		}
	}

	/**
	 * Gets archived conversations with optional date filtering.
	 *
	 * @param startDate - Optional start date timestamp
	 * @param endDate - Optional end date timestamp
	 * @returns Array of archived conversations
	 */
	static async getArchivedConversations(
		startDate?: number,
		endDate?: number
	): Promise<Array<DatabaseConversation & { archivedAt: number }>> {
		const params = new URLSearchParams();
		if (startDate) params.set('startDate', startDate.toString());
		if (endDate) params.set('endDate', endDate.toString());

		const url = `/api/chats?archived=true${params.toString() ? `&${params.toString()}` : ''}`;
		const response = await fetch(url);

		if (!response.ok) {
			throw new Error('Failed to fetch archived conversations');
		}

		return await response.json();
	}
}
