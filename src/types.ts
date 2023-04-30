import { ChatMessageHistory } from 'langchain/memory'

export interface Conversation {
    history: ChatMessageHistory
}

export interface Message {
    text: string
}