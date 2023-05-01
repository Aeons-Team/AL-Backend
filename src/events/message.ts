import { Socket } from 'socket.io'
import { ChatOpenAI } from 'langchain/chat_models/openai'
import { PineconeStore } from 'langchain/vectorstores/pinecone'
import { BaseCallbackHandler } from 'langchain/callbacks'
import { LLMResult } from 'langchain/dist/schema'
import { Conversation, Message } from '../types'

const openAIChat = new ChatOpenAI({
    modelName: 'gpt-3.5-turbo',
    openAIApiKey: process.env.OPEN_AI_API_KEY,
    temperature: 0.1,
    streaming: true
})

const message = async (socket: Socket, convo: Conversation, db: PineconeStore, msg: Message) => {
    if (socket.data.busy) return
    socket.data.busy = true

    const top = await db.similaritySearch(msg.text, 3)
    const topContent = top.map(x => x.pageContent)

    await Promise.all(topContent.map(text => convo.history.addUserMessage(text)))

    const messages = await convo.history.getMessages()
    convo.history.addUserMessage(msg.text)

    const handler = BaseCallbackHandler.fromMethods({
        handleLLMStart(llm: { name: string }, prompts: string[], runId: string) {
            socket.emit('response-start')
        },
    
        handleLLMEnd(output: LLMResult, runId: string) {
            socket.data.busy = false
            socket.emit('response-end')
        },

        handleLLMNewToken(token: string, runId: string) {
            socket.emit('response-text', { text: token })
        },

        handleLLMError(error: any, runId: string) {
            socket.data.busy = false
            socket.emit('response-error', error)
        }
    })

    const reply = await openAIChat.call(messages, undefined, [handler])
    convo.history.addAIChatMessage(reply.text)
}

export default message