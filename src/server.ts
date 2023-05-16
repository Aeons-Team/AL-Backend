import * as dotenv from 'dotenv'
dotenv.config()

import express from 'express'
import http from 'http'
import cors from 'cors'
import path from 'path'
import { Server } from 'socket.io'
import { PineconeStore } from 'langchain/vectorstores/pinecone'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { TextLoader } from 'langchain/document_loaders/fs/text'
import { CharacterTextSplitter } from 'langchain/text_splitter'
import { ChatMessageHistory } from 'langchain/memory'
import { PineconeClient } from '@pinecone-database/pinecone'
import { message } from './events'
import { Conversation, Message } from './types'

const openAIEmbeddings = new OpenAIEmbeddings({ 
    modelName: 'text-embedding-ada-002',
    openAIApiKey: process.env.OPEN_AI_API_KEY
})

const initialMessage = 'Hey there, I am Ava, Aeons GPT powered chat assistant, How may I help you?'

async function prepare(): Promise<PineconeStore> {
    const { PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME } = process.env
    
    if (!PINECONE_API_KEY || !PINECONE_ENVIRONMENT || !PINECONE_INDEX_NAME) {
        throw new Error('pinecone configuration is not defined')
    }

    const client = new PineconeClient()

    await client.init({
        apiKey: PINECONE_API_KEY,
        environment: PINECONE_ENVIRONMENT
    })


    const description = await client.describeIndex({ indexName: PINECONE_INDEX_NAME })

    if (!description.status?.ready) {
        throw new Error('pinecone index not ready')
    }

    const index = client.Index(PINECONE_INDEX_NAME)

    const stats = await index.describeIndexStats({
        describeIndexStatsRequest: {
            filter: {},
        },
    })

    if (stats.dimension != 1536) {
        throw new Error('pinecone index dimension should be 1536')
    }

    if (!stats.totalVectorCount) {
        console.log('populating pinecone index')

        const loader = new TextLoader(path.join(__dirname, 'docs.txt'))
        
        const docs = await loader.load()
    
        const splitter = new CharacterTextSplitter({ chunkSize: 600, chunkOverlap: 200, separator: ' ' })
    
        const chunks = await splitter.splitDocuments(docs)

        const db = await PineconeStore.fromDocuments(chunks, openAIEmbeddings, {
            pineconeIndex: index
        })

        return db
    }

    const db = await PineconeStore.fromExistingIndex(openAIEmbeddings, {
        pineconeIndex: index
    })

    return db
}

async function setup() {
    const db = await prepare()

    const app = express()
    app.use(cors({ origin: '*' }))

    const server = http.createServer(app)

    const io = new Server(server, {
        cors: {
            origin: '*',
            methods: ['GET'],
            credentials: false
        }
    })

    io.on('connection', (socket) => {
        const convo: Conversation = {
            history: new ChatMessageHistory()
        }

        convo.history.addAIChatMessage(initialMessage)

        socket.on('message', (msg: Message) => message(socket, convo, db, msg))

        socket.on('disconnect', async () => {
            socket.removeAllListeners()
            await convo.history.clear()
        })
    })

    const port = parseInt(process.env.PORT ?? "80")

    server.listen(port, async () => {
        console.log(`listening to port ${port}`)
    })
}

setup()