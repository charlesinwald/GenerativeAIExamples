"use client"

import { useState, useEffect } from "react"
import FileUpload from "@/components/FileUpload"
import ModelSelector from "@/components/ModelSelector"
import ChatInterface from "@/components/ChatInterface"
import EDAPanel from "@/components/EDAPanel"
import { api } from "@/lib/api-client"
import type { ModelInfo, DatasetUploadResponse, ChatMessage } from "@/lib/types"
import { Sparkles } from "lucide-react"

export default function Home() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string>("llama-3-1-nemotron-ultra-v1")
  const [datasetData, setDatasetData] = useState<DatasetUploadResponse | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  // Fetch available models on mount
  useEffect(() => {
    api.getModels().then(setModels).catch(console.error)
  }, [])

  const handleFileSelect = async (file: File) => {
    setIsUploading(true)
    try {
      const response = await api.uploadDataset(file, selectedModel)
      setDatasetData(response)
      setMessages([])
    } catch (error) {
      console.error("Upload error:", error)
      alert("Failed to upload dataset. Please try again.")
    } finally {
      setIsUploading(false)
    }
  }

  const handleModelChange = async (modelKey: string) => {
    setSelectedModel(modelKey)

    // If dataset is already loaded, regenerate EDA with new model
    if (datasetData) {
      setIsUploading(true)
      try {
        // Re-upload to trigger EDA regeneration
        const response = await fetch(`http://localhost:8000/api/upload?model_key=${modelKey}`, {
          method: "POST",
          // Note: This is a simplified approach. In production, store the file or use a different endpoint
        })
        if (response.ok) {
          const data = await response.json()
          setDatasetData(data)
        }
      } catch (error) {
        console.error("Model change error:", error)
      } finally {
        setIsUploading(false)
      }
    }
  }

  const handleSendMessage = async (message: string) => {
    if (!datasetData) return

    // Add user message
    const userMessage: ChatMessage = { role: "user", content: message }
    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Get recent user messages for context
      const recentMessages = messages.filter((m) => m.role === "user").slice(-3)
      const chatContext = recentMessages.map((m) => m.content).join("\n")

      const response = await api.queryDataset({
        dataset_id: datasetData.dataset_id,
        query: message,
        model_key: selectedModel,
        chat_context: chatContext || undefined,
      })

      // Create assistant message
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: response.explanation,
        code: response.code,
        thinking: response.thinking,
        plot: response.result.type === "plot" ? response.result.data : undefined,
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error("Query error:", error)
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, an error occurred while processing your query. Please try again.",
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleClearChat = () => {
    setMessages([])
  }

  return (
    <div className="h-screen flex flex-col bg-background">
      <header className="bg-card border-b border-border px-6 py-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground tracking-tight">Data Analysis Agent</h1>
              <p className="text-sm text-muted-foreground mt-0.5">AI-powered insights with NVIDIA Llama Nemotron</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-72">
              <ModelSelector
                models={models}
                selectedModel={selectedModel}
                onModelChange={handleModelChange}
                disabled={isUploading}
              />
            </div>
            <FileUpload onFileSelect={handleFileSelect} disabled={isUploading} />
          </div>
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel - EDA */}
        <aside className="w-2/5 bg-card border-r border-border overflow-hidden flex flex-col">
          <div className="px-6 py-4 border-b border-border bg-card/50">
            <h2 className="text-lg font-semibold text-foreground">Dataset Overview</h2>
            {datasetData && <p className="text-sm text-muted-foreground mt-1 font-mono">{datasetData.filename}</p>}
          </div>
          {isUploading ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                <p className="text-muted-foreground">Analyzing dataset...</p>
              </div>
            </div>
          ) : (
            <EDAPanel data={datasetData} />
          )}
        </aside>

        {/* Right Panel - Chat */}
        <main className="flex-1 bg-background overflow-hidden flex flex-col">
          <div className="px-6 py-4 border-b border-border flex items-center justify-between bg-card/30">
            <h2 className="text-lg font-semibold text-foreground">Chat with your data</h2>
            {messages.length > 0 && (
              <button
                onClick={handleClearChat}
                className="text-sm text-muted-foreground hover:text-foreground px-3 py-1.5 rounded-lg hover:bg-secondary/50 transition-colors"
              >
                Clear chat
              </button>
            )}
          </div>
          {!datasetData ? (
            <div className="flex-1 flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <Sparkles className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-lg">Upload a dataset to start chatting</p>
              </div>
            </div>
          ) : (
            <ChatInterface messages={messages} onSendMessage={handleSendMessage} isLoading={isLoading} />
          )}
        </main>
      </div>
    </div>
  )
}
