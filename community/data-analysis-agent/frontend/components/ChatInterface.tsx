"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Send, Code2, Brain } from "lucide-react"
import type { ChatMessage } from "@/lib/types"

interface ChatInterfaceProps {
  messages: ChatMessage[]
  onSendMessage: (message: string) => void
  isLoading?: boolean
}

export default function ChatInterface({ messages, onSendMessage, isLoading }: ChatInterfaceProps) {
  const [input, setInput] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput("")
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-muted-foreground mt-12">
            <p className="text-lg">Start asking questions about your data!</p>
            <p className="text-sm mt-2">Try asking about trends, correlations, or specific insights.</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-3xl rounded-xl p-4 ${
                  message.role === "user"
                    ? "bg-primary text-primary-foreground shadow-lg"
                    : "bg-card text-card-foreground border border-border shadow-sm"
                }`}
              >
                <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>

                {/* Display plot if available */}
                {message.plot && (
                  <div className="mt-4 rounded-lg overflow-hidden border border-border">
                    <img src={`data:image/png;base64,${message.plot}`} alt="Visualization" className="w-full" />
                  </div>
                )}

                {message.thinking && (
                  <details className="mt-4 border-t border-border/50 pt-3">
                    <summary className="cursor-pointer flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                      <Brain className="w-4 h-4" />
                      Model Thinking
                    </summary>
                    <pre className="mt-3 text-xs bg-secondary/50 p-3 rounded-lg overflow-x-auto font-mono border border-border">
                      {message.thinking}
                    </pre>
                  </details>
                )}

                {/* Collapsible code section */}
                {message.code && (
                  <details className="mt-4 border-t border-border/50 pt-3">
                    <summary className="cursor-pointer flex items-center gap-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                      <Code2 className="w-4 h-4" />
                      View Code
                    </summary>
                    <pre className="mt-3 text-xs bg-[#0d1117] text-gray-100 p-4 rounded-lg overflow-x-auto font-mono border border-border">
                      <code>{message.code}</code>
                    </pre>
                  </details>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-card border border-border rounded-xl p-4 shadow-sm">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="border-t border-border p-4 bg-card/30">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your data..."
            disabled={isLoading}
            className="flex-1 px-4 py-3 bg-secondary border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 text-foreground placeholder:text-muted-foreground transition-all"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-5 py-3 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 font-medium shadow-sm hover:shadow-md transition-all"
          >
            <Send className="w-4 h-4" />
            <span className="hidden sm:inline">Send</span>
          </button>
        </div>
      </form>
    </div>
  )
}
