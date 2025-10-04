"use client"
import type { ModelInfo } from "@/lib/types"

interface ModelSelectorProps {
  models: ModelInfo[]
  selectedModel: string
  onModelChange: (modelKey: string) => void
  disabled?: boolean
}

export default function ModelSelector({ models, selectedModel, onModelChange, disabled }: ModelSelectorProps) {
  return (
    <div className="w-full">
      <label className="block text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">AI Model</label>
      <select
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        disabled={disabled}
        className="w-full px-3 py-2 bg-secondary border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed text-foreground text-sm transition-all"
      >
        {models.map((model) => (
          <option key={model.key} value={model.key}>
            {model.name}
          </option>
        ))}
      </select>
    </div>
  )
}
