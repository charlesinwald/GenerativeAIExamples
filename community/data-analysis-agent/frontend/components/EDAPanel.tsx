"use client"
import type { DatasetUploadResponse } from "@/lib/types"
import { Database, AlertCircle, TrendingUp, BarChart3, Activity } from "lucide-react"

interface EDAPanelProps {
  data: DatasetUploadResponse | null
}

export default function EDAPanel({ data }: EDAPanelProps) {
  if (!data) {
    return (
      <div className="p-8 text-center text-muted-foreground h-full flex items-center justify-center">
        <div>
          <Database className="w-16 h-16 mx-auto mb-4 opacity-30" />
          <p className="text-lg">Upload a CSV file to begin analysis</p>
        </div>
      </div>
    )
  }

  const { shape, eda_results, insights, preview, columns } = data
  const stats = eda_results.statistical_summary
  const profiling = eda_results.data_profiling

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <section>
        <h2 className="text-base font-semibold mb-3 flex items-center gap-2 text-foreground">
          <Database className="w-4 h-4 text-primary" />
          Dataset Preview
        </h2>
        <div className="bg-card rounded-xl border border-border overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-border">
              <thead className="bg-secondary/50">
                <tr>
                  {columns.slice(0, 5).map((col) => (
                    <th
                      key={col}
                      className="px-4 py-3 text-left text-xs font-semibold text-foreground uppercase tracking-wider"
                    >
                      {col}
                    </th>
                  ))}
                  {columns.length > 5 && (
                    <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground">
                      +{columns.length - 5} more
                    </th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {preview.slice(0, 5).map((row, idx) => (
                  <tr key={idx} className="hover:bg-secondary/30 transition-colors">
                    {columns.slice(0, 5).map((col) => (
                      <td key={col} className="px-4 py-3 text-sm text-foreground font-mono">
                        {String(row[col] ?? "N/A").substring(0, 30)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-base font-semibold mb-3 flex items-center gap-2 text-foreground">
          <TrendingUp className="w-4 h-4 text-primary" />
          Statistical Summary
        </h2>
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-gradient-to-br from-primary/10 to-primary/5 rounded-xl p-4 border border-primary/20">
            <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-1">Rows</p>
            <p className="text-2xl font-bold text-primary">{shape[0].toLocaleString()}</p>
          </div>
          <div className="bg-gradient-to-br from-chart-2/10 to-chart-2/5 rounded-xl p-4 border border-chart-2/20">
            <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-1">Columns</p>
            <p className="text-2xl font-bold text-chart-2">{shape[1]}</p>
          </div>
          <div className="bg-gradient-to-br from-chart-3/10 to-chart-3/5 rounded-xl p-4 border border-chart-3/20">
            <p className="text-xs text-muted-foreground font-medium uppercase tracking-wide mb-1">Memory</p>
            <p className="text-2xl font-bold text-chart-3">{stats.basic_info.memory_usage.toFixed(1)} MB</p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-base font-semibold mb-3 flex items-center gap-2 text-foreground">
          <AlertCircle className="w-4 h-4 text-primary" />
          Data Quality
        </h2>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-card rounded-xl border border-border p-4 hover:border-primary/50 transition-colors">
            <p className="text-xs text-muted-foreground font-medium mb-1">Missing Values</p>
            <p className="text-xl font-bold text-chart-4">{stats.missing_data.missing_percentage.toFixed(1)}%</p>
          </div>
          <div className="bg-card rounded-xl border border-border p-4 hover:border-primary/50 transition-colors">
            <p className="text-xs text-muted-foreground font-medium mb-1">Duplicates</p>
            <p className="text-xl font-bold text-destructive">
              {profiling.data_quality.duplicate_percentage.toFixed(1)}%
            </p>
          </div>
          <div className="bg-card rounded-xl border border-border p-4 hover:border-primary/50 transition-colors">
            <p className="text-xs text-muted-foreground font-medium mb-1">Constant Columns</p>
            <p className="text-xl font-bold text-foreground">{profiling.data_quality.constant_columns.length}</p>
          </div>
          <div className="bg-card rounded-xl border border-border p-4 hover:border-primary/50 transition-colors">
            <p className="text-xs text-muted-foreground font-medium mb-1">High Cardinality</p>
            <p className="text-xl font-bold text-foreground">{profiling.data_quality.high_cardinality.length}</p>
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-base font-semibold mb-3 flex items-center gap-2 text-foreground">
          <BarChart3 className="w-4 h-4 text-primary" />
          Column Analysis
        </h2>
        <div className="bg-card rounded-xl border border-border overflow-hidden shadow-sm">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-border">
              <thead className="bg-secondary/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                    Column
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                    Missing %
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                    Unique %
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {Object.entries(profiling.column_analysis)
                  .slice(0, 10)
                  .map(([col, analysis]) => (
                    <tr key={col} className="hover:bg-secondary/30 transition-colors">
                      <td className="px-4 py-3 text-sm font-medium text-foreground font-mono">{col}</td>
                      <td className="px-4 py-3 text-sm text-muted-foreground">
                        <span className="px-2 py-1 bg-secondary rounded text-xs font-medium">{analysis.dtype}</span>
                      </td>
                      <td className="px-4 py-3 text-sm text-foreground">{analysis.null_percentage.toFixed(1)}%</td>
                      <td className="px-4 py-3 text-sm text-foreground">{analysis.unique_percentage.toFixed(1)}%</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* High Correlations */}
      {eda_results.correlation_analysis.high_correlations &&
        eda_results.correlation_analysis.high_correlations.length > 0 && (
          <section>
            <h2 className="text-base font-semibold mb-3 flex items-center gap-2 text-foreground">
              <Activity className="w-4 h-4 text-primary" />
              High Correlations
            </h2>
            <div className="bg-card rounded-xl border border-border overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-border">
                  <thead className="bg-secondary/50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                        Variable 1
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                        Variable 2
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-semibold text-foreground uppercase tracking-wider">
                        Correlation
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {eda_results.correlation_analysis.high_correlations.map((corr, idx) => (
                      <tr key={idx} className="hover:bg-secondary/30 transition-colors">
                        <td className="px-4 py-3 text-sm text-foreground font-mono">{corr.var1}</td>
                        <td className="px-4 py-3 text-sm text-foreground font-mono">{corr.var2}</td>
                        <td className="px-4 py-3 text-sm font-bold text-primary">{corr.correlation.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}

      <section>
        <h2 className="text-base font-semibold mb-3 flex items-center gap-2 text-foreground">
          <span className="text-primary">💡</span>
          Suggested Analysis Questions
        </h2>
        <div className="bg-gradient-to-br from-primary/5 to-primary/10 rounded-xl p-5 border border-primary/20">
          <p className="text-sm whitespace-pre-wrap text-foreground leading-relaxed">{insights}</p>
        </div>
      </section>

      {/* Visualizations */}
    {eda_results.visualizations && (
      <section className="space-y-6">
        <h2 className="text-base font-semibold mb-3 flex items-center gap-2 text-foreground">
          <BarChart3 className="w-4 h-4 text-primary" />
          Visualizations
        </h2>
        
        {/* Distribution Plots */}
        {eda_results.visualizations.distribution_plots && eda_results.visualizations.distribution_plots.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-muted-foreground">Distribution Analysis</h3>
            <div className="space-y-4">
              {eda_results.visualizations.distribution_plots.map((imgBase64, idx) => (
                <div key={idx} className="bg-card rounded-xl border border-border overflow-hidden shadow-sm">
                  <img 
                    src={`data:image/png;base64,${imgBase64}`}
                    alt={`Distribution plot ${idx + 1}`}
                    className="w-full"
                  />
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Correlation Heatmap */}
        {eda_results.visualizations.correlation_heatmap && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-muted-foreground">Correlation Heatmap</h3>
            <div className="bg-card rounded-xl border border-border overflow-hidden shadow-sm">
              <img 
                src={`data:image/png;base64,${eda_results.visualizations.correlation_heatmap}`}
                alt="Correlation heatmap"
                className="w-full"
              />
            </div>
          </div>
        )}
        
        {/* Categorical Plots */}
        {eda_results.visualizations.categorical_plots && eda_results.visualizations.categorical_plots.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-muted-foreground">Categorical Analysis</h3>
            <div className="space-y-4">
              {eda_results.visualizations.categorical_plots.map((imgBase64, idx) => (
                <div key={idx} className="bg-card rounded-xl border border-border overflow-hidden shadow-sm">
                  <img 
                    src={`data:image/png;base64,${imgBase64}`}
                    alt={`Categorical plot ${idx + 1}`}
                    className="w-full"
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
)}
    </div>
  )
}
