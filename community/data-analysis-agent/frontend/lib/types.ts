// Type definitions for the Data Analysis Agent

export interface ModelInfo {
  key: string;
  name: string;
  url: string;
}

export interface DatasetUploadResponse {
  dataset_id: string;
  filename: string;
  preview: Record<string, any>[];
  columns: string[];
  shape: [number, number];
  eda_results: EDAResults;
  insights: string;
}

export interface EDAResults {
  statistical_summary: StatisticalSummary;
  data_profiling: DataProfiling;
  correlation_analysis: CorrelationAnalysis;
  distribution_analysis: DistributionAnalysis;
  visualizations?: {
    distribution_plots?: string[];
    correlation_heatmap?: string | null;
    categorical_plots?: string[];
  };
}

export interface StatisticalSummary {
  basic_info: {
    shape: [number, number];
    memory_usage: number;
    dtypes: Record<string, number>;
  };
  missing_data: {
    total_missing: number;
    missing_percentage: number;
    columns_with_missing: Record<string, number>;
  };
  numeric_summary?: Record<string, any>;
  categorical_summary?: Record<string, any>;
}

export interface DataProfiling {
  data_quality: {
    duplicate_rows: number;
    duplicate_percentage: number;
    constant_columns: string[];
    high_cardinality: string[];
  };
  column_analysis: Record<string, ColumnAnalysis>;
}

export interface ColumnAnalysis {
  dtype: string;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  unique_percentage: number;
  most_common?: Record<string, number>;
  outliers?: number;
}

export interface CorrelationAnalysis {
  correlation_matrix?: Record<string, any>;
  high_correlations?: Array<{
    var1: string;
    var2: string;
    correlation: number;
  }>;
  numeric_columns?: string[];
  message?: string;
}

export interface DistributionAnalysis {
  [key: string]: {
    skewness: number;
    kurtosis: number;
    normality_test: string;
  } | { message: string };
}

export interface QueryRequest {
  dataset_id: string;
  query: string;
  model_key: string;
  chat_context?: string;
}

export interface QueryResponse {
  code: string;
  result: ExecutionResult;
  thinking: string;
  explanation: string;
  should_plot: boolean;
}

export interface ExecutionResult {
  type?: string;
  data?: any;
  shape?: [number, number];
  length?: number;
  error?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  plot?: string;
  code?: string;
  thinking?: string;
}
