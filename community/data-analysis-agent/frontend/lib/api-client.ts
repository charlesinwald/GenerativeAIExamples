// API client for Data Analysis Agent backend

import axios from 'axios';
import type { ModelInfo, DatasetUploadResponse, QueryRequest, QueryResponse } from './types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Get available models
  getModels: async (): Promise<ModelInfo[]> => {
    const response = await apiClient.get<ModelInfo[]>('/api/models');
    return response.data;
  },

  // Upload dataset
  uploadDataset: async (file: File, modelKey: string): Promise<DatasetUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<DatasetUploadResponse>(
      `/api/upload?model_key=${modelKey}`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  },

  // Query dataset
  queryDataset: async (request: QueryRequest): Promise<QueryResponse> => {
    const response = await apiClient.post<QueryResponse>('/api/query', request);
    return response.data;
  },

  // Delete dataset
  deleteDataset: async (datasetId: string): Promise<void> => {
    await apiClient.delete(`/api/dataset/${datasetId}`);
  },
};

export default apiClient;
