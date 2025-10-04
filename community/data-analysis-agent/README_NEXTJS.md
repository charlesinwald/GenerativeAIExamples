# Data Analysis Agent - Next.js Frontend

A modern Next.js frontend for the Data Analysis Agent, replacing the Streamlit interface with a custom web application.

## Architecture

### Backend (FastAPI)
- **Location**: `backend/`
- **Main Files**:
  - `main.py` - FastAPI application with REST endpoints
  - `agents.py` - Refactored agent classes from original Streamlit app
  - `requirements.txt` - Python dependencies

### Frontend (Next.js)
- **Location**: `frontend/`
- **Tech Stack**:
  - Next.js 15 (App Router)
  - TypeScript
  - Tailwind CSS
  - Axios for API calls
  - Lucide React for icons

## Installation & Setup

### 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variable
export NVIDIA_API_KEY="your_nvidia_api_key"

# Run the FastAPI server
python main.py
```

The backend will run on `http://localhost:8000`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The frontend will run on `http://localhost:3000`

## Features

### Backend API Endpoints

- `GET /api/models` - Get available LLM models
- `POST /api/upload` - Upload CSV and generate EDA analysis
- `POST /api/query` - Process natural language queries
- `POST /api/chat` - Stream chat responses (for future enhancement)
- `DELETE /api/dataset/{id}` - Delete dataset from memory

### Frontend Components

1. **FileUpload** - Drag-and-drop CSV upload interface
2. **ModelSelector** - Dropdown to select NVIDIA Llama models
3. **EDAPanel** - Left sidebar showing:
   - Dataset preview
   - Statistical summary
   - Data quality metrics
   - Column analysis
   - High correlations
   - AI-generated insights

4. **ChatInterface** - Right panel with:
   - Chat history
   - Message input
   - Loading states
   - Collapsible code and thinking sections
   - Inline plot visualization

## Usage

1. **Start both servers**:
   ```bash
   # Terminal 1 - Backend
   cd backend && python main.py

   # Terminal 2 - Frontend
   cd frontend && npm run dev
   ```

2. **Open browser**: Navigate to `http://localhost:3000`

3. **Select model**: Choose from available NVIDIA Llama models

4. **Upload dataset**: Click "Upload CSV File" and select a CSV file

5. **Analyze data**:
   - View automatic EDA analysis in the left panel
   - Ask questions in natural language in the chat
   - View generated visualizations inline
   - Expand code/thinking sections to see details

## API Integration

The frontend communicates with the backend via REST API:

```typescript
// Example: Upload dataset
const response = await api.uploadDataset(file, selectedModel);

// Example: Query dataset
const response = await api.queryDataset({
  dataset_id: datasetId,
  query: "Show me the distribution of age",
  model_key: selectedModel
});
```

## Key Differences from Streamlit

| Feature | Streamlit | Next.js |
|---------|-----------|---------|
| Architecture | Monolithic | Separated frontend/backend |
| State Management | Session state | React state + API |
| Visualization | Matplotlib (server-side) | Base64 images from backend |
| Deployment | Single app | Frontend + Backend services |
| Customization | Limited | Full control over UI/UX |
| Performance | Page reloads | SPA with partial updates |

## Development Notes

### Adding New Features

1. **Backend**: Add endpoints in `backend/main.py`
2. **Frontend**:
   - Add types in `lib/types.ts`
   - Add API calls in `lib/api-client.ts`
   - Create/update components in `components/`

### Styling

The app uses Tailwind CSS. Modify styles in:
- `app/globals.css` - Global styles
- Component files - Component-specific styles

### Environment Variables

Create `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Production Deployment

### Backend
Deploy FastAPI with:
- Gunicorn/Uvicorn workers
- NGINX reverse proxy
- Redis for session storage (replace in-memory dict)

### Frontend
Deploy Next.js with:
- Vercel (recommended)
- Docker container
- Static export (`npm run build`)

Update `NEXT_PUBLIC_API_URL` to point to production backend.

## Troubleshooting

**CORS Errors**:
- Ensure backend CORS middleware allows frontend origin
- Check `allow_origins` in `backend/main.py`

**API Connection Failed**:
- Verify both servers are running
- Check `NEXT_PUBLIC_API_URL` in `.env.local`

**Module Not Found**:
- Run `npm install` in frontend directory
- Run `pip install -r requirements.txt` in backend directory

## License

Apache-2.0 (same as original project)
