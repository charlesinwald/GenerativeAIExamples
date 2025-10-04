# Quick Start Guide - Next.js Frontend

This guide will help you get the Next.js version of the Data Analysis Agent up and running in minutes.

## Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA API Key ([Get one here](https://build.nvidia.com/))

## Option 1: Manual Setup (Recommended for Development)

### Step 1: Set Environment Variable

```bash
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

### Step 2: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### Step 3: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 4: Start Both Servers

#### Terminal 1 - Backend
```bash
cd backend
python main.py
```

Backend will run on: http://localhost:8000

#### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

Frontend will run on: http://localhost:3000

### Step 5: Open Your Browser

Navigate to: **http://localhost:3000**

---

## Option 2: Using Startup Script

```bash
# Set API key
export NVIDIA_API_KEY="your_nvidia_api_key_here"

# Make script executable (first time only)
chmod +x start.sh

# Run the script
./start.sh
```

This will start both servers automatically.

---

## Option 3: Using Docker Compose

```bash
# Set API key in environment
export NVIDIA_API_KEY="your_nvidia_api_key_here"

# Start services
docker-compose up
```

Frontend: http://localhost:3000
Backend: http://localhost:8000

---

## Usage

1. **Select a Model**: Choose from NVIDIA Llama 3.1 Nemotron Ultra or Llama 3.3 Nemotron Super

2. **Upload Dataset**: Click "Upload CSV File" and select a CSV file
   - Try the Titanic dataset: `wget https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv`

3. **View EDA**: The left panel will automatically show:
   - Dataset preview
   - Statistical summary
   - Data quality metrics
   - Column analysis
   - AI-generated insights

4. **Ask Questions**: Use natural language in the chat:
   - "What is the average age by class?"
   - "Show me a bar chart of survival rates by gender"
   - "Which features have missing values?"
   - "Plot the distribution of fare prices"

5. **View Results**:
   - See explanations in the chat
   - View visualizations inline
   - Expand "View Code" to see generated Python
   - Expand "Model Thinking" to see reasoning process

---

## Troubleshooting

### Backend won't start
- Check if NVIDIA_API_KEY is set: `echo $NVIDIA_API_KEY`
- Verify Python version: `python --version` (should be 3.10+)
- Install dependencies: `cd backend && pip install -r requirements.txt`

### Frontend won't start
- Verify Node.js version: `node --version` (should be 18+)
- Install dependencies: `cd frontend && npm install`
- Check if port 3000 is available

### Can't connect to backend
- Ensure backend is running on port 8000
- Check `.env.local` in frontend folder has: `NEXT_PUBLIC_API_URL=http://localhost:8000`

### CORS errors
- Make sure both servers are running
- Frontend should be on `localhost:3000`
- Backend should be on `localhost:8000`

---

## Project Structure

```
data-analysis-agent/
├── backend/              # FastAPI backend
│   ├── main.py          # API endpoints
│   ├── agents.py        # LLM agents
│   └── requirements.txt
├── frontend/            # Next.js frontend
│   ├── app/            # Pages
│   ├── components/     # React components
│   ├── lib/           # Utilities & API client
│   └── package.json
├── start.sh            # Startup script
└── docker-compose.yml  # Docker configuration
```

---

## Next Steps

- Explore different datasets
- Try various analysis questions
- Switch between different models
- Customize the UI in `frontend/components/`
- Add new API endpoints in `backend/main.py`

For more details, see [README_NEXTJS.md](./README_NEXTJS.md)
