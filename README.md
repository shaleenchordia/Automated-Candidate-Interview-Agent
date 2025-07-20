# HEYI: AI-Powered Interview Assistant

## Overview

**HEYI** is a production-ready AI Interview Assistant designed for technical hiring. It leverages modern NLP, LLMs, and speech-to-text technologies to automate candidate screening, skill detection, and personalized interview question generation.

---

## Features

- **Resume Parsing & Skill Detection:**  
  Extracts candidate skills and experience levels from uploaded resumes using a custom weighted keyword and embedding-based system.

- **Dynamic Question Generation:**  
  Uses local LLMs (Ollama with Llama 3.2) to generate unique, job-category-specific interview questions tailored to each candidate’s detected skills and experience.

- **Speech-to-Text (Voice Input):**  
  Integrates local [OpenAI Whisper](https://github.com/openai/whisper) for free, offline transcription of candidate audio responses.

- **Interview Evaluation:**  
  Scores candidate responses and stores results in MongoDB, focusing on top skills and job category preference for efficient analytics.

- **Streamlit Web Interface:**  
  Provides an interactive, user-friendly UI for candidates and interviewers.

---

## Technical Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend:** Python 3.9+, modular service architecture
- **LLM:** [Ollama](https://ollama.com/) (Llama 3.2, local inference)
- **Speech-to-Text:** [OpenAI Whisper](https://github.com/openai/whisper) (local, free)
- **Database:** MongoDB (candidate profiles, interview results)
- **ML Pipeline:** Custom skill detection, extensible for future model training

---

## Folder Structure

```
backend/
├── main_app.py                # Streamlit main application
├── requirements.txt           # Python dependencies
├── config.py                  # Configuration settings
├── app/
│   ├── services/              # Modular service layer
│   │   ├── speech_service.py
│   │   ├── dynamic_question_generator.py
│   │   ├── resume_analyzer.py
│   │   ├── candidate_service.py
│   │   └── ... (other services)
│   ├── ml/                    # ML pipeline and skill detection
│   ├── utils/                 # Utility functions
│   └── __init__.py
└── .env                       # Environment variables (API keys, DB URI)
```

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd heyi/backend
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install openai-whisper
   brew install ffmpeg  # MacOS only; use apt-get for Linux
   ```

4. **Start Ollama (for LLM)**
   - Download and install [Ollama](https://ollama.com/download)
   - Pull the Llama 3.2 model:
     ```bash
     ollama pull llama3.2
     ```
   - Start Ollama server (usually runs on `localhost:11434`)

5. **Configure environment variables**
   - Create a `.env` file with your MongoDB URI and other settings.

6. **Run the Streamlit app**
   ```bash
   streamlit run main_app.py
   ```

---

## Usage

1. **Upload Resume:**  
   Candidate uploads their resume (PDF or text).

2. **Skill Detection:**  
   System analyzes resume and extracts top skills and experience level.

3. **Select Job Category:**  
   Candidate selects their preferred job category.

4. **Interview Questions:**  
   LLM generates unique, skill-based questions for the candidate.

5. **Voice Response:**  
   Candidate records or uploads audio answers; Whisper transcribes to text.

6. **Evaluation & Storage:**  
   Answers are scored and top skills + job category are stored in MongoDB.

---

## Data Storage (MongoDB)

Only essential data is stored for each candidate:
```json
{
  "name": "Candidate Name",
  "job_profile": {
    "category": "Software Development",
    "skills_identified": ["Python Programming", "SQL Databases"]
  },
  "skill_summary": {
    "top_2_skills": ["Python Programming", "SQL Databases"],
    "technical_level": "Junior",
    "job_category_preference": "Software Development",
    "interview_score": 8.5
  }
}
```

---

## Extensibility

- **ML Pipeline:** Easily add custom model training for skill classification or response evaluation.
- **LLM Models:** Swap out Ollama models or integrate OpenAI/HuggingFace APIs.
- **Speech-to-Text:** Upgrade Whisper model size for higher accuracy.

---
