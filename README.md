# Lecture-to-Revision AI 🎓

An AI-powered study assistant that converts YouTube lectures and PDF documents into revision material.

## Features

- Import YouTube transcripts
- Upload text-based PDF documents
- Ask questions about study material using OpenRouter
- Generate structured revision notes
- Create interactive flashcards
- Generate multiple-choice quizzes
- Support English, Hindi, and Marathi responses
- Download revision notes as Markdown
- Dark-themed Streamlit interface

## Technology Stack

- Python
- Streamlit
- OpenRouter API
- SQLite
- YouTube Transcript API
- PyMuPDF

## Run Locally

1. Clone this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure the `OPENROUTER_API_KEY` environment variable.
4. Start the application:

   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Secrets

Add your API key in the Streamlit Cloud app's Secrets settings:

```toml
OPENROUTER_API_KEY = "your-key"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

Never commit API keys, database files, or personal information.

## Important Notes

- YouTube transcript availability depends on the video and supported languages.
- Scanned PDFs may require OCR before their text can be extracted.
- The current app uses lightweight keyword retrieval. Semantic FAISS + BM25 retrieval and persistent cloud storage can be added in a later integration step.
- Local SQLite files on Streamlit Cloud should not be treated as permanent storage.

## License

Add a license before distributing this project publicly.
