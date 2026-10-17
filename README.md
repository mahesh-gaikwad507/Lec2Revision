# Lecture-to-Revision AI

A Streamlit learning assistant that turns PDFs and YouTube transcripts
into summaries, question-answering sessions, flashcards, quizzes,
exam-style questions, and mind maps.

## Features

- Email verification with six-digit OTP
- Password hashing and password reset
- Subject- and topic-based learning workspaces
- PDF text extraction
- YouTube transcript import
- YouTube video discovery and engagement statistics
- Semantic retrieval using Sentence Transformers and FAISS
- Keyword retrieval using BM25
- AI answers and revision content using OpenRouter
- Persistent SQLite storage for accounts, materials, chats, and progress
- Markdown download for generated revision notes

## Requirements

- Python 3.10 or newer
- OpenRouter API key
- SMTP credentials for email verification and password reset
- YouTube Data API key for YouTube discovery

## Local setup

1. Create and activate a virtual environment.

   Windows:

       python -m venv .venv
       .venv\Scripts\activate

   macOS/Linux:

       python -m venv .venv
       source .venv/bin/activate

2. Install dependencies.

       pip install -r requirements.txt

3. Create the Streamlit configuration directory.

       mkdir .streamlit

4. Copy `.streamlit/secrets.toml.example` to
   `.streamlit/secrets.toml`.

5. Add your real API keys, SMTP credentials, and a random OTP_PEPPER
   to `.streamlit/secrets.toml`.

6. Run the application.

       streamlit run app.py

## Persistent data

By default, the SQLite database is stored at:

    data/lecture_revision.db

You can set LTR_DATA_DIR to change the data directory.

Back up the database regularly. Local filesystem storage may not persist
between deployments on some hosting platforms.

## Deployment

For Streamlit Community Cloud:

1. Push the project files to GitHub.
2. Create a new app from the repository.
3. Select `app.py` as the main file.
4. Configure the API keys and SMTP values in the app's Secrets settings.
5. Use managed persistent storage or a hosted database if data must survive
   redeployments and be shared across multiple app instances.

Do not commit `.streamlit/secrets.toml`, passwords, or API keys.

## Limitations

- Scanned PDFs require OCR before their text can be extracted.
- YouTube transcript import requires accessible captions.
- FAISS indexes are currently created at query time rather than persisted.
- Interactive quiz scoring, scheduled flashcard reviews, notifications,
  and image generation need additional implementation.
