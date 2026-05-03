import pandas as pd
import re
import spacy
import kagglehub
import os
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load lightweight spaCy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Load BERT model once
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


# TEXT CLEANING
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])



# SKILL EXTRACTION (NLP)
def extract_skills(text):
    doc = nlp(text[:1000])  # limit size for speed
    return list(set([
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN"]
    ]))


# EXPERIENCE EXTRACTION

def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s+years?', text.lower())
    return max([int(x) for x in matches]) if matches else 0


# LOAD JOB DATA
def load_jobs_data(path1, path2):
    jobs_df = pd.read_csv(path1)
    jobs2_df = pd.read_json(path2, lines=True)

    jobs_df.columns = jobs_df.columns.str.lower().str.strip()
    jobs2_df.columns = jobs2_df.columns.str.lower().str.strip()

    def combine(df, cols):
        available = [c for c in cols if c in df.columns]
        return df[available].fillna('').astype(str).agg(' '.join, axis=1)

    jobs_df['text'] = combine(jobs_df, ['title', 'description'])
    jobs2_df['text'] = combine(jobs2_df, ['job_title', 'job_description'])

    jobs_df['cleaned'] = jobs_df['text'].apply(clean_text)
    jobs2_df['cleaned'] = jobs2_df['text'].apply(clean_text)

    jobs = pd.concat([jobs_df, jobs2_df], ignore_index=True)
    jobs = jobs.drop_duplicates(subset=['cleaned']).reset_index(drop=True)

    # Limit for performance (important)
    jobs = jobs.sample(500, random_state=42)

    return jobs


# BUILD MODEL
class JobRecommender:

    def __init__(self, jobs_df):
        self.jobs_df = jobs_df

        print("Extracting job skills...")
        docs = list(nlp.pipe(self.jobs_df['text'], batch_size=50))

        self.jobs_df['skills'] = [
            list(set([t.lemma_.lower() for t in doc if t.pos_ in ["NOUN", "PROPN"]]))
            for doc in docs
        ]

        print("Extracting experience...")
        self.jobs_df['experience'] = self.jobs_df['text'].apply(extract_experience)

        print("Building TF-IDF...")
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.job_vectors = self.vectorizer.fit_transform(self.jobs_df['cleaned'])

        print("Building BERT embeddings...")
        self.job_embeddings = bert_model.encode(
            self.jobs_df['cleaned'].tolist(),
            batch_size=32,
            show_progress_bar=True
        )

        print("Model Ready!!")

    def recommend(self, resume_text, top_n=5):

        cleaned = clean_text(resume_text)
        resume_vec = self.vectorizer.transform([cleaned])
        resume_emb = bert_model.encode([cleaned])[0]

        resume_skills = extract_skills(resume_text)
        resume_exp = extract_experience(resume_text)

        tfidf_sim = cosine_similarity(resume_vec, self.job_vectors)[0]
        bert_sim = cosine_similarity([resume_emb], self.job_embeddings)[0]

        scores = []

        for i in range(len(self.jobs_df)):
            job_skills = self.jobs_df.iloc[i]['skills']
            job_exp = self.jobs_df.iloc[i]['experience']

            skill_overlap = len(set(resume_skills) & set(job_skills))
            exp_match = 1 if resume_exp >= job_exp else 0

            score = (
                0.48 * tfidf_sim[i] +
                0.38 * bert_sim[i] +
                0.09 * skill_overlap +
                0.05 * exp_match
            )

            scores.append(score)

        top_indices = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:top_n]

        results = self.jobs_df.iloc[top_indices].copy()
        results['score'] = [scores[i] for i in top_indices]

        return results[['title', 'company_name', 'location', 'score']].to_dict(orient="records")

def build_model():

    postings = kagglehub.dataset_download("arshkon/linkedin-job-postings")
    monster_jobs = kagglehub.dataset_download("promptcloud/monster-usa-job-postings-dataset")

    jobs = load_jobs_data(
        os.path.join(postings, "postings.csv"),
        os.path.join(monster_jobs, "Monster_usa_job_listings_dataset_20190601_20190930__20k_data.ldjson")
    )
    return JobRecommender(jobs)

def make_handler(model):
    class RequestHandler(BaseHTTPRequestHandler):
        def _set_json_headers(self, status_code):
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.end_headers()

        def _write_json(self, status_code, payload):
            self._set_json_headers(status_code)
            self.wfile.write(json.dumps(payload).encode("utf-8"))

        def do_OPTIONS(self):
            self._set_json_headers(200)

        def do_POST(self):
            if self.path != "/recommend":
                self._write_json(404, {"error": "Route not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._write_json(400, {"error": "Invalid JSON body"})
                return

            resume_text = payload.get("resume_text", "").strip()
            top_n = payload.get("top_n", 5)

            if not resume_text:
                self._write_json(400, {"error": "resume_text is required"})
                return

            if not isinstance(top_n, int) or top_n <= 0:
                self._write_json(400, {"error": "top_n must be a positive integer"})
                return

            results = model.recommend(resume_text, top_n=top_n)
            self._write_json(200, {"results": results})

        def log_message(self, format, *args):
            return

    return RequestHandler

def run_server(host="0.0.0.0", port=8080):
    model = build_model()
    handler = make_handler(model)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Job recommendation API listening on http://{host}:{port}")
    server.serve_forever()

def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    run_server(host=host, port=port)

if __name__ == "__main__":
    main()
