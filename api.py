from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

print("Loading models...")

ratings_df = pd.read_csv("ratings.csv")
books_df = pd.read_csv("books.csv")

bookid_to_index = pd.Series(books_df.index, index=books_df['book_id']).to_dict()

with open("Pickle/book_content_cosine_sim_matrix.pkl", 'rb') as f:
    content_sim_matrix = pickle.load(f)

with open("Pickle/book_cf_model.pkl", 'rb') as f:
    cf_model = pickle.load(f)

print("Models loaded!")

def hybrid_recommend_book(user_id, book_title, alpha=0.5, num_recommendations=10):
    book_title_norm = book_title.lower().strip()
    book_row = books_df[books_df['title'].str.lower().str.strip() == book_title_norm]
    
    if book_row.empty:
        return []
    
    book_id = int(book_row['book_id'].values[0])
    book_idx = bookid_to_index.get(book_id)
    
    if book_idx is None:
        return []
    
    book_similarities = content_sim_matrix[book_idx]
    content_min, content_max = book_similarities.min(), book_similarities.max()
    content_norm = (book_similarities - content_min) / (content_max - content_min + 1e-8)
    similar_indices = content_norm.argsort()[::-1]
    
    def get_cf_score(uid, bid):
        try:
            pred = cf_model.predict(uid, bid)
            return (pred.est - 1) / 4.0
        except:
            return 0.0
    
    results = []
    for idx in similar_indices:
        rec_book_id = int(books_df.iloc[idx]['book_id'])
        if rec_book_id == book_id:
            continue
        
        content_score = float(content_norm[idx])
        collab_score = get_cf_score(user_id, rec_book_id)
        hybrid_score = alpha * content_score + (1 - alpha) * collab_score
        
        results.append({
            'book_id': rec_book_id,
            'title': books_df.iloc[idx]['title'],
            'authors': books_df.iloc[idx]['authors'],
            'hybrid_score': float(hybrid_score)
        })
        
        if len(results) >= num_recommendations:
            break
    
    return sorted(results, key=lambda x: x['hybrid_score'], reverse=True)

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "Book API Running"})

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = data.get('user_id', 1)
        book_title = data.get('book_title')
        alpha = float(data.get('alpha', 0.5))
        num_recs = int(data.get('num_recommendations', 10))
        
        if not book_title:
            return jsonify({"error": "book_title required"}), 400
        
        recommendations = hybrid_recommend_book(user_id, book_title, alpha, num_recs)
        
        return jsonify({
            "status": "success",
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/books/search', methods=['GET'])
def search_books():
    try:
        query = request.args.get('query', '').lower()
        
        if not query:
            return jsonify({"error": "query required"}), 400
        
        matches = books_df[books_df['title'].str.lower().str.contains(query, na=False)]
        results = matches.head(20)[['book_id', 'title', 'authors']].to_dict('records')
        
        return jsonify({"status": "success", "books": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


