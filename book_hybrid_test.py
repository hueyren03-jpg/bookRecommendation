"""
Book Hybrid Recommendation System - Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

# Set page config
st.set_page_config(
    page_title="Book Recommender System",
    page_icon="📚",
    layout="wide"
)

# Set random seed
random.seed(42)
np.random.seed(42)

# ==============================================================================
# LOAD DATA AND MODELS
# ==============================================================================

@st.cache_resource
def load_data():
    """Load all data and models"""
    try:
        # Load data
        ratings_df = pd.read_csv("ratings.csv")
        books_df = pd.read_csv("books.csv")
        
        # Create mappings
        bookid_to_index = pd.Series(books_df.index, index=books_df['book_id']).to_dict()
        index_to_bookid = pd.Series(books_df['book_id'].values, index=books_df.index).to_dict()
        
        # Load models
        with open("Pickle/book_content_cosine_sim_matrix.pkl", 'rb') as f:
            content_sim_matrix = pickle.load(f)
        
        with open("Pickle/book_cf_model.pkl", 'rb') as f:
            cf_model = pickle.load(f)
        
        return ratings_df, books_df, bookid_to_index, index_to_bookid, content_sim_matrix, cf_model
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load data
with st.spinner("Loading models and data..."):
    ratings_df, books_df, bookid_to_index, index_to_bookid, content_sim_matrix, cf_model = load_data()

# ==============================================================================
# HYBRID RECOMMENDATION FUNCTION
# ==============================================================================

def hybrid_recommend_book(user_id, book_title, alpha=0.5, num_recommendations=10):
    """Generate hybrid recommendations"""
    
    book_title_norm = book_title.lower().strip()
    book_row = books_df[books_df['title'].str.lower().str.strip() == book_title_norm]
    
    if book_row.empty:
        return pd.DataFrame()
    
    book_id = int(book_row['book_id'].values[0])
    book_idx = bookid_to_index.get(book_id)
    
    if book_idx is None:
        return pd.DataFrame()
    
    # Content-based similarities
    book_similarities = content_sim_matrix[book_idx]
    content_min, content_max = book_similarities.min(), book_similarities.max()
    content_norm = (book_similarities - content_min) / (content_max - content_min + 1e-8)
    similar_indices = content_norm.argsort()[::-1]
    
    # Collaborative filtering scores
    def get_cf_score(user_id, book_id):
        try:
            pred = cf_model.predict(str(user_id), str(book_id))
            return (pred.est - 1) / 4
        except:
            return 0
    
    # Collect recommendations
    valid_indices = []
    for idx in similar_indices:
        rec_book_id = int(books_df.iloc[idx]['book_id'])
        if rec_book_id != book_id:
            valid_indices.append(idx)
        if len(valid_indices) >= num_recommendations:
            break
    
    # Build results
    results = []
    for idx in valid_indices:
        rec_book_id = int(books_df.iloc[idx]['book_id'])
        rec_title = books_df.iloc[idx]['title']
        rec_authors = books_df.iloc[idx]['authors']
        
        content_score = float(content_norm[idx])
        collab_score = get_cf_score(user_id, rec_book_id)
        hybrid_score = alpha * content_score + (1 - alpha) * collab_score
        
        content_pct = alpha * content_score / hybrid_score * 100 if hybrid_score > 0 else 0
        collab_pct = (1 - alpha) * collab_score / hybrid_score * 100 if hybrid_score > 0 else 0
        
        results.append({
            'book_id': rec_book_id,
            'title': rec_title,
            'authors': rec_authors,
            'hybrid_score': hybrid_score,
            'content_score': content_score,
            'collab_score': collab_score,
            'content_pct': content_pct,
            'collab_pct': collab_pct
        })
    
    return pd.DataFrame(results).sort_values(by='hybrid_score', ascending=False).head(num_recommendations)

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

# Title
st.title("📚 Book Hybrid Recommendation System")
st.markdown("*Combining Content-Based Filtering + Collaborative Filtering*")

# Sidebar
st.sidebar.header("⚙️ Settings")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📊 Model Performance", "ℹ️ About"])

# ==============================================================================
# TAB 1: GET RECOMMENDATIONS
# ==============================================================================

with tab1:
    st.header("Get Book Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User ID input
        user_id = st.number_input(
            "Enter User ID",
            min_value=1,
            max_value=int(ratings_df['user_id'].max()),
            value=19643,
            help="Enter a user ID from the dataset"
        )
        
        # Book title input (searchable dropdown)
        book_titles = books_df['title'].tolist()
        selected_book = st.selectbox(
            "Select a Book",
            options=book_titles,
            index=book_titles.index("Harry Potter and the Half-Blood Prince (Harry Potter, #6)") if "Harry Potter and the Half-Blood Prince (Harry Potter, #6)" in book_titles else 0,
            help="Start typing to search for a book"
        )
    
    with col2:
        # Alpha slider
        alpha = st.slider(
            "Alpha (Content Weight)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = Pure Collaborative, 1 = Pure Content-Based"
        )
        
        # Number of recommendations
        num_recs = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=5
        )
    
    # Get recommendations button
    if st.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            recommendations = hybrid_recommend_book(user_id, selected_book, alpha, num_recs)
        
        if not recommendations.empty:
            st.success(f"✅ Found {len(recommendations)} recommendations!")
            
            # Display recommendations
            st.subheader(f"Top {num_recs} Recommendations")
            
            for idx, row in recommendations.iterrows():
                with st.expander(f"**{idx+1}. {row['title']}**", expanded=(idx < 3)):
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.markdown(f"**Author:** {row['authors']}")
                        st.markdown(f"**Hybrid Score:** {row['hybrid_score']:.4f}")
                        
                        # Score breakdown
                        st.progress(row['content_pct']/100, text=f"Content: {row['content_pct']:.1f}%")
                        st.progress(row['collab_pct']/100, text=f"Collaborative: {row['collab_pct']:.1f}%")
                    
                    with col_b:
                        # Pie chart for score breakdown
                        fig = go.Figure(data=[go.Pie(
                            labels=['Content', 'Collaborative'],
                            values=[row['content_pct'], row['collab_pct']],
                            hole=0.4,
                            marker_colors=['#FF6B6B', '#4ECDC4']
                        )])
                        fig.update_layout(
                            showlegend=False,
                            height=150,
                            margin=dict(l=0, r=0, t=0, b=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Could not generate recommendations. Please try a different book.")

# ==============================================================================
# TAB 2: MODEL PERFORMANCE
# ==============================================================================

with tab2:
    st.header("Model Performance Evaluation")
    
    # Evaluation settings
    st.sidebar.subheader("📊 Evaluation Settings")
    num_test_users = st.sidebar.slider("Number of Test Users", 10, 200, 100, 10)
    eval_threshold = st.sidebar.slider("Similarity Threshold", 0.01, 0.20, 0.095, 0.005)
    
    if st.button("🚀 Run Evaluation", type="primary"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Evaluation function
        def evaluate_hybrid(test_users, alpha=0.5, k_values=[5, 10, 15, 20], threshold=0.095):
            precision_scores = {k: [] for k in k_values}
            recall_scores = {k: [] for k in k_values}
            evaluated_count = 0
            total = len(test_users)
            
            for i, user_id in enumerate(test_users):
                # Update progress
                progress_bar.progress((i + 1) / total)
                status_text.text(f"Evaluating user {i+1}/{total}...")
                
                user_liked_books = ratings_df[
                    (ratings_df['user_id'] == user_id) & 
                    (ratings_df['rating'] >= 4)
                ]['book_id'].astype(int).tolist()
                
                if len(user_liked_books) < 3:
                    continue
                
                split_point = max(1, int(len(user_liked_books) * 0.6))
                train_books = user_liked_books[:split_point]
                test_books = set(user_liked_books[split_point:])
                
                for seed_book_id in train_books[:3]:
                    seed_title_rows = books_df.loc[books_df['book_id'] == seed_book_id, 'title'].values
                    if len(seed_title_rows) == 0:
                        continue
                    seed_title = seed_title_rows[0]
                    
                    for k in k_values:
                        recommended_df = hybrid_recommend_book(user_id, seed_title, alpha, k)
                        
                        if recommended_df.empty:
                            continue
                        
                        recommended_list = recommended_df['book_id'].astype(int).tolist()
                        evaluated_count += 1
                        
                        relevant_count = 0
                        for rec_book_id in recommended_list:
                            if rec_book_id in test_books:
                                relevant_count += 1
                                continue
                            
                            for test_book_id in test_books:
                                idx_rec = bookid_to_index.get(rec_book_id)
                                idx_test = bookid_to_index.get(test_book_id)
                                
                                if idx_rec is None or idx_test is None:
                                    continue
                                
                                sim_score = content_sim_matrix[idx_rec, idx_test]
                                if sim_score >= threshold:
                                    relevant_count += 1
                                    break
                        
                        precision = relevant_count / len(recommended_list) if recommended_list else 0
                        recall = relevant_count / len(test_books) if test_books else 0
                        
                        precision_scores[k].append(precision)
                        recall_scores[k].append(recall)
            
            avg_precision = {k: np.mean(precision_scores[k]) if precision_scores[k] else 0 for k in k_values}
            avg_recall = {k: np.mean(recall_scores[k]) if recall_scores[k] else 0 for k in k_values}
            
            return avg_precision, avg_recall, evaluated_count
        
        # Run evaluation
        test_users = ratings_df['user_id'].unique()[:num_test_users]
        precision, recall, eval_count = evaluate_hybrid(test_users, alpha=0.5, threshold=eval_threshold)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Evaluation complete! ({eval_count} evaluations)")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Precision @ K")
            k_values = [5, 10, 15, 20]
            
            # Metrics
            for k in k_values:
                st.metric(
                    f"Precision@{k}",
                    f"{precision[k]:.4f}",
                    f"{precision[k]*100:.2f}%"
                )
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=k_values,
                y=[precision[k] for k in k_values],
                mode='lines+markers',
                name='Precision',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title="Precision vs K",
                xaxis_title="K",
                yaxis_title="Precision",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Recall @ K")
            
            # Metrics
            for k in k_values:
                st.metric(
                    f"Recall@{k}",
                    f"{recall[k]:.4f}",
                    f"{recall[k]*100:.2f}%"
                )
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=k_values,
                y=[recall[k] for k in k_values],
                mode='lines+markers',
                name='Recall',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title="Recall vs K",
                xaxis_title="K",
                yaxis_title="Recall",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Precision-Recall Curve
        st.subheader("📉 Precision-Recall Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[recall[k] for k in k_values],
            y=[precision[k] for k in k_values],
            mode='lines+markers',
            name='P-R Curve',
            line=dict(color='#9B59B6', width=3),
            marker=dict(size=10),
            text=[f"k={k}" for k in k_values],
            textposition="top center"
        ))
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison
        st.subheader("🏆 Performance Comparison")
        comparison_df = pd.DataFrame({
            'System': ['Recipe Baseline', 'Your Hybrid Model'],
            'Precision@5': [0.4017, precision[5]],
            'Precision@10': [0.3089, precision[10]]
        })
        
        fig = go.Figure()
        for col in ['Precision@5', 'Precision@10']:
            fig.add_trace(go.Bar(
                name=col,
                x=comparison_df['System'],
                y=comparison_df[col],
                text=comparison_df[col].apply(lambda x: f"{x:.4f}"),
                textposition='auto'
            ))
        fig.update_layout(
            title="Your System vs Recipe Baseline",
            yaxis_title="Precision",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 3: ABOUT
# ==============================================================================

with tab3:
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### 📚 Book Hybrid Recommendation System
    
    This system combines two powerful recommendation techniques:
    
    #### 1️⃣ Content-Based Filtering
    - Uses book features (genres, descriptions, metadata)
    - Finds books similar to what you liked
    - Good for new users with few ratings
    
    #### 2️⃣ Collaborative Filtering
    - Uses ratings from similar users
    - Predicts what you'll rate highly
    - Discovers unexpected books you might love
    
    #### 🔄 Hybrid Approach
    The system combines both methods using a weighted average:
    ```
    Hybrid Score = α × Content Score + (1-α) × Collaborative Score
    ```
    
    Where α (alpha) controls the balance between the two approaches.
    
    ---
    
    ### 📊 Dataset Information
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Total Books", f"{len(books_df):,}")
    
    with col2:
        st.metric("⭐ Total Ratings", f"{len(ratings_df):,}")
    
    with col3:
        st.metric("👥 Total Users", f"{ratings_df['user_id'].nunique():,}")
    
    st.markdown("""
    ---
    
    ### 🎯 How to Use
    
    1. **Get Recommendations Tab:**
       - Enter a User ID
       - Select a book you like
       - Adjust alpha to control recommendation style
       - Click "Get Recommendations"
    
    2. **Model Performance Tab:**
       - Configure evaluation settings in sidebar
       - Click "Run Evaluation" to test model performance
       - View precision, recall, and comparison charts
    
    ---
    
    ### 🔧 Technical Details
    
    - **Content Similarity:** TF-IDF + Cosine Similarity
    - **Collaborative Model:** SVD Matrix Factorization
    - **Evaluation Method:** 60/40 Train-Test Split
    - **Similarity Threshold:** 0.095 (80th percentile)
    
    ---
    
    Made with ❤️ using Streamlit
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Info")
st.sidebar.info(f"""
**Books:** {len(books_df):,}  
**Ratings:** {len(ratings_df):,}  
**Users:** {ratings_df['user_id'].nunique():,}
""")