from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data and prepare model when the app starts
def prepare_model():
    # Load the dataset
    df = pd.read_csv('indian movies.csv')
    
    # Create formatted tags
    df['formatted_tags'] = df['Language'] + ', ' + df['Genre']
    df['formatted_tags'] = df['formatted_tags'].apply(lambda x: x.lower())
    
    # Keep necessary columns
    df = df[['ID', 'Movie Name', 'formatted_tags']]
    df['Movie Name_lower'] = df['Movie Name'].str.lower()
    df.reset_index(drop=True, inplace=True)
    
    # Convert tags to feature vectors
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['formatted_tags']).toarray()
    
    # Compute cosine similarity matrix
    similarity = cosine_similarity(vectors)
    
    return df, similarity

# Load data when starting the app
df, similarity = prepare_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie_name = request.form['movie_name'].strip().lower()
        results = recommend(movie_name)
        return render_template('index.html', results=results, search_query=movie_name)
    return render_template('index.html', results=None, search_query=None)

def recommend(movie):
    movie = movie.lower()
    results = {'found': False, 'input_movie': '', 'recommendations': []}
    
    if movie not in df['Movie Name_lower'].values:
        return results
    
    movie_index = df[df['Movie Name_lower'] == movie].index[0]
    results['input_movie'] = df.loc[movie_index, 'Movie Name']
    results['found'] = True
    
    distances = similarity[movie_index]
    
    # Get top 5 similar movies (excluding itself)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    movies_list = [i for i in movies_list if i[0] != movie_index][:5]
    
    for i in movies_list:
        recommended_movie = df.iloc[i[0]]['Movie Name']
        recommended_tags = df.iloc[i[0]]['formatted_tags']
        similarity_score = i[1]
        
        results['recommendations'].append({
            'movie': recommended_movie,
            'tags': recommended_tags,
            'score': f"{similarity_score:.4f}"
        })
    
    return results

if __name__ == '__main__':
    app.run(debug=True)