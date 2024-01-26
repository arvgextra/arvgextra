import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ast
import dash
from dash import html
from dash import dcc
from dash import no_update
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
import plotly.express as px


# NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Goodreads dataframe from webscraping.py
goodreads = pd.read_csv('goodreads.csv')

recommended_columns = ['title', 'author', 'rating', 'main genre', 'similarity_score']

# Sentiment scores
sentiment_scores = []

# Stopwords to remove
to_remove = ["?", "!", ",", ".", "(", ")", "and"]

# Calculate sentiment scores
for desc in goodreads['description']:
    desc = desc.lower()
    for word in to_remove:
        desc = desc.replace(word, "")
    scores = analyzer.polarity_scores(desc)
    sentiment_scores.append(scores['compound'])

goodreads['sentiment'] = sentiment_scores

# Get full set of genres
full_genres = set()
for genres in goodreads["genres"]:
    genres = ast.literal_eval(genres)
    full_genres.update(genres)

# List of genres
genres = list(full_genres)

# Create dataframe with genre columns
genre_columns = [pd.Series([1 if genre in x else 0 for x in goodreads['genres']],
                           name=genre) for genre in genres]
genre_df = pd.concat(genre_columns, axis=1)

# Concatenate dataframes
goodreads = pd.concat([goodreads, genre_df], axis=1)

# Clustering features
features = ["sentiment", "rating", "pages"] + genres

for genre in genres:
    features.append(genre)

x = goodreads.loc[:, features].values

# keys are k (number of clusters), values are mean_d (mean distance
# from each sample to its cluster centroid)
mean_distance_dict = dict()
for n_clusters in range(1, 8):
    # fit kmeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(x)
    y = kmeans.predict(x)

    # compute & store mean distance
    mean_distance = -kmeans.score(x)
    mean_distance_dict[n_clusters] = mean_distance

# get k values and mean distance from dictionary
k_value = list((mean_distance_dict.keys()))
distance_mean = list((mean_distance_dict.values()))

# Choose the number of clusters based on the elbow point (e.g., 2 or 3)
final_clusters = 3

# Create the final KMeans model
final_kmeans = KMeans(n_clusters=final_clusters)
final_kmeans.fit(x)

# Get the cluster labels for each data point
final_labels = final_kmeans.labels_

# Add the cluster labels to your original DataFrame (assuming 'goodreads' is your original DataFrame)
goodreads['cluster'] = final_labels

# cluster centroids
final_centroids = final_kmeans.cluster_centers_
inertia = final_kmeans.inertia_

# Features to visualize
features_to_visualize = features

# 10 clusters
final_clusters = 10

# Create the final KMeans model
final_kmeans = KMeans(n_clusters=final_clusters)
final_kmeans.fit(x)

# Get the cluster labels for each data point
final_labels = final_kmeans.labels_

# Add the cluster labels to df
goodreads['cluster'] = final_labels

# distances to centroids
distances_to_centroids = final_kmeans.transform(x)


def recommend_books(user_likes, clustering_model, dataframe, features, num_recommendations):
    """
    :param user_likes: list of book/books the user likes
    :param clustering_model: final k_means model
    :param dataframe: goodreads dataframe
    :param features: features that want to be considered for recommender
    :param num_recommendations: number of outputted recs
    :return: reccomended_books: dataframe with information of reccommended books
    """
    # Assign clusters to user-liked books
    user_liked_clusters = clustering_model.predict(dataframe.loc[dataframe['title'].isin(user_likes), features])

    # Calculate distances to centroids for user-liked books
    distances_to_centroids = clustering_model.transform(
        dataframe.loc[dataframe['title'].isin(user_likes), features].values)

    # Calculate similarity scores for user-liked books
    similarity_scores_user_liked = 1 / (1 + distances_to_centroids + 1e-6)

    # Sum the similarity scores across clusters
    total_similarity_scores_user_liked = similarity_scores_user_liked.sum(axis=0)

    # Find the cluster with the highest total similarity for user-liked books
    recommended_cluster = total_similarity_scores_user_liked.argmax()

    # Get book recommendations from the recommended cluster
    recommended_books = dataframe[dataframe['cluster'] == recommended_cluster].copy()

    # Exclude books the user already likes from recommendations
    recommended_books = recommended_books[~recommended_books['title'].isin(user_likes)]

    # Calculate distances to centroids for recommended books
    distances_to_centroids_recommended = clustering_model.transform(recommended_books[features].values)

    # Calculate similarity scores for recommended books
    similarity_scores_recommended = 1 / (1 + distances_to_centroids_recommended + 1e-6)

    # Sum the similarity scores across recommended books
    total_similarity_scores_recommended = similarity_scores_recommended.sum(
        axis=1)  # Sum across rows instead of columns

    # Add a 'similarity_score' column to the DataFrame
    recommended_books['similarity_score'] = total_similarity_scores_recommended

    # Sort the recommended books by their similarity score
    recommended_books = recommended_books.sort_values(by='similarity_score', ascending=False)

    # Keep only the desired number of recommendations
    recommended_books = recommended_books.head(num_recommendations)

    # Reset the index of the DataFrame
    recommended_books.reset_index(drop=True, inplace=True)
    recommended_books['similarity_score'] = recommended_books['similarity_score'].apply(lambda x: round(x, 3))
    return recommended_books



DEFAULT_IMG = 'http:/images/icons/avatar_book-sm.png'


def get_image(title, author):
    """
    :param title: book title
    :param author: book author
    :return: url of book cover image
    """
    result_string = title + " " + author
    result_string = result_string.replace(" ", "+")
    url = f'https://openlibrary.org/search?q={result_string}&mode=everything'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all(class_='searchResultItem')
    for r in results:
        img_tag = r.find('img')
        if img_tag:
            # Construct full URL
            base = img_tag.get('src')
            full_url = "http:" + base
            if full_url == DEFAULT_IMG:
                return None
            else:
                return full_url

# Dash App

# define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])

colors = {
    'background': '#F3F6FA',
    'text': '#2E3E4E',
    'accent': '#4CAF50',
}

# Define the layout of the web app using Dash Bootstrap Components (dbc)
app.layout = dbc.Container([
    # Header section
    html.H1("Book Recommender", style={'fontWeight': 'bold', 'color': colors['text']}, className='text-center'),
    html.P("Choose your favorite book, choose how many recommendations you want, and see the book covers!",
           style={'color': colors['text']}),

    # Input section
    dbc.Row([
        dbc.Col([
            html.H5("Enter a book you like:", style={'color': colors['text']}),
            dcc.Input(id='input', placeholder='Enter a book title:', style={'width': '100%'})
        ], md=3),
        dbc.Col([
            html.H5("Number of recommendations:", style={'color': colors['text']}),
            dcc.Slider(id='num-recs-slider', min=1, max=10, step=1, value=5)
        ], md=3)
    ], className='my-2'),

    # Main content section
    html.Div(
        children=[
            # Button to trigger recommendations
            dbc.Button(
                "Recommend",
                id='submit',
                style={
                    'width': '100px',
                    'background-color': colors['accent'],
                    'transition': 'background-color 0.3s',
                },
                className='me-2',
            ),

            # Loading indicator while recommendations are being computed
            dcc.Loading(
                dbc.Row([
                    dbc.Col([
                        # Table to display recommended books
                        html.Div(id='table', style={'overflowY': 'scroll', 'height': '300px'})
                    ], md=6),
                    dbc.Col([
                        # Graph to visualize genre distribution of recommended books
                        dcc.Graph(id='graph', style={'height': '300px'})
                    ])
                ]),
                id="loading",
                type="default"
            ),

            # Header for book covers section
            html.H2("Book Covers", style={'fontWeight': 'bold', 'color': colors['text']}),

            # Container for displaying book covers
            html.Div(id='image-div', style={'display': 'flex', 'flexWrap': 'wrap'}, className='mt-2')
        ],
        className='ms-auto d-flex flex-column'
    ),

], fluid=True)

# Callback function to update recommendations based on user input
@app.callback(
    Output('table', 'children'),
    Output('graph', 'figure'),
    Output('image-div', 'children'),
    Input('submit', 'n_clicks'),
    State('input', 'value'),
    State('num-recs-slider', 'value'),
)
def update_recommendations(n_clicks, input_value, num_recs):
    if input_value:
        # Assume 'recommend_books' returns a DataFrame with recommendations
        recommended_books = recommend_books(
            [input_value],
            final_kmeans,
            goodreads,
            features_to_visualize,
            num_recs
        )

        # Create a DataTable to display recommended books
        datatable = dash_table.DataTable(
            data=recommended_books[recommended_columns].to_dict('records'),
            columns=[{'name': i, 'id': i} for i in recommended_columns]
        )

        # Clean the data for visualization
        recommended_books = recommended_books.apply(
            lambda x: x.str.replace("'", "").str.replace('"', '') if x.dtype == 'O' else x)

        # Create a DataFrame for genre visualization
        rec_split = recommended_books[['title', 'genres']].copy()
        rec_split['genres'] = rec_split['genres'].apply(lambda x: x.split(', '))
        rec_split['genres'] = rec_split['genres'].apply(lambda x: list(set([genre.strip("[]") for genre in x])))
        rec_split['genres'] = rec_split['genres'].apply(lambda x: [genre.strip("[]") for genre in x])
        rec_split['count'] = rec_split['genres'].apply(len)

        # Get unique genres for color mapping
        big_genre_list = list(set([genre for sublist in rec_split['genres'] for genre in sublist]))
        unique_genres = list(set(big_genre_list))

        # Create a color map for each genre
        color_set = px.colors.qualitative.Alphabet
        color_map = {genre: color_set[i] for i, genre in enumerate(unique_genres)}

        # Create traces for genre distribution graph
        data_traces = []
        for genre in unique_genres:
            count_per_book = rec_split['genres'].apply(lambda x: x.count(genre))
            trace = go.Bar(
                x=rec_split['title'],
                y=count_per_book,
                name=genre,
                marker_color=color_map[genre]
            )
            data_traces.append(trace)

        # Create layout for genre distribution graph
        layout = go.Layout(
            title='Book Titles and Genre Count',
            xaxis=dict(title='Book Titles'),
            yaxis=dict(title='Count'),
            barmode='stack'
        )

        # Create figure for genre distribution graph
        fig = go.Figure(data=data_traces, layout=layout)

        # Get image URLs for recommended books
        image_urls = []
        default_img = "https://publications.iarc.fr/uploads/media/default/0001/02/thumb_1290_default_publication.jpeg"
        for index, row in recommended_books.iterrows():
            title = row['title']
            author = row['author']
            image_url = get_image(title, author)
            image_urls.append(image_url)

        # Create image elements for recommended books
        if image_urls:
            image_elements = [
                html.Img(src=url or default_img)
                for url in image_urls
            ]
        else:
            image_elements = html.Div("No images to display.")

        return datatable, fig, image_elements

    # Return default values if no book title is entered
    fig = px.bar()
    return no_update, fig, no_update






x = goodreads.loc[:, features_to_visualize].values

# Compress using PCA
pca = PCA(n_components=2, whiten=False)
x_compress = pca.fit_transform(x)

# Add compressed features back into the DataFrame (for plotting)
goodreads['pca0'] = x_compress[:, 0]
goodreads['pca1'] = x_compress[:, 1]

# Scatter plot using Plotly Express
hover_data = ['main genre'] + features_to_visualize
fig = px.scatter(
    goodreads,
    x='pca0',
    y='pca1',
    hover_data=hover_data,
    color='main genre',
    title="fig 3: Principle Component Analysis Map"
)

# Show the plot
fig.show()



if __name__ == '__main__':
    app.run_server(debug=True)
