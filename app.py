import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="MovieLens Explorer", layout="wide")

@st.cache_data
def load_data():
    movies = pd.read_csv('movies.csv')
    ratings = pd.read_csv('ratings.csv')
    links = pd.read_csv('links.csv', dtype={'imdbId': str})
    return movies, ratings, links

def main():
    st.title("MovieLens Explorer")

    st.markdown(
        """
        This application, "MovieLens Explorer," is designed for educational and learning purposes only.
        It utilizes a small version of the MovieLens dataset.
        App built to explore Antigravity's agent development process.
        For more details on the dataset, visit the official MovieLens website:
        [MovieLens ml-latest-small README](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)
        """
    )

    # Load data
    with st.spinner('Loading data...'):
        movies, ratings, links = load_data()

    # Preprocessing
    # Merge ratings with movies to get genres for filtering
    # We don't merge everything into one big df immediately to save memory/time, 
    # but for the bar chart we need ratings + genres.
    
    # Extract all unique genres
    all_genres = set()
    for genres in movies['genres'].str.split('|'):
        all_genres.update(genres)
    all_genres = sorted(list(all_genres))

    # Sidebar
    st.sidebar.header("Filters")
    
    # Genre filter (Requirement 1 & 2)
    selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + all_genres)

    # Filter movies by genre first if selected
    if selected_genre != "All":
        filtered_movies = movies[movies['genres'].str.contains(selected_genre, regex=False)]
    else:
        filtered_movies = movies

    # --- Requirement 1: Bar chart of count of ratings by rating ---
    st.header("Rating Distribution")
    
    # Filter ratings to only include those for the filtered movies
    filtered_ratings = ratings[ratings['movieId'].isin(filtered_movies['movieId'])]
    
    rating_counts = filtered_ratings['rating'].value_counts().sort_index()
    st.bar_chart(rating_counts)

    # --- Requirement 2: Top 10 movies by avg rating ---
    st.header("Top 10 Movies by Average Rating")

    # Minimum number of ratings filter
    min_ratings = st.sidebar.slider("Minimum Number of Ratings", 0, 500, 100)

    # Calculate avg rating and count
    # Group by movieId
    movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['avg_rating', 'rating_count']
    movie_stats = movie_stats.reset_index()

    # Filter by min ratings
    movie_stats = movie_stats[movie_stats['rating_count'] >= min_ratings]

    # Join with filtered movies (which handles the genre filter)
    # Inner join will keep only movies that match both genre and min rating criteria
    top_movies = pd.merge(filtered_movies, movie_stats, on='movieId', how='inner')

    # Sort by avg rating
    top_movies = top_movies.sort_values(by='avg_rating', ascending=False).head(10)

    # --- Requirement 3: IMDb links ---
    # Join with links
    top_movies = pd.merge(top_movies, links, on='movieId', how='left')

    # Create link column
    # https://www.imdb.com/title/tt0114709/
    # imdbId in links.csv does not have 'tt' prefix and might need padding?
    # Checking links.csv content from previous turn: "1,0114709,862" -> looks like it has leading zeros but no 'tt'.
    # The user request says: "https://www.imdb.com/title/tt0114709/"
    # So we need to add 'tt'.
    
    def make_imdb_link(imdb_id):
        if pd.isna(imdb_id):
            return ""
        return f"https://www.imdb.com/title/tt{imdb_id}/"

    top_movies['IMDb Link'] = top_movies['imdbId'].apply(make_imdb_link)

    # Display table
    # We'll use st.dataframe with column configuration for links if using newer streamlit, 
    # or just markdown links. 
    # St.dataframe with LinkColumn is available in newer versions. 
    # Let's try to make it a clickable link in a table or just show the URL.
    # The requirement says "create a link", usually implies clickable.
    # Using st.data_editor or st.column_config.LinkColumn is the modern way.
    
    display_df = top_movies[['title', 'genres', 'avg_rating', 'rating_count', 'IMDb Link']]
    
    st.dataframe(
        display_df,
        column_config={
            "IMDb Link": st.column_config.LinkColumn("IMDb Link")
        },
        hide_index=True
    )

    # --- Requirement: LLM Powered Recommender ---
    st.header("LLM Movie Recommender")
    st.write("Select a movie to get recommendations based on its IMDb link context.")

    # Dropdown to select a movie
    # We can use the filtered movies list or the full list. 
    # Let's use the filtered list so the user can narrow down choices first.
    movie_titles = filtered_movies['title'].tolist()
    selected_movie_title = st.selectbox("Select a movie for recommendations", [""] + movie_titles)

    if st.button("Get Recommendations") and selected_movie_title:
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        else:
            try:
                client = openai.OpenAI(api_key=api_key)

                with st.spinner("Generating recommendations..."):

                    # 2. Ask LLM for recommendations based on search context
                    prompt = f"""
                    Generate 3 movie recommendations similar to "{selected_movie_title}".
                    For each recommended film, include:
                    - Title
                    - Release year
                    - 1–2 sentence reason for the recommendation
                    - IMDb link

                    Return results as a bullet list.
                    """

                    response = client.responses.create(
                        model="gpt-5",
                        input=[{"role": "user", "content": prompt}],
                        tools=[{"type": "web_search"}],
                        tool_choice="auto",
                        reasoning={"effort": "low"},
                        text={"format": {"type": "text"}}
                    )
                                    
                    st.success("Recommendations:")
                    recommendations_text = response.output_text
                    # Regex to find IMDb links
                    # Matches "https://www.imdb.com/title/tt" followed by 7 or more digits, ending with a '/'
                    imdb_link_pattern = r"https://www\.imdb\.com/title/tt\d{7,}/"

                    def replace_with_markdown_link(match):
                        url = match.group(0)
                        return f"[IMDb Link]({url})"

                    formatted_recommendations = re.sub(imdb_link_pattern, replace_with_markdown_link, recommendations_text)
                    st.markdown(formatted_recommendations)

                # --- Requirement: YouTube Trailer ---
                # Moved to separate section below

            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.divider()

    # --- Requirement: Separate Trailer Feature ---
    st.header("LLM Movie Trailer Finder")
    st.write("Select a movie to find its YouTube trailer.")

    trailer_movie_title = st.selectbox("Select a movie for trailer", [""] + movie_titles, key="trailer_movie_select")

    if st.button("Get Trailer") and trailer_movie_title:
        with st.spinner("Finding trailer..."):
            trailer_prompt = f"Find a valid and available YouTube trailer link for '{trailer_movie_title}'. Only return the link, no other text."
            try:
                # Check for API key again just in case
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    st.error("OpenAI API key not found.")
                else:
                    client = openai.OpenAI(api_key=api_key)
                    response_2 = client.responses.create(
                        model="gpt-5",
                        input=[{"role": "user", "content": trailer_prompt}],
                        tools=[{"type": "web_search"}],
                        tool_choice="auto",
                        reasoning={"effort": "low"},
                        text={"format": {"type": "text"}}
                    )
                    trailer_link = response_2.output_text
                    st.session_state['trailer_link'] = trailer_link
            except Exception as e:
                st.error(f"Could not find trailer: {e}")

    # Display trailer from session state
    if st.session_state['trailer_link']:
        st.header(f"Trailer: {trailer_movie_title}")
        st.video(st.session_state['trailer_link'])


if __name__ == "__main__":
    main()
