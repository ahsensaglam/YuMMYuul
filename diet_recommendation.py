import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google_images_search import GoogleImagesSearch
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import json

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

###############  ORIGINAL DATAFRAME ###############
# url = 'https://raw.githubusercontent.com/zakaria-narjis/Diet-Recommendation-System/main/Data/dataset.csv'
# df = pd.read_csv(url, compression='gzip')
# df= pd.read_pickle(r"C:\Users\minek\PycharmProjects\miuul_python\Bitirme_Projesi\dataset.pkl")

###############  LOAD DATAFRAMES ###############
df15= pd.read_pickle(r"C:\Users\minek\PycharmProjects\miuul_python\Bitirme_Projesi\df15.pkl")
df30= pd.read_pickle(r"C:\Users\minek\PycharmProjects\miuul_python\Bitirme_Projesi\df30.pkl")
df45= pd.read_pickle(r"C:\Users\minek\PycharmProjects\miuul_python\Bitirme_Projesi\df45.pkl")
df60= pd.read_pickle(r"C:\Users\minek\PycharmProjects\miuul_python\Bitirme_Projesi\df60.pkl")
df120= pd.read_pickle(r"C:\Users\minek\PycharmProjects\miuul_python\Bitirme_Projesi\df120.pkl")

ingredients = np.load(r'C:\Users\minek\PycharmProjects\miuul_python\Bitirme_Projesi\ingredients.npy')

## DF cleaning

# Assuming your data is stored in a DataFrame called df
# Make sure to import the necessary libraries and load your data before running this code

# Define a function to extract ingredients within c() parentheses
# def extract_ingredients(text):
#     match = re.search(r'c\((.*?)\)', text)
#     if match:
#         return match.group(1)
#     else:
#         return ""
#
#
# # Apply the function to the RecipeIngredientParts column
# df15['RecipeIngredientParts'] = df15['RecipeIngredientParts'].apply(extract_ingredients)
# df15['RecipeInstructions'] = df15['RecipeInstructions'].apply(extract_ingredients)
#
# df30['RecipeIngredientParts'] = df30['RecipeIngredientParts'].apply(extract_ingredients)
# df30['RecipeInstructions'] = df30['RecipeInstructions'].apply(extract_ingredients)
#
# df45['RecipeIngredientParts'] = df45['RecipeIngredientParts'].apply(extract_ingredients)
# df45['RecipeInstructions'] = df45['RecipeInstructions'].apply(extract_ingredients)
#
# df60['RecipeIngredientParts'] = df60['RecipeIngredientParts'].apply(extract_ingredients)
# df60['RecipeInstructions'] = df60['RecipeInstructions'].apply(extract_ingredients)
#
# df120['RecipeIngredientParts'] = df120['RecipeIngredientParts'].apply(extract_ingredients)
# df120['RecipeInstructions'] = df120['RecipeInstructions'].apply(extract_ingredients)


# df15.to_pickle('df15.pkl')
# df30.to_pickle('df30.pkl')
# df45.to_pickle('df45.pkl')
# df60.to_pickle('df60.pkl')
# df120.to_pickle('df120.pkl')


st.markdown("""<style>.custom-title {color: #e74c3c;  /* Kırmızı renk kodu */}</style>""", unsafe_allow_html=True)
st.markdown("<h1 class='custom-title'>WHAT DO YOU WANT TO EAT?</h1>", unsafe_allow_html=True)

st.subheader("How Much Time Do You Have?")


def get_dataframe(selected_option, selected_type):
    if selected_option == "0-15 minutes" and selected_type == "low calorie":
        return df15[df15["k_means_clusters"] == 2].reset_index(drop=True)
    elif selected_option == "0-15 minutes" and selected_type == "medium calories":
        return df15[df15["k_means_clusters"] == 1].reset_index(drop=True)
    elif selected_option == "0-15 minutes" and selected_type == "high calorie":
        return df15[df15["k_means_clusters"] == 3].reset_index(drop=True)
    elif selected_option == "15-30 minutes" and selected_type == "low calorie":
        return df30[df30["k_means_clusters"] == 1].reset_index(drop=True)
    elif selected_option == "15-30 minutes" and selected_type == "medium calories":
        return df30[df30["k_means_clusters"] == 3].reset_index(drop=True)
    elif selected_option == "15-30 minutes" and selected_type == "high calorie":
        return df30[df30["k_means_clusters"] == 2].reset_index(drop=True)
    if selected_option == "30-45 minutes" and selected_type == "low calorie":
        return df45[df45["k_means_clusters"] == 1].reset_index(drop=True)
    elif selected_option == "30-45 minutes" and selected_type == "medium calories":
        return df45[df45["k_means_clusters"] == 3].reset_index(drop=True)
    elif selected_option == "30-45 minutes" and selected_type == "high calorie":
        return df45[df45["k_means_clusters"] == 2].reset_index(drop=True)
    elif selected_option == "45-60 minutes" and selected_type == "low calorie":
        return df60[df60["k_means_clusters"] == 2].reset_index(drop=True)
    elif selected_option == "45-60 minutes" and selected_type == "medium calories":
        return df60[df60["k_means_clusters"] == 1].reset_index(drop=True)
    elif selected_option == "45-60 minutes" and selected_type == "high calorie":
        return df60[df60["k_means_clusters"] == 3].reset_index(drop=True)
    elif selected_option == "60-120 minutes" and selected_type == "low calorie":
        return df120[df120["k_means_clusters"] == 1].reset_index(drop=True)
    elif selected_option == "60-120 minutes" and selected_type == "medium calories":
        return df120[df120["k_means_clusters"] == 2].reset_index(drop=True)
    elif selected_option == "60-120 minutes" and selected_type == "high calorie":
        return df120[df120["k_means_clusters"] == 3].reset_index(drop=True)


selected_time = st.selectbox(label="Select Duration",
                              options=["0-15 minutes", "15-30 minutes", "30-45 minutes", "45-60 minutes",
                                       "60-120 minutes"],index=0)

st.subheader("Do you want to lose weight? What kind of meal would you like?")

selected_type = st.selectbox(label="Select Calories Type",
                             options=["low calorie", "medium calories", "high calorie"],
                             index=0)


selected_ingredients = st.multiselect(label="Select a ingredients from the dropdown", options=ingredients,
                                          default=["sugar"])



selected_df = get_dataframe(selected_time, selected_type)
df = selected_df.reset_index(drop=True)


################################# STREAMLIT ########################################################################


def parse_ingredients(ingredients_str):
    try:
        return [ingredient.strip()[1:-1] for ingredient in ingredients_str.strip('[]').split(',')]
    except:
        return []


def content_based_recommender(ingredient_list, dataframe):
    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # Process each food name in the list individually
    processed_ingredients = [' '.join(parse_ingredients(ingredients)) for ingredients in dataframe['RecipeIngredientParts']]
    # Fit and transform the processed ingredient parts to create a TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_ingredients)
    # Process the input ingredient_list similarly
    processed_input = ' '.join(ingredient_list)
    # Calculate the cosine similarity between recipes based on ingredient parts
    cosine_sim = linear_kernel(tfidf_vectorizer.transform([processed_input]), tfidf_matrix)
    # Get the similarity scores for all recipes compared to the given ingredient list
    similarity_scores = pd.Series(cosine_sim.flatten())
    # Get the indices of the top 10 similar recipes
    recommended_indices = similarity_scores.sort_values(ascending=False)[:10].index
    # Return the names of the recommended recipes
    return dataframe['Name'].iloc[recommended_indices]


def search_google_images(query, api_key, cse_id):
    gis = GoogleImagesSearch(api_key, cse_id)

    # Define search parameters
    _search_params = {
        'q': query,
        'num': 1,  # Number of results to fetch
        'safe': 'off',  # You can set this to 'medium' or 'high' for safe search
    }

    # Perform the search
    gis.search(search_params=_search_params)

    # Get the first result
    result = gis.results()[0]

    return result.url


def search_youtube_videos(query, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Define search parameters
    request = youtube.search().list(
        q=query,
        type='video',
        part='id',
        maxResults=1  # Number of results to fetch
    )

    # Execute the search
    response = request.execute()

    # Get the video ID from the response
    video_id = response['items'][0]['id']['videoId']

    # Construct the video URL
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    return video_url

# API keys for image and video search
google_api_key = "AIzaSyDERCwNaXxtCegeElR31r5XyuXOHg0473o"
google_cse_id = "45b5cec8ecf9344ad"
youtube_api_key = "AIzaSyDERCwNaXxtCegeElR31r5XyuXOHg0473o"



# Get or create the session state
if 'selected_food2' not in st.session_state:
    st.session_state.selected_food2 = None

if 'recommended_food_names' not in st.session_state:
    st.session_state.recommended_food_names = None


show_recommendation = st.button('Show Recommendation')

if show_recommendation:
    st.session_state.recommended_food_names = content_based_recommender(selected_ingredients, df)
    st.write(f"Recommended foods for {selected_ingredients}:", st.session_state.recommended_food_names)




if st.session_state.recommended_food_names is not None:
    st.session_state.selected_food2 = st.selectbox("Select a recommended food", st.session_state.recommended_food_names, index=0)
else:
    st.session_state.selected_food2 = None

# Perform the search and display the image and video
if st.session_state.selected_food2:
    try:
        st.subheader("Ingredients:")
        st.write(df[df["Name"] == st.session_state.selected_food2]["RecipeIngredientParts"].values[0])

        st.subheader("Recipe:")
        st.write(df[df["Name"] == st.session_state.selected_food2]["RecipeInstructions"].values[0])
    except Exception as e:
        # If an error occurs, simply do nothing
        pass

    # try:
    #     image_url = search_google_images(st.session_state.selected_food2, google_api_key, google_cse_id)
    #     video_url = search_youtube_videos(st.session_state.selected_food2, youtube_api_key)
    #
    #     # Display the image
    #     st.image(image_url, caption=f'{st.session_state.selected_food2} - Google Image Search Result', use_column_width=True)
    #
    #     # Display the video as playable in Streamlit
    #     st.video(video_url)
    # except Exception as e:
    #     # If an error occurs, simply do nothing
    #     pass




