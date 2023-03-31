import streamlit as st
import numpy as np
import pandas
from streamlit_option_menu import option_menu
from prediction import predict, genre_columns
import time

tags_dict = {'Bernoulli Naive Bayes': 'suggested_tags_BernoulliNB', 
             'Logistic Regression': 'suggested_tags_LogisticRegression', 
             'LightGBM': 'suggested_tags_LGBMClassifier',
             'Chat-GPT': 'suggested_tags_ChatGPT'
             }

if "method" not in st.session_state:
    st.session_state.method = 'Bernoulli Naive Bayes'  

with st.sidebar:
    selected = option_menu("Tagging System", ["Home", 'Settings', 'Prediction'], 
        icons=['house-fill', 'gear-fill', 'chat-square-quote-fill'], menu_icon="bookmarks-fill", default_index=1)

if selected == "Home":
    image = '.\LOGO.png'
    
    col1, col2= st.columns([5,1])
    
    with col1:
        st.write("")
        st.write("")

        st.title('Automated Tagging System')
    
    with col2:
        st.image(image, width=200)
    
    st.header('_proof of concept_: :blue[IMDB Movie Genre Tagging]')
    st.write('')
    st.write('This app predictgenre for IMDB movies basewd on the plot synopsis! :movie_camera:')

if selected == 'Prediction':

    def clear_text():
        st.session_state["text_area_key"] = ""

    txt = st.text_area(label='Movie plot synopsis', placeholder="Insert plot synopsis here.", key='text_area_key')
    st.button(label='Clear Plot Synopsis', type="secondary", on_click=clear_text)

    col1, col2= st.columns([3,2])

    with col1:
        if st.button(label='Generate Genre', type="primary"):
            if not txt:
                st.error("Please enter a topic")
            else:
                with st.spinner("Please wait while the plot synopsis is processed..."):
                    time.sleep(2)
                    predictions_BernoulliNB, predictions_LogisticRegression, predictions_LGBMClassifier, predictions_ChatGPT = predict(x_test=txt)
                    # st.markdown(f"Here is a list of suggested genre for this movie based on **:red[{st.session_state.method}]** method:")
                    suggested_tags_BernoulliNB = genre_columns[np.array(predictions_BernoulliNB[0,:].tolist(), dtype=bool)]
                    suggested_tags_LogisticRegression = genre_columns[np.array(predictions_LogisticRegression[0,:].tolist(), dtype=bool)]
                    suggested_tags_LGBMClassifier = genre_columns[np.array(predictions_LGBMClassifier[0,:].tolist(), dtype=bool)]
                    suggested_tags_ChatGPT = predictions_ChatGPT
                    suggested_tags = eval(tags_dict[st.session_state.method])
                    
                    if not isinstance(suggested_tags, pandas.core.indexes.base.Index):
                        if suggested_tags == []:
                            st.error('Not able to predict any tags!')
                        else:
                            st.success("Here is a list of predicted tags:")
                            for item in suggested_tags:
                                st.write(':bookmark:', f"{item}".title())                     
                    elif suggested_tags.shape[0] == 0:
                        st.error('Not able to predict any tag!')
                    else:    
                        st.success("Here is a list of predicted tags:")
                        for item in suggested_tags:
                            st.write(':bookmark:', f"{item}".title())

        with col2:
            method = st.radio(
                    "Choose a prediction methods:",
                    ('Bernoulli Naive Bayes', 'Logistic Regression', 'LightGBM', 'Chat-GPT'),
                    key='method_radio')
            st.session_state.method = method
        