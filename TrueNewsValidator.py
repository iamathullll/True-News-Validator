import streamlit as st
import pickle
import spacy

with open('/Users/athulkrishnavv/PycharmProjects/project_NLP/savedobjects.pkl', "rb") as file:
    loaded_dict = pickle.load(file)

model = loaded_dict['Model']
nlp = loaded_dict['NLP']
vectorizer = loaded_dict['vectorizer']

def recommender(text):
    text = nlp(text)
    text = " ".join([i.lemma_ for i in text if not i.is_stop and not i.is_punct])
    res = str(text)
    vector = vectorizer.transform([res])
    res = model.predict(vector)
    if res == 0:
        return "Fake News"
    else:
        return "Real News"

st.title("TrueNews Validator")#Streamlit App
st.write("Enter a news text to determine whether it is Real or Fake.")
user_input = st.text_area("Enter news text:", height=150)# User input

# Check if user input is provided
if st.button("Check"):
    if user_input:
        result = recommender(user_input)
        st.write(f"The news is classified as: **{result}**")
    else:
        st.write("Please enter some text to check.")