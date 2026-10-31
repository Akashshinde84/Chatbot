import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

vectorizer = TfidfVectorizer()
clf=LogisticRegression(random_state=0, max_iter=10000)

tags=[]
patterns=[]
intents = globals().get("intents", [])

for intent in intents:
    for pattern in intent.get("patterns", []):
        tags.append(intent["tag"])
        patterns.append(pattern)

x=vectorizer.fit_transform(patterns)
y=tags
clf.fit(x,y) 


def chatbot(input_text):
    input_text=vectorizer.transform([input_text])
    tag=clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag']==tag:
            response=random.choice(intent['response'])
            return response 


counter=0
def main():
    global counter
    st.title("CHATBOT")
    st.write("Welcome to the chatbot!")

    counter+=1
    user_input=st.text_input("You:",key=f"user_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("chatbot:",value=response,height=100,max_chars=None,key=f"chatbot_response_{counter}")

        if response.lower() in['goodbye','bye']:
            st.write("Thank you")
            st.stop()
if __name__=='__main__':
    main()