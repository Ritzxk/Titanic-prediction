import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessing pipeline and model
with open('pipe.pkl', 'rb') as pipe_file:
    preprocessing_pipe = pickle.load(pipe_file)

with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define function to preprocess and predict
def preprocess_and_predict(pclass, sex, age, fare, embarked, family):
    # Create a dictionary with user inputs
    input_data = {
        'Pclass': [pclass],  # Wrap in a list to make it 2D
        'Sex': [sex],  # Wrap in a list to make it 2D
        'Age': [age],  # Wrap in a list to make it 2D
        'Fare': [fare],  # Wrap in a list to make it 2D
        'Embarked': [embarked],  # Wrap in a list to make it 2D
        'family_size': [family]
    }



    # Convert to DataFrame and preprocess the data
    input_df = pd.DataFrame(input_data)
    preprocessed_data = preprocessing_pipe.transform(input_df)

    # Predict using the model
    prediction = model.predict(preprocessed_data)

    return prediction[0]

# # Add custom CSS to set the background image
# st.markdown(
#     """
#     <style>
#     body {
#         background-image: url('https://images.nationalgeographic.org/image/upload/t_edhub_resource_key_image/v1638882458/EducationHub/photos/titanic-sinking.jpg');
#         background-size: cover;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

css = """
    <style>
        body {
            background-image: url('https://i0.wp.com/insights-on.com/wp-content/uploads/2021/03/10-sn56-20201221-titanicsinking-hr.jpg?fit=1024%2C572&ssl=1');
            background-size: cover;
            background-repeat: no-repeat;
        }
        .stApp {
            background-color: rgba(0,0,0,0.5);
            padding: 20px;
            border-radius: 10px;
        }

    </style>
"""

# Push the CSS to the page
st.markdown(css, unsafe_allow_html=True)

# Streamlit app
st.title('Titanic Survival Prediction App')

st.write('Please provide the following details to predict survival:')

# Input fields for user
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.radio('Sex', ['male', 'female'])
age = st.slider('Age', min_value=0, max_value=100, step=1)
fare = st.slider('Fare', min_value=0.0, max_value=1000.0, step=0.01, format="%.2f")
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
family = st.selectbox('family_size', ['Alone', 'Medium', 'Large'])

# Display input values in a list
st.subheader('Input Values:')
st.write('Pclass:', pclass)
st.write('Sex:', sex)
st.write('Age:', age)
st.write('Fare:', fare)
st.write('Embarked:', embarked)
st.write('Family Size:', family)

# Predict survival
if st.button('Predict Survival'):
    # Preprocess and predict
    prediction = preprocess_and_predict(pclass, sex, age, fare, embarked, family)
    
    # Display result
    if prediction == 0:
        st.write('Based on the provided information, the prediction is: Not Survived')
    else:
        st.write('Based on the provided information, the prediction is: Survived')
        
# data visualization
df = pd.read_csv('cleaned_train.csv')

# plotting 
st.subheader('Count of People Who Survived vs. Not Survived')
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df, x='Survived', ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not Survived', 'Survived'])
ax.set_xlabel('Fate')
ax.set_ylabel('Number of People')
plt.tight_layout()
st.pyplot(fig)