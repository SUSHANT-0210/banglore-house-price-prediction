import streamlit as st
import pickle
import numpy as np
import sklearn



#load the model
with open('model.pkl', 'rb') as f:
    lr_clf = pickle.load(f)


columns = pickle.load(open("col.pkl", "rb"))
locations = columns[3:]


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(columns == location)[0]

    x = np.zeros(len(columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1  # If it is, it sets the element at loc_index in the array x to 1,
        # indicating the presence of that location in the one-hot encoded feature vector.

    return lr_clf.predict([x])[0]


st.title('Banglore House Prediction')

area = st.text_input("Area (in sqft)", placeholder="Enter the area")
bathroom = st.text_input("Bathroom", placeholder="Enter the Bathroom")
bhk = st.text_input("BHK", placeholder="Enter the BHK")


option_loc = st.selectbox(
    "Location",
    locations)


if st.button("Predict"):
    if area and bhk and bathroom:
        try:
            area = float(area)
            bath = int(bathroom)
            bhk = int(bhk)
            result = predict_price(option_loc, area, bathroom, bhk)
            st.success(f'The estimated price is: ₹ {result:,.2f} ''lakhs')
        except ValueError:
            st.error("Please enter valid inputs")

    else:
        st.error("Please fill in all the fields")



