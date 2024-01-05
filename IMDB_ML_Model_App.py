#Machine Learning Model to Streamlit
import streamlit as st
import streamlit.components.v1 as components
import base64
from streamlit.components.v1 import html
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import numpy as np
import json
with open('config/filepaths.json') as f:
    PATHS = json.load(f)

#title
st.title("Predicting Movie Review Ratings - Machine Learning Model")
#render SVG to display on streamlit
def render_svg(svg_string):
    """Renders the given svg string."""
    c = st.container()
    with c:
        html(svg_string)

render_svg(open(PATHS['images']['TheMovieDBLogo']).read())
render_svg(open(PATHS['images']['TMDBLogo']).read())

#define a function to load ml training and test data
@st.cache_resource
def load_Xy_data(PATHS, data_type='train'):
    path = PATHS['data']['ml'][data_type]
    return joblib.load(path)

#load the training and testing data
X_train, y_train = load_Xy_data(PATHS, data_type='train')
X_test, y_test = load_Xy_data(PATHS, data_type='test')

#load ML model
def load_model_ml(PATHS,type='logreg'):
    path = PATHS['models'][type]
    return joblib.load(path)

#load model
logreg = load_model_ml(PATHS, type='logreg')

#Obtain text to predict from input box
X_to_pred = st.text_input("### Enter review text to predict here:", 
                         value="The movie was great! Highly Recommend.")

#basic function to obtain prediction
lookup_dict = {"Low":"Low Review", "High":"High Review"}
def make_prediction(X_to_pred, lookup_dict=lookup_dict):
    pred_class = logreg.predict([X_to_pred])[0]
    pred_class = lookup_dict[pred_class]
    return pred_class

@st.cache_resource
def get_explainer(class_names = None):
    lime_explainer = LimeTextExplainer(class_names=class_names)
    return lime_explainer
def explain_instance(explainer, X_to_pred,predict_func):
    explanation = explainer.explain_instance(X_to_pred, predict_func)
    return explanation.as_html(predict_proba=False)
# Create the lime explainer
explainer = get_explainer(class_names = sorted(list(lookup_dict.values())))

#classification matrics streamlit function
def classification_metrics_streamlit(y_true, y_pred, label='',
                           figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False,values_format=".2f"):
    """Modified version of classification metrics function from Intro to Machine Learning.
    Updates:
    - Reversed raw counts confusion matrix cmap  (so darker==more).
    - Added arg for normalized confusion matrix values_format
    """
    # Get the classification report
    report = classification_report(y_true, y_pred)
    
    ## Save header and report
    header = "-"*70
    final_report = "\n".join([header,f" Classification Metrics: {label}", header,report,"\n"])
        
    ## CONFUSION MATRICES SUBPLOTS
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    
    # Create a confusion matrix  of raw counts (left subplot)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=None, 
                                            cmap='gist_gray_r',# Updated cmap
                                            values_format="d", 
                                            colorbar=colorbar,
                                            ax = axes[0]);
    axes[0].set_title("Raw Counts")
    
    # Create a confusion matrix with the data with normalize argument 
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            normalize=normalize,
                                            cmap=cmap, 
                                            values_format=values_format, #New arg
                                            colorbar=colorbar,
                                            ax = axes[1]);
    axes[1].set_title("Normalized Confusion Matrix")
    
    # Adjust layout and show figure
    fig.tight_layout()

    return final_report, fig

# Trigger prediction and explanation with a button
if st.button("Get prediction"):
    pred_class_name = make_prediction(X_to_pred)
    st.markdown(f"##### Predicted category:  {pred_class_name}")
    # Get the Explanation as html and display using the .html component.
    html_explanation = explain_instance(explainer, X_to_pred, logreg.predict_proba)
    components.html(html_explanation, height=400)
else: 
    st.empty()

st.divider()
st.subheader('Evaluate Machine Learning Model')

## To place the 3 checkboxes side-by-side
col1,col2,col3 = st.columns(3)
show_train = col1.checkbox("Show training data.", value=True)
show_test = col2.checkbox("Show test data.", value=True)
show_model_params =col3.checkbox("Show model params.", value=False)



#show model evaluation button
if st.button("Show model evaluation."):
    
    if show_train == True:
        # Display training data results
        y_pred_train = logreg.predict(X_train)
        report_str, conf_mat = classification_metrics_streamlit(y_train, y_pred_train, label='Training Data')
        st.text(report_str)
        st.pyplot(conf_mat)
        st.text("\n\n")

    if show_test == True: 
        # Display the trainin data resultsg
        y_pred_test = logreg.predict(X_test)
        report_str, conf_mat = classification_metrics_streamlit(y_test, y_pred_test, cmap='Reds',label='Test Data')
        st.text(report_str)
        st.pyplot(conf_mat)
        st.text("\n\n")

    if show_model_params:
        # Display model params
        st.markdown("####  Model Parameters:")
        st.write(logreg.get_params())

else:
    st.empty()