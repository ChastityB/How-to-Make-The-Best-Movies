#Deep NLP Model to NLP
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
import tensorflow as tf

with open('config/filepaths.json') as f:
    PATHS = json.load(f)

#title
st.title("Predicting Movie Review Ratings - Deep NLP Model")
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
def load_tf_dataset(path):
    ds = tf.data.Dataset.load(path)
    return ds

#load the training and testing data
#load training and test data
train_ds = load_tf_dataset(PATHS['data']['tf']['train_tf'])
test_ds = load_tf_dataset(PATHS['data']['tf']['test_tf'])

#load TF model
def load_network(path):
    model = tf.keras.models.load_model(path)
    return model
path_model = PATHS['models']['gru']

#load model
#load best Deep NLP model
gru_model = load_network(path_model)

#Obtain text to predict from input box
X_to_pred = st.text_input("### Enter review text to predict here:", value="The movie was great! Highly Recommend.")

#convert y to sklearn classes
def convert_y_to_sklearn_classes(y, verbose=False):
    # If already one-dimension
    if np.ndim(y)==1:
        if verbose:
            print("- y is 1D, using it as-is.")
        return y
        
    # If 2 dimensions with more than 1 column:
    elif y.shape[1]>1:
        if verbose:
            print("- y is 2D with >1 column. Using argmax for metrics.")   
        return np.argmax(y, axis=1)
    
    else:
        if verbose:
            print("y is 2D with 1 column. Using round for metrics.")
        return np.round(y).flatten().astype(int)

#get true pred labels
def get_true_pred_labels(model,ds):
    """Gets the labels and predicted probabilities from a Tensorflow model and Dataset object.
    Adapted from source: https://stackoverflow.com/questions/66386561/keras-classification-report-accuracy-is-different-between-model-predict-accurac
    """
    y_true = []
    y_pred_probs = []
    
    # Loop through the dataset as a numpy iterator
    for images, labels in ds.as_numpy_iterator():
        
        # Get prediction with batch_size=1
        y_probs = model.predict(images, batch_size=1, verbose=0)
        # Combine previous labels/preds with new labels/preds
        y_true.extend(labels)
        y_pred_probs.extend(y_probs)
    ## Convert the lists to arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    return y_true, y_pred_probs
        
#basic function to obtain prediction
lookup_dict_neural = {1: 'Low Review', 0:'High Review'}
def predict_decode_deep(X_to_pred, network, lookup_dict, return_index=True):
    if isinstance(X_to_pred, str):
        X = [X_to_pred]
    else: 
        X = [X_to_pred]
    
    pred_probs = network.predict(X)
    
    pred_class = convert_y_to_sklearn_classes(pred_probs)
    
    class_name = lookup_dict[pred_class[0]]
    
    return class_name

    
if st.button("Get prediction"):
        pred_class_name = predict_decode_deep(X_to_pred, gru_model, lookup_dict_neural)
        st.markdown(f"##### Neural Network Predicted category:  {pred_class_name}")
    
else: 
    st.empty()

st. divider()
st.subheader('Evaluate Neural Network')
col1,col2 = st.columns(2)
show_train = col1.checkbox("Show training data.", value=True)
show_test = col2.checkbox("Show test data.", value=True)

def classification_metrics_streamlit(y_true, y_pred, label='',
                           output_dict=False, figsize=(8,4),
                           normalize='true', cmap='Blues',
                           colorbar=False,values_format=".2f"):
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



def classification_metrics_streamlit_tensorflow(model,X_train=None, y_train=None, 
                                                label='Training Data',
                                    figsize=(6,4), normalize='true',
                                    output_dict = False,
                                    cmap_train='Blues',
                                    cmap_test="Reds",
                                    values_format=".2f", 
                                    colorbar=False):
    
    ## Check if X_train is a dataset
    if hasattr(X_train,'map'):
        # If it IS a Datset:
        # extract y_train and y_train_pred with helper function
        y_train, y_train_pred = get_true_pred_labels(model, X_train)
    else:
        # Get predictions for training data
        y_train_pred = model.predict(X_train)


     ## Pass both y-vars through helper compatibility function
    y_train = convert_y_to_sklearn_classes(y_train)
    y_train_pred = convert_y_to_sklearn_classes(y_train_pred)
    
    # Call the helper function to obtain regression metrics for training data
    report, conf_mat = classification_metrics_streamlit(y_train, y_train_pred, 
                                     output_dict=True, figsize=figsize,
                                         colorbar=colorbar, cmap=cmap_train, 
                                                        values_format=values_format,label=label)
    return report, conf_mat

#neural network button
if st.button("Show Neural Network evaluation"):
    with st.spinner("Please wait while the neural network is evaluated..."):
        if show_train == True:
            # Display training data results
            report_str, conf_mat = classification_metrics_streamlit_tensorflow(gru_model,label='Training Data',
                                                                               X_train=train_ds,
                                                                               )
            st.text(report_str)
            st.pyplot(conf_mat)
            st.text("\n\n")
    
        if show_test == True: 
            # Display training data results
            report_str, conf_mat = classification_metrics_streamlit_tensorflow(gru_model,label='Test Data',
                                                                               X_train=test_ds
                                                                               )
            st.text(report_str)
            st.pyplot(conf_mat)
            st.text("\n\n")
  
else:
    st.empty()

