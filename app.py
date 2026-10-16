#streamlit app
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from elbow import get_optimum_k

#page config
st.set_page_config(page_title="Customer Segmentation",
                   page_icon="logo.png")

#sidebar
with st.sidebar:
    st.title("Customer Segmentation")
    st.image("logo.png")

file  = st.file_uploader("Upload dataset: ",
                         type=["csv","xlsx"])

if file:
    ext = file.name.split(".")[-1]
    if ext=="csv":
        df = pd.read_csv(file)
    elif ext=="xlsx":
        df = pd.read_excel(file)
    else:
        st.error("un supported extension")
    columns = st.multiselect("Select features: ",
                   options=df.columns,
                   default=["Annual Income (k$)","Spending Score (1-100)"])
    df = df[columns]

    #model training
    K = get_optimum_k(df)
    KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
    model.fit(df)
    df["labels"] = model.labels_
    df

    #scatter chart
    st.scatter_chart(x="Annual Income (k$)",
                     y="Spending Score (1-100)",
                     data=df,
                     color="labels")
