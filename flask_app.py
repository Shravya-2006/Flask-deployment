#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
import base64

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    cluster_plot = None
    clustered_data = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)

            # Data Preprocessing
            df = df.fillna(df.median(numeric_only=True))
            categorical_columns = df.select_dtypes(include=["object"]).columns
            if len(categorical_columns) > 0:
                df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

            # Clustering
            selected_features = request.form.getlist("features")
            num_clusters = int(request.form.get("num_clusters", 3))

            if len(selected_features) >= 2:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[selected_features])

                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                df["Cluster"] = kmeans.fit_predict(scaled_data)

                # Convert dataframe to HTML
                clustered_data = df[["Cluster"] + selected_features].to_html(classes="table table-striped")

                # Create cluster plot
                if len(selected_features) == 2:
                    plt.figure(figsize=(8, 5))
                    sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]], hue=df["Cluster"], palette="viridis")
                    plt.title("Cluster Visualization")

                    img = io.BytesIO()
                    plt.savefig(img, format="png")
                    img.seek(0)
                    cluster_plot = base64.b64encode(img.getvalue()).decode()

    return render_template("index.html", cluster_plot=cluster_plot, clustered_data=clustered_data)

if __name__ == "__main__":
    app.run(debug=True)

