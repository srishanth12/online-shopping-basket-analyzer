import numpy as np
import pandas as pd
from faker import Faker
import random
import plotly.graph_objects as go
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore",category=DeprecationWarning)

np.random.seed(32)
random.seed(32)

fake =  Faker()

def generate_data(num_products=10, num_customers=100, num_transactions=500):
    products = [fake.word() for _ in range(num_products)]
    
    transactions = []

    for _ in range(num_transactions):
        customer_id = random.randint(1, num_customers)
        basket_size = random.randint(1, 5)
        basket = random.sample(products, basket_size)
        transactions.append({
            'customer_id': customer_id,  # Corrected typo here
            'products': basket
        })

    df = pd.DataFrame(transactions)
    df_encoded = df.explode('products').pivot_table(
        index='customer_id',
        columns='products',
        aggfunc=lambda x: 1,
        fill_value=0
    )

    return df_encoded
        #APRIORI ALGO! INTRO (work as detective -> which items are friends  often hangingout togther in basket{search for common pairs})

def simple_apriori(df, min_support=0.1, min_confidence=0.5):
    def support(item_set):
        return (df[list(item_set)].sum(axis=1) == len(item_set)).mean()
    
    items = set(df.columns)
    item_sets = [frozenset([item]) for item in items]  # Create frozenset for each individual item
    rules = []

    for k in range(2, len(items) + 1):
        item_sets = [s for s in combinations(items, k) if support(s) >= min_support]

        for item_set in item_sets:
            item_set = frozenset(item_set)
            for i in range(1, len(item_set)):
                for antecedent in combinations(item_set, i):
                    antecedent = frozenset(antecedent)  # shirts
                    consequent = item_set - antecedent  # items - shirts = pants
                    confidence = support(item_set) / support(antecedent)
                    if confidence >= min_confidence:
                        lift = confidence / support(consequent)
                        rules.append({
                            'antecedent': ','.join(antecedent),
                            'consequent': ','.join(consequent),
                            'support': support(item_set),
                            'confidence': confidence,
                            'lift': lift
                        })

                        if len(rules) >= 10:  # Stop if we have 10 rules
                            break

    return pd.DataFrame(rules).sort_values('lift', ascending=False)

                            
                            #K-means starting!(selecting best leaders)

def perform_kmeans_with_progress(df, n_clusters=3,update_intervals=5):
    
    scaler = StandardScaler() #transform data to understand all mem same lan
    df_scaled = scaler.fit_transform(df)

    Kmeans = KMeans(n_clusters=n_clusters, random_state = 32, max_iter=100)

    with tqdm(total=Kmeans.max_iter, desc="k-means Clustering") as pbar: # tqdm used to compare progress


        for i in range(Kmeans.max_iter):
            Kmeans.fit(df_scaled)
            pbar.update(1)

            if i  % update_intervals == 0:
                yield Kmeans.labels_

                if Kmeans.n_iter_ <= i + 1:
                    break

                return Kmeans.labels_
                                        
                                        #visualize the data

def visualize_apriori_rules(rules, top_n=10):
    top_rules = rules.head(top_n)

    fig = px.scatter_3d(
        top_rules, 
        x="support", 
        y="confidence", 
        z="lift",
        color="lift", 
        size="support",
        hover_name="antecedent", 
        hover_data=["consequent"],  # Corrected typo here
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        title=f"Top {top_n} Association Rules"
    )

    return fig


def visualize_kmeans_cluster(df, cluster_labels):
    pca = PCA(n_components=3)  # Use PCA to reduce the dimensionality to 3 for visualization
    pca_result = pca.fit_transform(df)
    
    fig = px.scatter_3d(
        x=pca_result[:, 0], 
        y=pca_result[:, 1], 
        z=pca_result[:, 2],
        color=cluster_labels,  # Correctly use cluster_labels as color
        title='Customer Cluster Visualization'
    )

    return fig

    
def main():
    print("Gathering synthetic Data........")

    df_encoded = generate_data(num_products=10,num_customers=100,num_transactions=500)
    print("Data Gathering Complete!")
    print(f"Dataset shape: {df_encoded.shape}")

    print("Performing Apriori Algo...")
    rules= simple_apriori(df_encoded, min_support=0.1, min_confidence=0.5)


    if not rules.empty:
        print(f"Apriori algo complete.vfond{len(rules)} rules.")
        viz = visualize_apriori_rules(rules)
        viz.write_html("apriori3d.html")
        print("apriori rules visuals saved as 'apriori3d.html")
    else:
        print("apriori algo failed!")

    print("performing K-means.")

    Kmeans_generator = perform_kmeans_with_progress(df_encoded, n_clusters=3, update_intervals=5)

    for i, labels in enumerate(Kmeans_generator):
        print(f"kmeans iteration {i*5}")
        viz = visualize_kmeans_cluster(df_encoded, labels)
        viz.write_html(f"customer_clusters_3d_step_{i}.html")
        print(f"intermediate visuals saved as customer_clusters_3d_step_{i}.html")

        final_labels = labels #last generated labels
        print("k-means clustering complete.")

        final_viz = visualize_kmeans_cluster(df_encoded, final_labels)
        final_viz.write_html("customer_cluster3dfianl.html")
        print("final customer cluster saved.")

    print("analysis complete!")

if __name__ == "__main__":
    main()
