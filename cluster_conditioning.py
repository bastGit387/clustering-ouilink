import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def simplify_unit(unit):
    unit = str(unit).strip().lower()
    if unit in ['kg', 'k', 'kilogramme', 'kilo', 'kilos~']:
        return 'kg'
    elif unit in ['g', 'gramme']:
        return 'g'
    elif unit in ['l', 'litre']:
        return 'l'
    elif unit in ['cl']:
        return 'cl'
    elif unit in ['ml']:
        return 'ml'
    elif unit in ['btl', 'bt', 'bot']:
        return 'bottle'
    elif unit in ['u', 'unt', 'unite', 'pce', 'pu', 'pc']:
        return 'unit'
    elif unit in ['col', 'bqt', 'bte', 'sac']:
        return 'package'
    else:
        return unit


def cluster_conditioning(input_path, output_path, n_clusters=5):
    df = pd.read_csv(input_path)

    # Nettoyage des unités
    df['conditioning_unit_simplified'] = df['conditioning_unit'].apply(simplify_unit)

    # Réduction à 20 unités fréquentes
    top_units = df['conditioning_unit_simplified'].value_counts().nlargest(20).index
    df['unit_final'] = df['conditioning_unit_simplified'].apply(lambda x: x if x in top_units else 'other')

    # Encodage One-Hot
    encoder = OneHotEncoder(sparse=False)
    X_encoded = encoder.fit_transform(df[['unit_final']])

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_encoded)

    # PCA pour visualisation (facultatif)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_encoded)
    df['pca_1'] = components[:, 0]
    df['pca_2'] = components[:, 1]

    # Sauvegarde du fichier avec les clusters
    df.to_csv(output_path, index=False)
    print(f"✅ Résultats enregistrés dans {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering des unités de conditionnement")
    parser.add_argument("--input", type=str, required=True, help="Chemin du fichier CSV en entrée")
    parser.add_argument("--output", type=str, required=True, help="Chemin du fichier CSV de sortie")
    parser.add_argument("--n_clusters", type=int, default=5, help="Nombre de clusters (défaut : 5)")
    args = parser.parse_args()

    cluster_conditioning(args.input, args.output, args.n_clusters)
