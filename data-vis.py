"""
For data visualization
"""
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd


def main():
    x, y = load_breast_cancer(return_X_y=True)
    x = StandardScaler().fit_transform(x)

    # ref: https://www.datatechnotes.com/2020/11/tsne-visualization-example-in-python.html
    tsne = TSNE(n_components=2, learning_rate='auto', n_iter=1000, verbose=1,)
    z = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=df).set(title="cancer data T-SNE projection")
    plt.show()


if __name__ == "__main__":
    main()
