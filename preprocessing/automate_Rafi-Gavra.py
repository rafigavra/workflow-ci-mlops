import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    preprocess_data(
        "datasets_raw/Titanic-Dataset.csv",
        "titanic_preprocessed.csv"
    )
