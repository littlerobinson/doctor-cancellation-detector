from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    """
    A class to preprocess the data for machine learning models.

    Attributes:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        X (pd.DataFrame): The features.
        y (pd.DataFrame): The target.
        preprocessor (ColumnTransformer): The preprocessor for numerical and categorical features.
    """

    def __init__(self, df, X, y):
        """
        Initializes the DataPreprocessor with the input DataFrame and the target column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target column.
        """
        self.df = df
        self.X = X
        self.y = y
        self.preprocessor = None

    def preprocess(self):
        """
        Preprocesses the data by separating features and target, and creating a preprocessor for numerical and categorical features.
        """
        # Convert integer columns to float64 to avoid schema enforcement warnings
        self.X = self.X.astype(
            {
                col: "float64"
                for col in self.X.select_dtypes(include=["int64", "int32"]).columns
            }
        )

        # Identify numerical and categorical columns
        numeric_features = self.X.select_dtypes(include=["float64"]).columns
        categorical_features = self.X.select_dtypes(
            include=["object", "category"]
        ).columns

        # Create a preprocessor for numerical and categorical columns
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        categorical_transformer = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(drop="first"),
                )  # first column will be dropped to avoid creating correlations between features
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        self.preprocessor.fit_transform(self.X)
