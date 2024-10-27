from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    """
    A class to preprocess the data for machine learning models.

    Attributes:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        X (pd.DataFrame): The features DataFrame.
        y (pd.Series): The target Series.
        preprocessor (ColumnTransformer): The preprocessor for numerical and categorical features.
    """

    def __init__(self, df, target_column):
        """
        Initializes the DataPreprocessor with the input DataFrame and the target column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target column.
        """
        self.df = df
        self.target_column = target_column
        self.X = None
        self.y = None
        self.preprocessor = None

    def preprocess(self):
        """
        Preprocesses the data by separating features and target, and creating a preprocessor for numerical and categorical features.
        """
        # Separate features and target
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

        # Convert integer columns to float64 to avoid schema enforcement warnings
        self.X = self.X.astype(
            {col: "float64" for col in self.X.select_dtypes(include=["int64"]).columns}
        )

        # Identify numerical and categorical columns
        numeric_features = self.X.select_dtypes(include=["int64", "float64"]).columns
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

    def get_preprocessed_data(self):
        """
        Returns the preprocessed data.

        Returns:
            tuple: A tuple containing the non preprocessed features, the preprocessed features and the target.
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor is not initialized. Call preprocess() first."
            )
        return self.preprocessor.fit_transform(self.X), self.y
