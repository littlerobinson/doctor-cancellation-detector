from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


class ModelTrainer:
    """
    A class to train machine learning models.

    Attributes:
        data_preprocessor (DataPreprocessor): The data preprocessor.
        pipeline (object): The machine learning model pipeline.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        y_pred (np.ndarray): The predictions for the test set.
        y_train_pred (np.ndarray): The predictions for the training set.
    """

    def __init__(self, data_preprocessor, pipeline, test_size=0.2, random_state=42):
        """
        Initializes the ModelTrainer with the preprocessor, model, test size, and random state.

        Args:
            preprocessor (DataPreprocessor): The data preprocessor.
            pipeline (object): Pipeline object.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator.
        """
        self.data_preprocessor = data_preprocessor
        self.pipeline = pipeline
        self.test_size = test_size
        self.random_state = random_state
        self.y_pred = None
        self.y_train_pred = None

    def train(self):
        """
        Trains the machine learning model using the preprocessed data.
        """

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_preprocessor.X,
            self.data_preprocessor.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.data_preprocessor.y,
        )

        # Train the model
        self.pipeline.fit(X_train, y_train)

        # Make predictions
        # Not need to transform X_test, already done in data_processor
        self.y_pred = self.pipeline.predict(X_test)

        self.y_train_pred = self.pipeline.predict(X_train)

        # Evaluate the model
        accuracy = accuracy_score(y_test, self.y_pred)
        f1 = f1_score(y_test, self.y_pred)

        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(y_test, self.y_pred))

        return accuracy, f1

    def get_prediction(self):
        """
        Return the prediction.
        """
        if self.y_pred is None:
            raise ValueError("Model is not train. Call train method first.")
        return self.y_pred
