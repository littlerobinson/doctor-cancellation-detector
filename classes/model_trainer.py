from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


class ModelTrainer:
    """
    A class to train machine learning models.

    Attributes:
        preprocessor (DataPreprocessor): The data preprocessor.
        model (object): The machine learning model.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
    """

    def __init__(self, preprocessor, model, test_size=0.2, random_state=42):
        """
        Initializes the ModelTrainer with the preprocessor, model, test size, and random state.

        Args:
            preprocessor (DataPreprocessor): The data preprocessor.
            model (object): The machine learning model.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): The seed used by the random number generator.
        """
        self.preprocessor = preprocessor
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.y_train = None
        self.y_test = None
        self.X_train = None
        self.X_test = None
        self.y_pred = None
        self.y_train_pred = None

    def train(self):
        """
        Trains the machine learning model using the preprocessed data.
        """
        # Get the preprocessed data
        X_preprocessed, y = self.preprocessor.get_preprocessed_data()

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_preprocessed,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Train the model
        self.model.fit(self.X_train, self.y_train)

        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_train_pred = self.model.predict(self.X_train)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)

        print("Accuracy:", accuracy)
        print(
            "Classification Report:\n", classification_report(self.y_test, self.y_pred)
        )

        return accuracy, f1

    def get_prediction(self):
        """
        Return the prediction.
        """
        if self.y_pred is None:
            raise ValueError("Model is not train. Call train method first.")
        return self.y_pred
