import pandas as pd


class text_prediction():
    """
    Take single file content as one string variable and generate label prediction
    """
    def __init__(self, data, vectorizer, model):
        self.data = data
        self.vectorizer = vectorizer
        self.model = model

    def tfidf_transform(self):
        """
        TF-IDF transform string content into word tf-idf vector
        :return: transformed vector
        """
        df = self.vectorizer.transform([self.data])
        return df

    def label_prediction(self):
        """
        Prediction label
        :return: predicted label
        """
        df = self.tfidf_transform()
        pred = self.model.predict(df)[0]
        print("Predicted label: " + pred)
        return pred

    def confidence(self):
        """
        Generate the probability of predicted label
        :return: prediction probability
        """
        df = self.tfidf_transform()
        confidence = max(self.model.predict_proba(df)[0]) * 100
        confidence = round(confidence, 2)
        print("Confidence level: " + str(confidence) + "%")
        return confidence

    def prediction_probability(self):
        """
        Display prediction probability for all labels in descending order
        :return: pandas dataframe for label probabilities
        """
        df = self.tfidf_transform()
        pred_prob = pd.DataFrame(self.model.predict_proba(df)[0],
                                 index=self.model.classes_,
                                 columns=["Probability"])
        pred_prob = pred_prob.sort_values(by="Probability", ascending=False)
        return pred_prob


class batch_prediction():
    """
    Take csv file with each row as string content from a document and predict labels for each document with confidence
    """
    def __init__(self, data, vectorizer, model):
        self.data = data
        self.vectorizer = vectorizer
        self.model = model

    def drop_empty(self):
        """
        Remove document with empty content
        :return: cleaned dataframe with no empty content
        """
        empty_content = self.data[self.data.isna().any(axis=1)]
        if len(empty_content) > 0:
            print("Document No." + str(empty_content.index[0]) + "is empty. Removing from file...")
            self.data.dropna(inplace=True)
        return self.data

    def tfidf_transform(self):
        """
        TF-IDF transform string content into word tf-idf matrix
        :return: transformed matrix
        """
        data = self.drop_empty()
        df = self.vectorizer.transform(data.iloc[:, 0])
        return df

    def label_prediction(self):
        """
        Predict labels for all documents
        :return: a list of predicted labels
        """
        df = self.tfidf_transform()
        pred = self.model.predict(df).tolist()
        return pred

    def confidence(self):
        """
        Generate the probability of predicted label
        :return: prediction probability for each document predicted label
        """
        df = self.tfidf_transform()
        conf = pd.DataFrame(self.model.predict_proba(df)).max(axis=1)
        return conf

    def save_prediction(self):
        """
        Organize prediction and confidence into one single dataframe and save prediction as csv file along with raw data
        :return: dataframe with prediction and confidence for each document
        """
        # Save prediction as csv
        result = pd.DataFrame({"Prediction": self.label_prediction(),
                               "Confidence": self.confidence()})
        print(result)

        update_file = self.data.merge(result, left_index=True, right_index=True)
        update_file.to_csv("Prediction.csv")
        return update_file











