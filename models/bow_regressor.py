from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

class BoWRegressor:
    def __init__(self, max_features=10000):        
        self.vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
        
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    
    def fit(self, X_train, y_train):
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        self.classifier.fit(X_train_vec, y_train)
    
    def forward(self, X_test):
        X_test_vec = self.vectorizer.transform(X_test)
        
        return self.classifier.predict(X_test_vec)
    
    def __call__(self, X_test):
        return self.forward(X_test)
    