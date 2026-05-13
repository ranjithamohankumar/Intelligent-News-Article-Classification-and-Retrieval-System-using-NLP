# ==============================
# Intelligent News Classification & Retrieval System
# ==============================

import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# डाउनलोड required NLTK data
nltk.download('stopwords')

# ==============================
# 1. Sample Dataset
# ==============================
data = {
    'text': [
        "India wins cricket match",
        "Stock market crashes today",
        "New AI technology launched",
        "Government passes new law",
        "Football team wins championship",
        "Tech companies invest in AI",
        "Elections results announced",
        "Business profits increase this quarter"
    ],
    'category': [
        "Sports", "Business", "Tech", "Politics",
        "Sports", "Tech", "Politics", "Business"
    ]
}

df = pd.DataFrame(data)

# ==============================
# 2. Preprocessing (Tokenization + Stemming)
# ==============================
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    
    processed_words = [
        stemmer.stem(word) for word in words if word not in stop_words
    ]
    
    return " ".join(processed_words)

df['processed_text'] = df['text'].apply(preprocess)

# ==============================
# 3. Feature Extraction (TF-IDF)
# ==============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['category']

# ==============================
# 4. Train Model (Naive Bayes)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

# ==============================
# 5. Model Evaluation
# ==============================
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# ==============================
# 6. Classification Function
# ==============================
def classify_news(article):
    processed = preprocess(article)
    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)
    return prediction[0]

# ==============================
# 7. Searchable Retrieval System
# ==============================
def search_news(keyword):
    keyword = keyword.lower()
    results = df[df['text'].str.lower().str.contains(keyword)]
    return results[['text', 'category']]

# ==============================
# 8. User Interaction
# ==============================
while True:
    print("\n1. Classify News")
    print("2. Search News")
    print("3. Exit")
    
    choice = input("Enter choice: ")
    
    if choice == '1':
        article = input("\nEnter news article:\n")
        category = classify_news(article)
        print("\nPredicted Category:", category)
    
    elif choice == '2':
        keyword = input("\nEnter keyword to search:\n")
        results = search_news(keyword)
        
        if results.empty:
            print("\nNo results found.")
        else:
            print("\nSearch Results:")
            print(results)
    
    elif choice == '3':
        print("Exiting...")
        break
    
    else:
        print("Invalid choice!")