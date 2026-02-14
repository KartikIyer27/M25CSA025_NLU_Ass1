
import sys
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# Check command line input

if len(sys.argv) < 2:
    print("Usage: python sports_politics.py <input_text_file>")
    sys.exit(1)

input_file = sys.argv[1]

if not os.path.exists(input_file):
    print("Error: input text file not found.")
    sys.exit(1)


# Reading documents from a folder

def read_documents_from_folder(folder_path):
    docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            full_path = os.path.join(folder_path, file_name)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    docs.append(text)
    return docs

# Loading dataset only 2 classes 

sports_folder = "data/sport"
politics_folder = "data/politics"

if not os.path.exists(sports_folder) or not os.path.exists(politics_folder):
    print("Error: sports or politics folder not found inside data/")
    sys.exit(1)

sports_docs = read_documents_from_folder(sports_folder)
politics_docs = read_documents_from_folder(politics_folder)

texts = sports_docs + politics_docs
labels = ["SPORTS"] * len(sports_docs) + ["POLITICS"] * len(politics_docs)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)



# Model 1: Naive Bayes + Bag of Words

bow = CountVectorizer(lowercase=True, stop_words="english")

X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)

nb_preds = nb_model.predict(X_test_bow)
nb_acc = accuracy_score(y_test, nb_preds)

print("Naive Bayes (Bag of Words) Accuracy:", round(nb_acc, 4))



# Model 2: Logistic Regression + TF-IDF

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

lr_preds = lr_model.predict(X_test_tfidf)
lr_acc = accuracy_score(y_test, lr_preds)

print("Logistic Regression (TF-IDF) Accuracy:", round(lr_acc, 4))


# Model 3: SVM + TF-IDF

svm_model = LinearSVC()
svm_model.fit(X_train_tfidf, y_train)

svm_preds = svm_model.predict(X_test_tfidf)
svm_acc = accuracy_score(y_test, svm_preds)

print("SVM (TF-IDF) Accuracy:", round(svm_acc, 4))


# Read input document to classify

with open(input_file, "r", encoding="utf-8") as f:
    input_text = f.read().strip()


# Final classification (using SVM)

input_vector = tfidf.transform([input_text])
final_label = svm_model.predict(input_vector)[0]

print("\nInput document classification:")
print(final_label)
