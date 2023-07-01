import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess the text
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word not in string.punctuation and word not in stopwords.words('english')]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Load the fine-tuned model and tokenizer
model_path = "Compressed_Distilled_Model"

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_path)

labels_dict = {0:"business",1:"culture",2:"lifestyle",3:"news",4:"opinion",5:"sport"}

# Make sure the model is in evaluation mode
model.eval()

def get_prediction(query):
    # Preprocess the query
    preprocessed_query = preprocess_text(query)

    # Tokenize the preprocessed query
    inputs = tokenizer(preprocessed_query, return_tensors="pt")

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label
    _, predicted_label = torch.max(outputs.logits, dim=1)
    return predicted_label.item()

if __name__ == "__main__":
    query = input("Enter your query: ")
    result = get_prediction(query)
    print(f"Predicted result: {labels_dict[result]}")
