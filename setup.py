import nltk
import os

# Define the folder where you want to download the NLTK data
custom_nltk_path = os.path.join(os.getcwd(), "nltk_data")

# Download 'punkt' and 'stopwords' into the specified folder
nltk.download('punkt', download_dir=custom_nltk_path)
nltk.download('stopwords', download_dir=custom_nltk_path)
nltk.download('punkt_tab', download_dir=custom_nltk_path)