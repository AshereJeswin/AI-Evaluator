import csv
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import re

app = Flask(__name__)

def save_user(email, password):
    with open("users.csv", mode="a", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([email, password])

def check_user(email, password):
    with open("users.csv", mode="r") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if row == [email, password]:
                return True
        return False

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    message = ''  
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        password2 = request.form['password2']

        if check_email_exists(email):
            message = 'This email is already registered'
        elif password == password2:
            save_user(email, password)
            return redirect(url_for('login', message='Redirecting...'))
        else:
            message = 'Passwords do not match. Try again'

    return render_template('signup.html', message=message)

def check_email_exists(email):
    with open("users.csv", mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == email:
                return True
    return False

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        if check_user(email, password):
            csv_files = ['Q1.csv', 'Q2.csv', 'Q3.csv', 'Q4.csv', 'Q5.csv','Q6.csv']

            questions = []

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                if 'Question' in df and 'Answer' in df and len(df) > 0:
                    question = df['Question'].iloc[0]
                    answer = df['Answer'].iloc[0]
                    questions.append({'question': question, 'answer': answer})

            return render_template('rate_questions.html', questions=questions)
        else:
            return render_template('login.html', message='Try again!')

    return render_template('login.html', message='')

# Load the CSV files
csv_files = ['Q1.csv', 'Q2.csv', 'Q3.csv', 'Q4.csv', 'Q5.csv', 'Q6.csv']
models = {}

for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    df = df[['Answer', 'Ratings']]
    df['Answer'] = df['Answer'].str.lower()
    df['Answer'] = df['Answer'].str.replace('[^\w\s]', '')
    df = df.dropna()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Answer'])
    y = df['Ratings']

    model = RandomForestRegressor()

    model.fit(X, y)

    models[csv_file] = {'model': model, 'vectorizer': vectorizer}

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def predict_rating(user_answer, model, vectorizer, csv_file, top_n=5):
    # Preprocess the user answer
    user_answer = user_answer.lower()
    user_answer = re.sub('[^\w\s]', '', user_answer)
    input_vector = vectorizer.transform([user_answer])

    df = pd.read_csv(csv_file)

    if df['Answer'].isna().any():
        df = df.dropna(subset=['Answer'])  # Remove rows with NaN in 'Answer'

    reference_answers = df['Answer'].tolist()

    if not reference_answers:
        return None 

    # Calculate cosine similarity
    reference_vectors = vectorizer.transform(reference_answers)
    similarities = cosine_similarity(input_vector, reference_vectors)

    # Indices of the top-N similar answers
    similar_indices = similarities.argsort(axis=1)[:, -top_n:][:, ::-1]

    # Extract ratings of similar answers
    similar_ratings = df.iloc[similar_indices.ravel()]['Ratings'].values.reshape(-1, top_n)

    # Calculate the weighted average of similar ratings based on cosine similarity
    weighted_ratings = np.sum(similar_ratings * similarities[:, -top_n:], axis=1) / np.sum(similarities[:, -top_n:], axis=1)
    predicted_rating_ = weighted_ratings[0]

    predicted_rating = model.predict(input_vector)[0]

    return predicted_rating , predicted_rating_

@app.route('/rate_questions', methods=['GET', 'POST'])
def rate_questions():
    if request.method == 'POST':
        user_answers = request.form.getlist('answers')
        predicted_ratings = []

        for csv_file, user_answer in zip(csv_files, user_answers):
            model_info = models.get(csv_file)
            if model_info:
                model = model_info['model']
                vectorizer = model_info['vectorizer']

                predicted_rating, x = predict_rating(user_answer, model, vectorizer, csv_file)
                if predicted_rating is not None:
                    predicted_ratings.append(predicted_rating)

        return render_template('predicted_ratings.html', predicted_ratings=predicted_ratings)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def generate_classification_report():
    reports = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if 'Answer' in df and 'Ratings' in df and len(df) > 0:
            true_ratings = df['Ratings'].tolist()
            predicted_ratings = []  

        for row in df.itertuples(index=False):
            user_answer = str(row.Answer)
            model_info = models.get(csv_file)
            if model_info:
                model = model_info['model']
                vectorizer = model_info['vectorizer']
                predicted_rating, _ = predict_rating(user_answer, model, vectorizer, csv_file)
                if predicted_rating is not None:
                    predicted_ratings.append(predicted_rating)


            if not predicted_ratings:
                continue  

        valid_indices = ~np.isnan(true_ratings)
        true_ratings = true_ratings[valid_indices]
        predicted_ratings = np.array(predicted_ratings)[valid_indices]

        mae = mean_absolute_error(true_ratings, predicted_ratings)
        mse = mean_squared_error(true_ratings, predicted_ratings)
        r2 = r2_score(true_ratings, predicted_ratings)

        report = {
            'MAE': mae,
            'MSE': mse,
            'R2': r2
        }
        reports.append({'csv_file': csv_file, 'report': report})

    return render_template('report.html', reports=reports)

@app.route('/generate_report', methods=['GET'])
def generate_report():
    return generate_classification_report()

if __name__ == '__main__':
    app.run(debug=True)
