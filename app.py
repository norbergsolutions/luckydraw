from flask import Flask, render_template, Response
import os
import requests
import zipfile
import mysql.connector
import glob
import random
from datetime import datetime
import schedule
import time
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# URLs for the files
file_urls = {
    "4D": "https://www.sportstoto.com.my/upload/4D.zip",
    "5D": "https://www.sportstoto.com.my/upload/5D.zip",
    "6D": "https://www.sportstoto.com.my/upload/6D.zip",
    "Star Toto 650": "https://www.sportstoto.com.my/upload/Toto650.zip",
    "Power Toto 655": "https://www.sportstoto.com.my/upload/Toto655.zip",
    "Supreme Toto 658": "https://www.sportstoto.com.my/upload/Toto658.zip",
}

# Mapping of game types to specific .txt filenames
game_to_file_map = {
    "4D": "4D.txt",
    "5D": "5D.txt",
    "6D": "6D.txt",
    "Star Toto 650": "Toto650.txt",
    "Power Toto 655": "Toto655.txt",
    "Supreme Toto 658": "Toto658.txt",
}

# Directory to store downloaded and extracted files
download_dir = "downloads"
os.makedirs(download_dir, exist_ok=True)

# Connect to the database
def connect_to_db():
    return mysql.connector.connect(
        host="DB_HOST",
        user="DB_USER",  # Replace with your MySQL username
        password="DB_PASSWORD",  # Replace with your MySQL password
        database="DB_NAME"
    )

# Clear all data in tables
def clear_all_tables():
    conn = connect_to_db()
    cursor = conn.cursor()
    tables = ["four_d", "five_d", "six_d", "star_toto_650", "power_toto_655", "supreme_toto_658"]
    for table in tables:
        print(f"Clearing data from table: {table}")
        cursor.execute(f"TRUNCATE TABLE {table};")
    conn.commit()
    cursor.close()
    conn.close()
    print("All tables have been cleared.")

# Process each file based on game type
def process_file(game, filepath):
    conn = connect_to_db()
    cursor = conn.cursor()

    with open(filepath, "r") as file:
        lines = file.readlines()

    # Skip the header row
    lines = lines[1:]

    if game == "4D":
        query = """
        INSERT INTO four_d (draw_no, draw_date, first_prize_no, second_prize_no, third_prize_no,
                            special_no1, special_no2, special_no3, special_no4, special_no5, special_no6,
                            special_no7, special_no8, special_no9, special_no10, consolation_no1, consolation_no2,
                            consolation_no3, consolation_no4, consolation_no5, consolation_no6, consolation_no7,
                            consolation_no8, consolation_no9, consolation_no10)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            first_prize_no = VALUES(first_prize_no),
            second_prize_no = VALUES(second_prize_no),
            third_prize_no = VALUES(third_prize_no),
            special_no1 = VALUES(special_no1),
            special_no2 = VALUES(special_no2),
            special_no3 = VALUES(special_no3),
            special_no4 = VALUES(special_no4),
            special_no5 = VALUES(special_no5),
            special_no6 = VALUES(special_no6),
            special_no7 = VALUES(special_no7),
            special_no8 = VALUES(special_no8),
            special_no9 = VALUES(special_no9),
            special_no10 = VALUES(special_no10),
            consolation_no1 = VALUES(consolation_no1),
            consolation_no2 = VALUES(consolation_no2),
            consolation_no3 = VALUES(consolation_no3),
            consolation_no4 = VALUES(consolation_no4),
            consolation_no5 = VALUES(consolation_no5),
            consolation_no6 = VALUES(consolation_no6),
            consolation_no7 = VALUES(consolation_no7),
            consolation_no8 = VALUES(consolation_no8),
            consolation_no9 = VALUES(consolation_no9),
            consolation_no10 = VALUES(consolation_no10)
        """
        for line in lines:
            parts = line.strip().split(",")
            draw_no, draw_date = parts[0], parts[1]

            # Ensure the date format is valid
            try:
                draw_date = datetime.strptime(draw_date, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                print(f"Invalid date format in line: {line}")
                continue

            values = [draw_no, draw_date] + parts[2:]
            cursor.execute(query, values)

    elif game == "5D":
        query = """
        INSERT INTO five_d (draw_no, draw_date, first_prize_no, second_prize_no, third_prize_no)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            first_prize_no = VALUES(first_prize_no),
            second_prize_no = VALUES(second_prize_no),
            third_prize_no = VALUES(third_prize_no)
        """
        for line in lines:
            parts = line.strip().split(",")
            draw_no, draw_date = parts[0], parts[1]

            # Ensure the date format is valid
            try:
                draw_date = datetime.strptime(draw_date, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                print(f"Invalid date format in line: {line}")
                continue

            values = [draw_no, draw_date] + parts[2:]
            cursor.execute(query, values)

    elif game == "6D":
        query = """
        INSERT INTO six_d (draw_no, draw_date, first_prize_no)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE first_prize_no = VALUES(first_prize_no)
        """
        for line in lines:
            parts = line.strip().split(",")
            draw_no, draw_date = parts[0], parts[1]

            # Ensure the date format is valid
            try:
                draw_date = datetime.strptime(draw_date, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                print(f"Invalid date format in line: {line}")
                continue

            values = [draw_no, draw_date] + parts[2:]
            cursor.execute(query, values)

    elif game == "Star Toto 650":
        query = """
        INSERT INTO star_toto_650 (draw_no, draw_date, drawn_no1, drawn_no2, drawn_no3, drawn_no4, drawn_no5,
                                   drawn_no6, bonus_no, jackpot1, jackpot2)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            drawn_no1 = VALUES(drawn_no1),
            drawn_no2 = VALUES(drawn_no2),
            drawn_no3 = VALUES(drawn_no3),
            drawn_no4 = VALUES(drawn_no4),
            drawn_no5 = VALUES(drawn_no5),
            drawn_no6 = VALUES(drawn_no6),
            bonus_no = VALUES(bonus_no),
            jackpot1 = VALUES(jackpot1),
            jackpot2 = VALUES(jackpot2)
        """
        for line in lines:
            parts = line.strip().split(",")
            draw_no, draw_date = parts[0], parts[1]

            # Ensure the date format is valid
            try:
                draw_date = datetime.strptime(draw_date, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                print(f"Invalid date format in line: {line}")
                continue

            values = [draw_no, draw_date] + parts[2:]
            cursor.execute(query, values)

    elif game == "Power Toto 655":
        query = """
        INSERT INTO power_toto_655 (draw_no, draw_date, drawn_no1, drawn_no2, drawn_no3, drawn_no4, drawn_no5,
                                    drawn_no6, jackpot)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            drawn_no1 = VALUES(drawn_no1),
            drawn_no2 = VALUES(drawn_no2),
            drawn_no3 = VALUES(drawn_no3),
            drawn_no4 = VALUES(drawn_no4),
            drawn_no5 = VALUES(drawn_no5),
            drawn_no6 = VALUES(drawn_no6),
            jackpot = VALUES(jackpot)
        """
        for line in lines:
            parts = line.strip().split(",")
            draw_no, draw_date = parts[0], parts[1]

            # Ensure the date format is valid
            try:
                draw_date = datetime.strptime(draw_date, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                print(f"Invalid date format in line: {line}")
                continue

            values = [draw_no, draw_date] + parts[2:]
            cursor.execute(query, values)

    elif game == "Supreme Toto 658":
        query = """
        INSERT INTO supreme_toto_658 (draw_no, draw_date, drawn_no1, drawn_no2, drawn_no3, drawn_no4, drawn_no5,
                                      drawn_no6, jackpot)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            drawn_no1 = VALUES(drawn_no1),
            drawn_no2 = VALUES(drawn_no2),
            drawn_no3 = VALUES(drawn_no3),
            drawn_no4 = VALUES(drawn_no4),
            drawn_no5 = VALUES(drawn_no5),
            drawn_no6 = VALUES(drawn_no6),
            jackpot = VALUES(jackpot)
        """
        for line in lines:
            parts = line.strip().split(",")
            draw_no, draw_date = parts[0], parts[1]

            # Ensure the date format is valid
            try:
                draw_date = datetime.strptime(draw_date, "%Y%m%d").strftime("%Y-%m-%d")
            except ValueError:
                print(f"Invalid date format in line: {line}")
                continue

            values = [draw_no, draw_date] + parts[2:]
            cursor.execute(query, values)

    conn.commit()
    cursor.close()
    conn.close()

def is_game_day():
    # Get the current day of the week (0=Monday, 1=Tuesday, ..., 6=Sunday)
    today = datetime.today().weekday()
    # Draw days are Wednesday (2), Saturday (5), and Sunday (6)
    return today in [2, 5, 6]

def scheduled_task():
    print("Running scheduled task...")
    if is_game_day():
        print("It's a draw day. Processing files and generating predictions...")
        clear_all_tables()
        download_and_process()
        generate_predictions_for_today()
    else:
        print("No predictions today. It's not a draw day.")

def generate_predictions_for_today():
    games = {
        "4D": 3,  # Generate 3 predictions
        "5D": 3,
        "6D": 3,
        "Star Toto 650": 1,  # Generate 1 prediction
        "Power Toto 655": 1,
        "Supreme Toto 658": 1,
    }

    for game, count in games.items():
        predictions = predict_numbers(game, count=count)
        for prediction in predictions:
            save_prediction_to_db(game, prediction)

def get_todays_predictions():
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)
    prediction_date = datetime.today().strftime('%Y-%m-%d')

    query = "SELECT game, predicted_numbers FROM predictions WHERE prediction_date = %s"
    cursor.execute(query, (prediction_date,))
    predictions = {row['game']: row['predicted_numbers'] for row in cursor.fetchall()}

    cursor.close()
    conn.close()
    return predictions

# Generate predictions for a game
def predict_numbers(game, count=1):
    predictions = []
    for _ in range(count):
        if game == "4D":
            predicted = [random.randint(0, 9) for _ in range(4)]  # Predict 4 digits
        elif game == "5D":
            predicted = [random.randint(0, 9) for _ in range(5)]  # Predict 5 digits
        elif game == "6D":
            predicted = [random.randint(0, 9) for _ in range(6)]  # Predict 6 digits
        elif game == "Star Toto 650":
            predicted = [random.randint(1, 50) for _ in range(6)]  # Predict 6 numbers (1–50)
            bonus = random.randint(1, 50)  # Bonus number
            predicted.append(f"Bonus: {bonus}")
        elif game == "Power Toto 655":
            predicted = [random.randint(1, 55) for _ in range(6)]  # Predict 6 numbers (1–55)
        elif game == "Supreme Toto 658":
            predicted = [random.randint(1, 58) for _ in range(6)]  # Predict 6 numbers (1–58)
        else:
            predicted = []

        predictions.append(" ".join(map(str, predicted)))  # Add the prediction as a string

    return predictions

# Save predictions to the database
def save_prediction_to_db(game, predicted_numbers):
    conn = connect_to_db()
    cursor = conn.cursor()
    prediction_date = datetime.today().strftime('%Y-%m-%d')

    # Check how many predictions already exist for this game today
    query_check = "SELECT COUNT(*) FROM predictions WHERE game = %s AND prediction_date = %s"
    cursor.execute(query_check, (game, prediction_date))
    count = cursor.fetchone()[0]

    # Allow up to 3 predictions for 4D, 5D, 6D; only 1 for others
    max_predictions = 3 if game in ["4D", "5D", "6D"] else 1
    if count >= max_predictions:
        print(f"Maximum predictions reached for {game} on {prediction_date}. Skipping.")
    else:
        # Insert new prediction
        query_insert = """
        INSERT INTO predictions (game, prediction_date, predicted_numbers)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query_insert, (game, prediction_date, predicted_numbers))
        conn.commit()
        print(f"Prediction saved for {game}: {predicted_numbers}")

    cursor.close()
    conn.close()

# Compare predictions with actual results
def compare_predictions_with_results(game, actual_numbers):
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)
    prediction_date = datetime.today().strftime('%Y-%m-%d')

    # Fetch the latest prediction for the game
    query = """
    SELECT * FROM predictions WHERE game = %s AND prediction_date = %s
    """
    cursor.execute(query, (game, prediction_date))
    prediction = cursor.fetchone()

    if prediction:
        # Compare predicted numbers with actual numbers
        predicted_numbers = prediction['predicted_numbers'].split()
        actual_numbers_list = actual_numbers.split()

        correct = sum(1 for p, a in zip(predicted_numbers, actual_numbers_list) if p == a)
        accuracy = (correct / len(actual_numbers_list)) * 100

        # Update the prediction entry with actual results and accuracy
        update_query = """
        UPDATE predictions
        SET actual_numbers = %s, accuracy = %s
        WHERE id = %s
        """
        cursor.execute(update_query, (actual_numbers, accuracy, prediction['id']))
        conn.commit()
        print(f"Updated prediction for {game}: Accuracy = {accuracy:.2f}%")
    else:
        print(f"No prediction found for {game} on {prediction_date}.")

    cursor.close()
    conn.close()

# Download, extract, and process
def download_and_process():
    for game, url in file_urls.items():
        print(f"Processing {game}...")
        zip_path = os.path.join(download_dir, f"{game}.zip")

        # Download the zip file
        response = requests.get(url)
        with open(zip_path, "wb") as file:
            file.write(response.content)

        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(download_dir)

        # Get the specific file name for the game
        expected_txt_name = os.path.join(download_dir, game_to_file_map[game])

        if not os.path.exists(expected_txt_name):
            print(f"Error: Expected file {expected_txt_name} not found for {game}.")
            continue

        # Process the extracted .txt file
        process_file(game, expected_txt_name)
        print(f"Finished processing {game}.")

# Function to generate trend plot for a game
def generate_trend_plot(game):
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)

    # Fetch accuracy data for the game
    query = """
    SELECT prediction_date, accuracy FROM predictions
    WHERE game = %s AND accuracy IS NOT NULL
    ORDER BY prediction_date ASC
    """
    cursor.execute(query, (game,))
    data = cursor.fetchall()

    cursor.close()
    conn.close()

    if not data:
        return None  # No data to plot

    # Extract dates and accuracy values
    dates = [row['prediction_date'] for row in data]
    accuracies = [row['accuracy'] for row in data]

    # Plot accuracy over time
    plt.figure(figsize=(8, 5))
    plt.plot(dates, accuracies, marker='o', linestyle='-', label=f"{game} Accuracy")
    plt.title(f"Accuracy Trend for {game}")
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Convert the plot to a base64 string
    plot_data = base64.b64encode(img.getvalue()).decode('utf-8')
    return plot_data

@app.route('/')
def index():
    games = ["4D", "5D", "6D", "Star Toto 650", "Power Toto 655", "Supreme Toto 658"]

    # Fetch today's predictions
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)
    prediction_date = datetime.today().strftime('%Y-%m-%d')

    query = """
    SELECT game, predicted_numbers FROM predictions WHERE prediction_date = %s
    """
    cursor.execute(query, (prediction_date,))
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    # Organize predictions by game
    predictions = {game: [] for game in games}
    for row in data:
        predictions[row['game']].append(row['predicted_numbers'])

    # Generate trend plots for each game
    trend_images = {}
    for game in games:
        trend_plot = generate_trend_plot(game)
        if trend_plot:
            trend_images[game] = trend_plot

    # Check if today is a game day
    game_day_status = "It's a game day!" if is_game_day() else "No games today."

    return render_template("index.html", games=games, predictions=predictions, trend_images=trend_images, game_day_status=game_day_status)

@app.route('/results/<game>')
def show_results(game):
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)

    # Map the game to its table name
    game_to_table = {
        "4D": "four_d",
        "5D": "five_d",
        "6D": "six_d",
        "Star Toto 650": "star_toto_650",
        "Power Toto 655": "power_toto_655",
        "Supreme Toto 658": "supreme_toto_658",
    }

    table_name = game_to_table.get(game)
    if not table_name:
        return "Invalid game!"

    # Fetch results from the database
    cursor.execute(f"SELECT * FROM {table_name}")
    results = cursor.fetchall()
    conn.close()

    return render_template("results.html", game=game, results=results)

@app.route('/predict/<game>')
def predict(game):
    predicted_numbers = predict_numbers(game)
    save_prediction_to_db(game, predicted_numbers)
    return f"Predicted numbers for {game}: {predicted_numbers}"

@app.route('/compare/<game>/<actual_numbers>')
def compare(game, actual_numbers):
    compare_predictions_with_results(game, actual_numbers)
    return f"Compared predictions for {game} with actual numbers: {actual_numbers}"

@app.route('/accuracy/<game>')
def show_accuracy(game):
    conn = connect_to_db()
    cursor = conn.cursor(dictionary=True)

    # Fetch accuracy data for the game
    query = """
    SELECT prediction_date, accuracy FROM predictions
    WHERE game = %s AND accuracy IS NOT NULL
    ORDER BY prediction_date ASC
    """
    cursor.execute(query, (game,))
    data = cursor.fetchall()

    cursor.close()
    conn.close()

    if not data:
        return f"No accuracy data available for {game}."

    # Extract dates and accuracy values
    dates = [row['prediction_date'] for row in data]
    accuracies = [row['accuracy'] for row in data]

    # Plot accuracy over time
    plt.figure(figsize=(10, 6))
    plt.plot(dates, accuracies, marker='o', linestyle='-', label=f"{game} Accuracy")
    plt.title(f"Accuracy Trend for {game}")
    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Return the plot as an image response
    return Response(img, mimetype='image/png')

if __name__ == "__main__":
    print("Starting the Flask app with scheduled tasks...")

    # Only process data if today is a draw day
    if is_game_day():
        print("Today is a draw day. Processing data...")
        clear_all_tables()
        download_and_process()
        generate_predictions_for_today()
    else:
        print("Today is not a draw day. Skipping initial processing.")

    # Schedule the task to run at 7:00 AM every day
    schedule.every().day.at("07:00").do(scheduled_task)

    # Run Flask in a separate thread
    from threading import Thread

    def run_flask():
        app.run(debug=False, use_reloader=False)

    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(1)