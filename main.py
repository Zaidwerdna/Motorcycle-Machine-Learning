import pandas as pd
import math
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('all_bikez_curated.csv')
df.columns = (
    df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
)

def text_scrubber(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\W+", "", str(value).strip().lower())

def number_checker(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def bucketizer(value, bucket_size, prefix):
    if number_checker(value):
        base = int(math.floor(float(value) / bucket_size) * bucket_size)
        return prefix + "_" + str(base)
    else:
        return ""

BUCKET_MAP = {
    "year": 1,
    "rating": 1,
    "displacement_ccm": 150,
    "power_hp": 20,
    "torque_nm": 25,
    "bore_mm": 15,
    "stroke_mm": 15,
    "fuel_capacity_lts": 5,
    "dry_weight_kg": 30,
    "wheelbase_mm": 50,
    "seat_height_mm": 20,
}

def build_feature_token(row):
    tokens = []

    for column_name, row_value in row.items():
        if pd.isna(row_value):
            continue

        if column_name in BUCKET_MAP and number_checker(row_value):
            size = BUCKET_MAP[column_name]
            token = bucketizer(row_value, size, column_name)
            if token:
                tokens.append(token)
            continue

        clean_token = text_scrubber(row_value)
        if clean_token:
            tokens.append(column_name + "_" + clean_token)

    return " ".join(tokens)

df["feature_tokens"] = df.apply(build_feature_token, axis=1)

vectorizer = TfidfVectorizer(min_df=1)
matrix = vectorizer.fit_transform(df["feature_tokens"])

knn = NearestNeighbors(metric = "cosine", algorithm = "brute")
knn.fit(matrix)

SIMPLE_COLUMNS = [
    "brand", "model", "year", "category",
    "displacement_ccm", "color_options", "seat_height_mm"
]
INTERMEDIATE_COLUMNS = [
    "power_hp", "torque_nm", "engine_cylinder",
    "fuel_capacity_lts", "dry_weight_kg"
]
ADVANCED_COLUMNS = [
    "engine_stroke", "gearbox", "bore_mm", "stroke_mm",
    "fuel_system", "fuel_control", "cooling_system",
    "transmission_type", "wheelbase_mm",
    "front_brakes", "rear_brakes", "front_suspension",
    "rear_suspension"
]
PROMPT_SIMPLE = {
    "brand": "What brand do you want? (e.g., Honda, Yamaha) ",
    "model": "Do you have a specific model in mind? ",
    "year": "What model year (or range)? ",
    "category": "What category of motorcycle? (e.g., sport, cruiser, touring) ",
    "displacement_ccm": "Engine size in cc? (e.g.,300, 600, 1000) ",
    "color_options": "Preferred color? (e.g., blue, black) ",
    "seat_height_mm": "Seat height in mm? (e.g., 810). This is important for short/tall riders.",
}
PROMPT_INTERMEDIATE = {
    "power_hp": "How much horsepower? ",
    "torque_nm": "How much torque (Nm)? ",
    "engine_cylinder": "How many cylinders? ",
    "fuel_capacity_lts": "Fuel capacity in liters? ",
    "dry_weight_kg": "Dry weight in kg? ",
}
PROMPT_ADVANCED = {
    "engine_stroke": "Engine stroke (e.g., 4-stroke)? ",
    "gearbox": "Gearbox details? (e.g., 6-speed) ",
    "bore_mm": "Bore in mm? ",
    "stroke_mm": "Stroke in mm? ",
    "fuel_system": "Fuel system type? (e.g., injection, carburetor) ",
    "fuel_control": "Fuel control system? (e.g., DOHC, SOHC) ",
    "cooling_system": "Cooling system? (e.g., liquid, air) ",
    "transmission_type": "Transmission type? (e.g., chain, belt) ",
    "wheelbase_mm": "Wheelbase in mm? ",
    "front_brakes": "Front brake type? ",
    "rear_brakes": "Rear brake type? ",
    "front_suspension": "Front suspension type? ",
    "rear_suspension": "Rear suspension type? ",
}

DIGIT_TO_WORD = {"1":"one", "2":"two", "3":"three", "4":"four", "5":"five", "6":"six", "7":"seven", "8":"eight"}

def collect_answers(level_choice, input_fn=input):
    """
    Ask questions based on the chosen level.
    level_choice: '1'/'simple', '2'/'intermediate', '3'/'advanced'
    Returns a dict {column: answer_string}.
    """
    print("\n(Press Enter to skip any question.)")
    answers = {}

    # Decide which columns and prompts to use without dict merging
    if level_choice == "1" or level_choice.lower() == "simple":
        columns = SIMPLE_COLUMNS
        prompts = PROMPT_SIMPLE
    elif level_choice == "2" or level_choice.lower() == "intermediate":
        # simple + intermediate
        columns = SIMPLE_COLUMNS + INTERMEDIATE_COLUMNS
        prompts = {}
        # copy prompts in simple
        for k in SIMPLE_COLUMNS:
            prompts[k] = PROMPT_SIMPLE[k]
        # add prompts in intermediate
        for k in INTERMEDIATE_COLUMNS:
            prompts[k] = PROMPT_INTERMEDIATE[k]
    else:
        # simple + intermediate + advanced
        columns = SIMPLE_COLUMNS + INTERMEDIATE_COLUMNS + ADVANCED_COLUMNS
        prompts = {}
        for k in SIMPLE_COLUMNS:
            prompts[k] = PROMPT_SIMPLE[k]
        for k in INTERMEDIATE_COLUMNS:
            prompts[k] = PROMPT_INTERMEDIATE[k]
        for k in ADVANCED_COLUMNS:
            prompts[k] = PROMPT_ADVANCED[k]

    # Ask questions in order
    for col in columns:
        question = prompts[col]
        value = input_fn(question)
        if value is None:
            value = ""
        value = value.strip()
        answers[col] = value

    return answers

def tokens_from_answers(answers):
    tokens = []
    i = 0
    for col in answers:
        raw = answers[col]
        if raw == "":
            continue

        # numeric/bucket columns
        if col in BUCKET_MAP:
            cleaned = raw.replace(",", " ")
            match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
            if match:
                num_str = match.group(1)
                if number_checker(num_str):
                    val = float(num_str)
                    tok = bucketizer(val, BUCKET_MAP[col], col)
                    if tok:
                        # add if not already present
                        if tok not in tokens:
                            tokens.append(tok)
            continue
        cleaned_text = text_scrubber(raw)
        if cleaned_text != "":
            tok2 = col + "_" + cleaned_text
            if tok2 not in tokens:
                tokens.append(tok2)

        i = i + 1

    # join space-separated
    return " ".join(tokens)

def run_qna_by_level(level_choice, top_k=5, input_fn=input):
    answers = collect_answers(level_choice, input_fn)
    query_tokens = tokens_from_answers(answers)

    if query_tokens == "":
        print("\nNo useful tokens from your answers. Try at least one value.")
        return

    q_vec = vectorizer.transform([query_tokens])
    n = len(df)
    if top_k < n:
        n_neighbors = top_k
    else:
        n_neighbors = n

    distances, indices = knn.kneighbors(q_vec, n_neighbors=n_neighbors)
    sims = 1.0 - distances[0]

    hits = df.iloc[indices[0]].copy()
    hits.insert(0, "similarity", np.round(sims, 4))

    show_cols = []
    candidates = [
        "similarity","brand","model","year","category",
        "displacement_ccm","power_hp","torque_nm","engine_cylinder",
        "seat_height_mm","dry_weight_kg","wheelbase_mm","color_options"
    ]
    for c in candidates:
        if c in hits.columns:
            show_cols.append(c)

    print("\nQuery tokens: " + query_tokens)
    print(hits[show_cols].to_string(index=False))

level = input("Choose level: 1 = Simple, 2=Intermediate, 3=Advanced): ")
run_qna_by_level(level, top_k=10)




