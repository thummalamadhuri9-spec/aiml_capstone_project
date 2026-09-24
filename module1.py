
import csv
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DataPipelineAssignment/1.0)"}


CATEGORIES = {
    "Travel": "travel_2",
    "Mystery": "mystery_3",
    "Historical Fiction": "historical-fiction_4",
    "Fiction": "fiction_10",
    "Fantasy": "fantasy_19",
}


def parse_star_rating(tag_classes):
    """The star rating is encoded as a CSS class, e.g. 'star-rating Three'."""
    words = {"One", "Two", "Three", "Four", "Five"}
    for cls in tag_classes:
        if cls in words:
            return cls
    return None


def scrape_category(category_name, slug):
    """Yield one dict per book across all paginated pages of a category."""
    page = 1
    while True:
        if page == 1:
            url = f"{BASE_URL}catalogue/category/books/{slug}/index.html"
        else:
            url = f"{BASE_URL}catalogue/category/books/{slug}/page-{page}.html"

        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            # Ran past the last page of this category.
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.select("article.product_pod")
        if not articles:
            break

        for art in articles:
            title = art.h3.a["title"]
            price_text = art.select_one("p.price_color").get_text(strip=True)
            star_rating_text = parse_star_rating(art.select_one("p.star-rating")["class"])
            availability_text = art.select_one("p.instock.availability").get_text(strip=True)

            yield {
                "title": title,
                "price_text": price_text,
                "star_rating_text": star_rating_text,
                "availability_text": availability_text,
                "category": category_name,
            }

        page += 1
        time.sleep(0.3)  # be polite to the practice server


def main(min_rows=60, min_categories=3, out_path="books_raw.csv"):
    rows = []
    categories_used = set()

    for category_name, slug in CATEGORIES.items():
        for row in scrape_category(category_name, slug):
            rows.append(row)
        categories_used.add(category_name)

        if len(rows) >= min_rows and len(categories_used) >= min_categories:
            # Already comfortably past both thresholds; keep going through
            # the categories we've started is fine, but we can stop early
            # once every category we intended to use has been scraped in
            # full (the loop above naturally finishes each category before
            # moving to the next, so no partial-category rows occur).
            pass

    print(f"Scraped {len(rows)} books across {len(categories_used)} categories.")
    assert len(rows) >= min_rows, "Did not reach the minimum row count"
    assert len(categories_used) >= min_categories, "Did not reach the minimum category count"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["title", "price_text", "star_rating_text", "availability_text", "category"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


import re
import sqlite3

import pandas as pd

GBP_TO_INR = 105.50

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def clean_price(price_text):
    """'£51.77' -> 51.77 (float). Returns None if unparseable."""
    if not isinstance(price_text, str):
        return None
    match = re.search(r"[\d]+\.?\d*", price_text.replace(",", ""))
    return float(match.group()) if match else None


def clean_rating(rating_text):
    return RATING_MAP.get(rating_text)


def clean_availability(availability_text):
    if not isinstance(availability_text, str):
        return None
    return "in stock" in availability_text.lower()


def load_and_clean(csv_path="books_raw.csv"):
    df = pd.read_csv(csv_path)

    df["price_gbp"] = df["price_text"].apply(clean_price)
    df["rating"] = df["star_rating_text"].apply(clean_rating)
    df["in_stock"] = df["availability_text"].apply(clean_availability)

    before = len(df)

    # Drop rows with unparseable rating/availability (categorical fields;
    # see docstring for justification).
    df = df.dropna(subset=["rating", "in_stock"])
    dropped_categorical = before - len(df)

    # Median-impute price_gbp within its category when missing, rather
    # than dropping the row.
    missing_price = df["price_gbp"].isna().sum()
    if missing_price:
        df["price_gbp"] = df.groupby("category")["price_gbp"].transform(
            lambda s: s.fillna(s.median())
        )
        # Fallback: if an entire category had no valid price at all,
        # fall back to the global median.
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    print(f"Loaded {before} raw rows -> {len(df)} clean rows "
          f"(dropped {dropped_categorical} for bad rating/availability, "
          f"median-imputed {missing_price} missing prices).")

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def build_database(df, db_path="books.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories (
            category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE books (
            book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            price_gbp   REAL NOT NULL,
            price_inr   REAL NOT NULL,
            rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            in_stock    INTEGER NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(category_id)
        );
        """
    )

    categories = sorted(df["category"].unique())
    cur.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(c,) for c in categories],
    )
    conn.commit()

    cat_id_map = dict(
        cur.execute("SELECT category_name, category_id FROM categories").fetchall()
    )

    book_rows = [
        (
            row.title,
            row.price_gbp,
            row.price_inr,
            row.rating,
            int(row.in_stock),
            cat_id_map[row.category],
        )
        for row in df.itertuples(index=False)
    ]
    cur.executemany(
        """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        book_rows,
    )
    conn.commit()
    conn.close()
    print(f"Loaded {len(book_rows)} books across {len(categories)} categories into {db_path}")


if __name__ == "__main__":
    cleaned = load_and_clean("books_raw.csv")
    build_database(cleaned, "books.db")

import sqlite3

import pandas as pd

DB_PATH = "books.db"

QUERIES = {
    # 1. SELECT / WHERE
    "Q1_in_stock_books": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE in_stock = 1
        LIMIT 10;
    """,

    # 2. ORDER BY + LIMIT
    "Q2_top10_most_expensive": """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,

    # 3. DISTINCT
    "Q3_distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,

    # 4. BETWEEN
    "Q4_midrange_price_books": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp;
    """,

    # 5. IN
    "Q5_specific_ratings": """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title;
    """,

    # 6. JOIN (books <-> categories): top-rated books per category
    "Q6_top_rated_per_category_join": """
        SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating = (
            SELECT MAX(b2.rating) FROM books b2 WHERE b2.category_id = b.category_id
        )
        ORDER BY c.category_name, b.title
        LIMIT 15;
    """,
}


def run_sql_queries(conn):
    results = {}
    for name, sql in QUERIES.items():
        df = pd.read_sql(sql, conn)
        results[name] = df
        print(f"\n=== {name} ===")
        print(sql.strip())
        print(df.to_string(index=False))
    return results


def pandas_only_join(conn):
    """Reproduce Q6 (the JOIN) using pd.merge on in-memory DataFrames, no SQL join."""
    books_df = pd.read_sql("SELECT * FROM books;", conn)
    categories_df = pd.read_sql("SELECT * FROM categories;", conn)

    merged = pd.merge(books_df, categories_df, on="category_id", how="inner")
    max_rating_per_cat = merged.groupby("category_id")["rating"].transform("max")
    top_rated = merged[merged["rating"] == max_rating_per_cat]
    top_rated = top_rated[["category_name", "title", "rating", "price_gbp"]].sort_values(
        ["category_name", "title"]
    ).reset_index(drop=True)
    return top_rated.head(15)


def main():
    conn = sqlite3.connect(DB_PATH)

    sql_results = run_sql_queries(conn)

    print("\n=== pandas pd.merge reproduction of the JOIN query (no SQL) ===")
    merge_result = pandas_only_join(conn)
    print(merge_result.to_string(index=False))

    print("\n=== Side-by-side check: pd.read_sql (JOIN query) vs pd.merge ===")
    sql_join_result = sql_results["Q6_top_rated_per_category_join"].reset_index(drop=True)
    match = sql_join_result.equals(merge_result.reset_index(drop=True))
    print(f"Row/column-wise equal: {match}")
    if not match:
        # Column name / dtype differences can make .equals() strict; show a
        # value-level comparison as a fallback so the equivalence is still visible.
        sorted_sql = sql_join_result.sort_values(list(sql_join_result.columns)).reset_index(drop=True)
        sorted_merge = merge_result.sort_values(list(merge_result.columns)).reset_index(drop=True)
        print("Value-level equal after sorting both frames:",
              sorted_sql.equals(sorted_merge))

    conn.close()


if __name__ == "__main__":
    main()