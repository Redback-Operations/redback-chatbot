import requests  # vulnerable version will be flagged
import sqlite3
from flask import render_template_string, request
import os

# Hardcoded secret (should be flagged)
secret_key = "super_secret_123"

# SQL injection risk
def get_user_data(user_id):
    conn = sqlite3.connect("example.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"  # vulnerable
    cursor.execute(query)
    return cursor.fetchall()

# SSRF vulnerability simulation
def fetch_url():
    url = request.args.get("url")
    response = requests.get(url)  # user-controlled input
    return response.text

# XSS: Unescaped rendering of user input
def unsafe_render():
    user_input = request.args.get("comment")
    return render_template_string(user_input)

# Tainted variable usage
def dangerous_use():
    user_input = input("Enter command:")
    os.system(user_input)  # tainted variable used directly

# Vulnerable library usage (mock simulation for your scanner)
response = requests.get("http://example.com")
print(response.status_code)
