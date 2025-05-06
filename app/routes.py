from flask import Blueprint, request, render_template
from .llm_utils import process_query

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            answer = process_query(question)
    return render_template("index.html", answer=answer)
