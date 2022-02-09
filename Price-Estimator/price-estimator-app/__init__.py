"""
Ensure your create an app on heroku and then follow the command below:
"""

from app.price_estimator import app

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = "8000")
