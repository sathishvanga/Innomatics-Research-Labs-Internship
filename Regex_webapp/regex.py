from flask import Flask, render_template, request, jsonify
import re

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('regex.html')

@app.route('/match', methods=['POST'])
def match():
    test_string = request.form['test_string']
    regex_pattern = request.form['regex_pattern']
    
    matches = re.findall(regex_pattern, test_string)
    
    return jsonify(matches=matches)

@app.route('/validate_email', methods=['POST'])
def validate_email():
    email = request.form['email']
    
    # Regular expression pattern for validating email addresses
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(email_pattern, email):
        email_result = f"{email} is a valid email"
    else:
        email_result = f"{email} is an invalid email"
    
    return jsonify(email_result=email_result)

if __name__ == '__main__':
    app.run(debug=True)
