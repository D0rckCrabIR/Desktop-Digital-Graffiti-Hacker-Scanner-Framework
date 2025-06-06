NWAPT.py
Usage
NWAPT.py is a web vulnerability scanner. Follow these steps to use it:
1. Clone the Repository
git clone https://github.com/D0rckCrabIR/Desktop-Digital-Graffiti-Hacker-Scanner-Framework.git
cd Desktop-Digital-Graffiti-Hacker-Scanner-Framework

2. Set Up a Virtual Environment
Linux/macOS
python3 -m venv myenv
source myenv/bin/activate

Windows
python -m venv myenv
myenv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Scanner
python3 NWAPT.py https://example.com

Replace https://example.com with the target URL.
Notes

Ensure payloads.json is in the project directory.
Reports (JSON, HTML, PDF) are saved in the reports directory.
Only scan websites you have permission to test.

