## Run locally on terminal

1. Clone the repo:
   ```bash
   git clone https://github.com/LamiKaan/RoomSearch.git
   ```
2. Navigate inside the project root directory:
   ```bash
   cd RoomSearch
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
   or
   ```bash
   python -m venv venv
   ```
4. Activate the environment:
   ```bash
   source venv/bin/activate
   ```
5. Install dependencies (a later version of python might be required, I used 3.13):
   ```bash
   pip install -r requirements.txt
   ```
6. Edit the ".env.example" file with a valid OpenAI API key and change file name to ".env".
   <br>
   <br>
7. Edit the "parameters_example.json" file with the URLs of the images and the default user queries, and change file name to "parameters.json".
   <br>
   <br>
8. Run the main file and start chatting :
   ```bash
   python RoomSearch.py
   ```
   Note: This has a 7-8 minutes initialization time at first run. For next runs, this won't repeat. If any failure occurs during initialization, you can run again and it continues from where it failed.
   <br>
   <br>
9. The system makes conversation with the user, understands the features of the room they want to search for, displays most similar images (close this window to continue the conversation and search with new queries), and prints their URLs to the screen.
   <br>
   <br>
10. To quit chat, type "exit" or "quit" to the user prompt:
   ```bash
   User: quit
   User: exit
   ```
