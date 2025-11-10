## Poly Persona Screenshots

These screenshots walk through a full session, start to finish.

1. Landing on the dashboard with all users. I select Ariel. (Create users initally)
![Screenshot 2025-11-10 194349](sshots/Screenshot%202025-11-10%20194349.png)

2. Ariel’s space opens with the default General Chat. I ask for a friend persona.
![Screenshot 2025-11-10 194438](sshots/Screenshot%202025-11-10%20194438.png)

3. The system confirms the new Friend persona.
![Screenshot 2025-11-10 194508](sshots/Screenshot%202025-11-10%20194508.png)

4. The Friend thread is now available in the sidebar.
![Screenshot 2025-11-10 194524](sshots/Screenshot%202025-11-10%20194524.png)

5. Chatting with the Friend persona for a bit.
![Screenshot 2025-11-10 194750](sshots/Screenshot%202025-11-10%20194750.png)

6. I request an Enemy persona right away.
![Screenshot 2025-11-10 194823](sshots/Screenshot%202025-11-10%20194823.png)

7. The Enemy persona spins up instantly.
![Screenshot 2025-11-10 194841](sshots/Screenshot%202025-11-10%20194841.png)

8. Switching back to the Friend persona.
![Screenshot 2025-11-10 194909](sshots/Screenshot%202025-11-10%20194909.png)

9. The Friend replies as expected, showing cross-thread continuity.
![Screenshot 2025-11-10 195028](sshots/Screenshot%202025-11-10%20195028.png)

We bring the last message to the previous persona in the current persona intentionally since it's good for some cases (e.g., “back to my mentor thread, should I focus on organic growth?”).

## Setup Guide
1. **Clone the repository**
   ```bash
   git clone https://github.com/arihara-sudhan/poly-personas.git
   cd poly-personas
   ```

2. **Create Virtual Environment and Install dependencies (uv sync is enough)**
   ```bash
   uv sync
   ```

3. **Activate the created virtual environment**
   ```bash
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```


4. **Configure environment variables**
   - Create a `.env` file.
   - Set `GEMINI_API_KEY` to a valid Gemini API key.

5. **Start the development server**
   ```bash
   uv run main.py
   ```

6. **Open the app**
   - Visit `http://127.0.0.1:8000` in your browser.

7. **Initially, we won't have users**
![Screenshot 2025-11-10 203633.png](sshots/Screenshot 2025-11-10 203633.png)

8. **Create One**
![Screenshot 2025-11-10 203712.png](sshots/Screenshot 2025-11-10 203712.png)

9. **Click on The User**
![Screenshot 2025-11-10 203727.png](sshots/Screenshot 2025-11-10 203727.png)

10. **Happy, Persona-Changing Interaction!**