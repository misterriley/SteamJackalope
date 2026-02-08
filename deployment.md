# Deployment Guide

This guide explains how to make the Steam Natural Language Search accessible from outside your local network.

## Option 1: Docker (Recommended for Cloud)
We have provided a `Dockerfile` that packages both the FastAPI backend and Streamlit frontend into a single container.

1.  **Build the image:**
    ```bash
    docker build -t steam-search .
    ```
2.  **Run the container:**
    ```bash
    docker run -p 8501:8501 -p 8000:8000 steam-search
    ```
3.  You can then deploy this container to services like **Google Cloud Run**, **AWS Fargate**, or **DigitalOcean App Platform**.

## Option 2: NGROK (Easiest for quick testing)
If you want to quickly show the site to someone without deploying to the cloud:
1.  Install [ngrok](https://ngrok.com/).
2.  Start your local environment using `run_test_env.bat`.
3.  In a new terminal, run:
    ```bash
    ngrok http 8501
    ```
4.  Ngrok will provide a public URL (e.g., `https://a1b2-c3d4.ngrok.io`) that anyone can use to access your site.

## Option 3: Manual VPS Setup (DigitalOcean/Linode/AWS EC2)
1.  Rent a Linux VPS (Ubuntu 22.04 recommended).
2.  Clone the repository and install dependencies:
    ```bash
    git clone <your-repo-url>
    cd SteamNaturalLanguageSearch
    pip install -r requirements.txt
    ```
3.  Run the servers in the background (using `nohup` or `tmux`):
    ```bash
    nohup python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 &
    nohup streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 &
    ```
4.  Access the site at `http://<your-vps-ip>:8501`.

## Network Architecture Security
*   **Internal Communication:** The frontend is configured to talk to the backend via `BACKEND_URL`. By default, this is `http://127.0.0.1:8000`. 
*   **Public Access:** Only port **8501** needs to be open to the public if you want users to use the UI. Port **8000** can remain private if the frontend and backend are on the same machine.
