# Deploying SteamJackalope on RackNerd VPS

## Why RackNerd?

RackNerd offers affordable VPS plans starting at ~$15/month with:
- 2-4 GB RAM (enough for this app)
- Full root access
- No restrictive memory limits

## Prerequisites

- A RackNerd VPS (recommended: 2GB+ RAM, Ubuntu 22.04 or Debian 12)
- Domain name (optional but recommended)
- Basic familiarity with SSH and Linux commands

## Step 1: Initial Server Setup

### 1.1 Connect to Your VPS

```bash
ssh root@your-server-ip
```

### 1.2 Update System

```bash
apt update && apt upgrade -y
```

### 1.3 Create a Non-Root User

```bash
adduser steamjack
usermod -aG sudo steamjack
```

Switch to the new user:

```bash
su - steamjack
```

## Step 2: Install Dependencies

```bash
# Update again as the new user
sudo apt update

# Install Python, pip, and essential tools
sudo apt install -y python3 python3-pip python3-venv git build-essential

# Optional but recommended: Install NVIDIA drivers if you have a GPU (rare on VPS)
# Most VPS plans use CPU only, which is fine for this app
```

## Step 3: Clone Your Repository

```bash
cd ~
git clone https://github.com/misterriley/SteamJackalope.git
cd SteamJackalope
```

## Step 4: Set Up Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install system dependencies for sentence-transformers and torch
sudo apt install -y libopenblas-dev libomp-dev
```

**Note:** On Ubuntu/Debian, you might need additional libraries for optimal performance. The `requirements.txt` includes `sentence-transformers[onnx]` which will install ONNX Runtime automatically.

## Step 5: Prepare Data Files

The large data files (embeddings, tag vectors, metadata) are tracked with Git LFS. You need to:

```bash
# Install Git LFS
sudo apt install -y git-lfs
git lfs install

# Pull the large files (this may take a while depending on your connection)
git lfs pull
```

Alternatively, if you have the data files locally, upload them via SCP/SFTP:

```bash
# From your local machine (adjust paths as needed)
scp embeddings_desc.npy steamjack@your-server-ip:~/SteamJackalope/
scp embeddings_structural.npy steamjack@your-server-ip:~/SteamJackalope/
scp steam_tag_vectors.npy steamjack@your-server-ip:~/SteamJackalope/
scp quality_scores_grid.npy steamjack@your-server-ip:~/SteamJackalope/
scp metadata.parquet steamjack@your-server-ip:~/SteamJackalope/
scp w_desc.npy steamjack@your-server-ip:~/SteamJackalope/
scp w_structural.npy steamjack@your-server-ip:~/SteamJackalope/
scp mean_desc.npy steamjack@your-server-ip:~/SteamJackalope/
scp mean_structural.npy steamjack@your-server-ip:~/SteamJackalope/
scp tag_vectors_norms.npy steamjack@your-server-ip:~/SteamJackalope/
```

## Step 6: Configure the Application

The default configuration in `common/constants.py` is already set for ONNX backend and lazy loading. Verify the paths are correct:

```bash
cd ~/SteamJackalope
# Check that all file paths in common/constants.py point to the root directory
# The defaults should work if all .npy files are in the repo root
```

Ensure data files are present:

```bash
ls -lh *.npy *.parquet
# Should show all required files with sizes:
# embeddings_desc.npy (~39 MB)
# embeddings_structural.npy (~5 MB)
# steam_tag_vectors.npy (~13 MB)
# quality_scores_grid.npy (~6 MB)
# metadata.parquet (~22 MB)
# w_desc.npy, w_structural.npy, mean_desc.npy, mean_structural.npy (~1-2 MB each)
# tag_vectors_norms.npy (~2 MB)
```

## Step 7: Create a Systemd Service for the Backend

Create a systemd service file to run the FastAPI backend as a daemon:

```bash
sudo nano /etc/systemd/system/steamjackalope-backend.service
```

Paste the following configuration:

```ini
[Unit]
Description=SteamJackalope Backend API
After=network.target

[Service]
Type=simple
User=steamjack
WorkingDirectory=/home/steamjack/SteamJackalope
Environment="PATH=/home/steamjack/SteamJackalope/venv/bin"
Environment="PYTHONPATH=/home/steamjack/SteamJackalope"
ExecStart=/home/steamjack/SteamJackalope/venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable steamjackalope-backend
sudo systemctl start steamjackalope-backend
sudo systemctl status steamjackalope-backend
```

You should see it running. Check logs if there are errors:

```bash
sudo journalctl -u steamjackalope-backend -f
```

## Step 8: Set Up Nginx as a Reverse Proxy

While you can access the API directly on port 8000, using Nginx allows you to:
- Serve both backend and frontend from the same domain
- Add SSL encryption easily
- Handle static files efficiently

```bash
sudo apt install -y nginx
```

### 8.1 Nginx Configuration

Create a configuration file:

```bash
sudo nano /etc/nginx/sites-available/steamjackalope
```

Paste this configuration (adjust `your-domain.com` or use your IP):

```nginx
upstream backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;  # Or just your IP address

    # Backend API proxy
    location /api/ {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Also proxy root requests to backend for health checks
    location / {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/steamjackalope /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl reload nginx
```

**Note:** This configuration proxies all requests to the backend. For a production setup with the frontend also on the same server, you'll need to adjust routing. See Step 9.

## Step 9: Running the Frontend

### Option A: Run Frontend Separately (Recommended for Testing)

The frontend runs on port 8501 by default and communicates with the backend. You can start it manually:

```bash
cd ~/SteamJackalope
source venv/bin/activate
streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
```

Or create a separate systemd service:

```bash
sudo nano /etc/systemd/system/steamjackalope-frontend.service
```

```ini
[Unit]
Description=SteamJackalope Frontend
After=network.target steamjackalope-backend.service

[Service]
Type=simple
User=steamjack
WorkingDirectory=/home/steamjack/SteamJackalope
Environment="PATH=/home/steamjack/SteamJackalope/venv/bin"
Environment="PYTHONPATH=/home/steamjack/SteamJackalope"
ExecStart=/home/steamjack/SteamJackalope/venv/bin/streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 --server.enableCORS false
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable steamjackalope-frontend
sudo systemctl start steamjackalope-frontend
sudo systemctl status steamjackalobe-frontend
```

### Option B: Serve Frontend Through Nginx (Advanced)

If you want a single domain with both services:
- Backend on `your-domain.com/api/`
- Frontend on `your-domain.com/`

Update Nginx config:

```nginx
upstream backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Frontend (Streamlit) - serves at root
    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then configure Streamlit to work behind a proxy by setting the base URL. In `~/.streamlit/config.toml` or via environment:

```bash
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_ENABLECORS=false
```

The frontend code in `app/app.py` needs to know the backend URL. By default it uses `BACKEND_URL` from constants which defaults to `http://127.0.0.1:8000`. When deployed, set an environment variable:

```bash
export BACKEND_URL="http://127.0.0.1:8000"  # If both on same server, use localhost
```

Or if using a separate subdomain for the backend API, set it accordingly.

## Step 10: SSL with Let's Encrypt

Secure your site with free SSL certificates:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

Follow the prompts. Certbot will automatically configure Nginx with SSL and set up automatic renewal.

## Step 11: Firewall Configuration

Ensure only necessary ports are open:

```bash
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP (for Let's Encrypt)
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable
```

You don't need to expose ports 8000 or 8501 externally - Nginx proxies to them locally.

## Step 12: Monitoring and Management

### Useful Commands

```bash
# Check service status
sudo systemctl status steamjackalope-backend
sudo systemctl status steamjackalope-frontend

# View logs
sudo journalctl -u steamjackalope-backend -f
sudo journalctl -u steamjackalope-frontend -f

# Restart services
sudo systemctl restart steamjackalope-backend
sudo systemctl restart steamjackalope-frontend

# Check Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Check Memory Usage

```bash
htop  # or
top -u steamjack
free -h
```

## Step 13: Updating the Application

When you need to deploy updates:

```bash
cd ~/SteamJackalope
git pull
source venv/bin/activate
pip install -r requirements.txt  # Update dependencies if needed

# Restart services
sudo systemctl restart steamjackalope-backend
sudo systemctl restart steamjackalope-frontend
```

## Step 14: Backup Strategy

### Back Up Data Files

The data files (embeddings, metadata, etc.) are large but static. Create a backup script:

```bash
nano ~/backup_steamjack.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/home/steamjack/backups"
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "$BACKUP_DIR/steamjackalope_data_$DATE.tar.gz" \
    ~/SteamJackalope/*.npy \
    ~/SteamJackalope/*.parquet \
    ~/SteamJackalope/*.json 2>/dev/null

# Keep only last 7 backups
ls -t "$BACKUP_DIR"/steamjackalope_data_*.tar.gz | tail -n +8 | xargs -r rm
```

Make it executable and add to cron:

```bash
chmod +x ~/backup_steamjack.sh
crontab -e
# Add: 0 2 * * * /home/steamjack/backup_steamjack.sh
```

### Back Up the Whole Repository

```bash
# In your backup script, also include config files
tar -czf "$BACKUP_DIR/steamjackalope_full_$DATE.tar.gz" ~/SteamJackalope --exclude=venv
```

## Troubleshooting

### Port Already in Use

```bash
sudo ss -tulpn | grep :8000
sudo ss -tulpn | grep :8501
# Kill any stray processes
sudo kill -9 <PID>
```

### Memory Issues

The app needs ~1.5-2 GB when fully loaded. Check your VPS plan:

```bash
free -h
```

If you're hitting memory limits:
1. Close unnecessary processes
2. Add swap space (temporary fix):
   ```bash
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```
   Note: Swap on SSD is fine but not ideal for performance.

### Model Loading Fails

Ensure ONNX model file is present:

```bash
ls -la ~/.cache/torch/sentence_transformers/all-MiniLM-L6-v2/onnx/
# Should show model_quint8_avx2.onnx

# If not, the model will be downloaded on first use (requires outbound internet)
```

## Cost Estimate

- **RackNerd VPS (2GB RAM)**: ~$15-20/month
- **Domain name**: ~$12/year (optional)
- **Total first year**: ~$180-240 + domain

## Next Steps

1. Set up the RackNerd VPS
2. Deploy the application
3. Test thoroughly with real data
4. Configure monitoring (optional: set up Prometheus/Grafana or just use `htop`)
5. Set up automated backups
6. Document any issues and solutions

---

**Need Help?**
- Check logs: `sudo journalctl -u steamjackalope-* -f`
- Test backend health: `curl http://localhost:8000/games`
- Test frontend: Visit `http://your-server-ip:8501` (or the configured port)
- Test full flow: `http://your-domain.com` (Nginx proxy)

Good luck!