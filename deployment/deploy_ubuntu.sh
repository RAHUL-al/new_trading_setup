#!/bin/bash
# run this script as root to set up the environment on Ubuntu 22.04

echo "🔄 Updating system and installing dependencies..."
apt update && apt upgrade -y
apt install -y python3-venv python3-pip nginx memcached supervisor

echo "📦 Creating Virtual Environment..."
cd /root/Trading_setup_code
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing requirements..."
# Install standard dependencies
pip install -r backend/requirements.txt
# Install other dependencies needed for your system
pip install fastapi uvicorn websockets gunicorn pandas numpy

echo "⚙️ Setting up Systemd Service for FastAPI..."
cp /root/Trading_setup_code/deployment/fastapi.service /etc/systemd/system/fastapi.service
systemctl daemon-reload
systemctl enable fastapi
systemctl start fastapi

echo "🌐 Setting up Nginx..."
cp /root/Trading_setup_code/deployment/trading_setup.nginx /etc/nginx/sites-available/trading_setup
# Replace IP inside Nginx config with the server's public IPv4 automatically
SERVER_IP=$(curl -s -4 ifconfig.me)
sed -i "s/YOUR_DOMAIN_OR_SERVER_IP/$SERVER_IP/g" /etc/nginx/sites-available/trading_setup

ln -s /etc/nginx/sites-available/trading_setup /etc/nginx/sites-enabled/
# Remove default nginx site
rm -f /etc/nginx/sites-enabled/default

# Restart Nginx
nginx -t && systemctl restart nginx

echo "✅ Deployment complete!"
echo "You can now access your API and Dashboard at: http://$SERVER_IP/dashboard"
