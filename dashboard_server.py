#!/usr/bin/env python3
"""
Dashboard Web Server

This script starts a web server to make the research dashboard accessible over the network.
It allows you to access your dashboard remotely using your server's IP address.

Usage:
  python dashboard_server.py [--port 8080] [--host 0.0.0.0]
"""

import os
import sys
import argparse
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
import json
import time
import socket
import logging
from pathlib import Path
from output_manager import OutputManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard_server_verbose.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('dashboard-server')

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for the dashboard server"""
    
    def __init__(self, *args, **kwargs):
        self.output_mgr = OutputManager()
        self.dashboard_dir = self.output_mgr.dashboard_dir
        self.base_output_dir = self.output_mgr.base_output_dir
        super().__init__(*args, directory=self.base_output_dir, **kwargs)
    
    def log_message(self, format, *args):
        """Override log_message to use our custom logger"""
        logger.info(f"{self.address_string()} - {format%args}")
    
    def do_GET(self):
        """Handle GET requests"""
        # Parse URL
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        
        logger.info(f"GET request for: {path}")
        
        # Handle API requests
        if path.startswith('/api/'):
            self.handle_api_request(path[5:], parse_qs(parsed_url.query))
            return
            
        # Handle root path - redirect to dashboard
        if path == '/':
            logger.info("Root path accessed - redirecting to dashboard")
            self.send_response(302)  # Found/Redirect
            self.send_header('Location', '/_dashboard/dashboard.html')
            self.end_headers()
            return
            
        # Ensure dashboard is updated
        if path == '/_dashboard/dashboard.html':
            # Regenerate dashboard if it doesn't exist or is older than 5 minutes
            dashboard_path = os.path.join(self.dashboard_dir, 'dashboard.html')
            if not os.path.exists(dashboard_path) or time.time() - os.path.getmtime(dashboard_path) > 300:
                logger.info("Regenerating dashboard")
                self.output_mgr._update_dashboard_index()
        
        # Default behavior - let SimpleHTTPRequestHandler handle it
        return super().do_GET()
    
    def handle_api_request(self, api_path, query_params):
        """Handle API requests"""
        logger.info(f"API request: {api_path} with params {query_params}")
        
        if api_path == 'list':
            # Get filter params
            days = int(query_params.get('days', [None])[0]) if query_params.get('days') else None
            model = query_params.get('model', [None])[0]
            analysis_type = query_params.get('analysis', [None])[0]
            
            # Get outputs
            outputs = self.output_mgr.list_outputs(days=days, model=model, analysis_type=analysis_type)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(outputs).encode())
            
        elif api_path == 'refresh':
            # Regenerate dashboard
            self.output_mgr._update_dashboard_index()
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "message": "Dashboard refreshed"}).encode())
            
        else:
            # Unknown API endpoint
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "API endpoint not found"}).encode())

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Handle requests in a separate thread."""
    pass

def get_ip_addresses():
    """Get all IP addresses of the server"""
    hostname = socket.gethostname()
    
    # Get all IP addresses
    addresses = {
        'hostname': hostname,
        'localhost': '127.0.0.1',
        'interfaces': {}
    }
    
    # Get all network interfaces
    try:
        for interface in socket.getaddrinfo(socket.gethostname(), None):
            ip = interface[4][0]
            family_name = 'IPv4' if interface[0] == socket.AF_INET else 'IPv6'
            if ip not in ['127.0.0.1', '::1']:
                if family_name not in addresses['interfaces']:
                    addresses['interfaces'][family_name] = []
                if ip not in addresses['interfaces'][family_name]:
                    addresses['interfaces'][family_name].append(ip)
    except Exception as e:
        logger.error(f"Error getting network interfaces: {e}")
    
    # Try to get public IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        addresses['public_ip'] = s.getsockname()[0]
        s.close()
    except Exception as e:
        logger.error(f"Error getting public IP: {e}")
        addresses['public_ip'] = "Could not determine"
    
    return addresses

def main():
    """Main function to start the dashboard server"""
    parser = argparse.ArgumentParser(
        description="Web server for Villanova Research Dashboard"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0 - all interfaces)"
    )
    
    args = parser.parse_args()
    
    # Ensure dashboard exists
    output_mgr = OutputManager()
    output_mgr._update_dashboard_index()
    
    # Get IP addresses for logging
    ip_info = get_ip_addresses()
    logger.info(f"Server IP information: {json.dumps(ip_info, indent=2)}")
    
    # Create the server - using ThreadingHTTPServer for better handling of concurrent requests
    try:
        handler = DashboardHandler
        server = ThreadingHTTPServer((args.host, args.port), handler)
        
        # Print info
        logger.info(f"Starting dashboard server on http://{args.host}:{args.port}/")
        logger.info(f"Available URLs:")
        logger.info(f"- Local: http://localhost:{args.port}/")
        
        if 'public_ip' in ip_info:
            logger.info(f"- Public: http://{ip_info['public_ip']}:{args.port}/")
        
        if 'interfaces' in ip_info:
            for family, ips in ip_info['interfaces'].items():
                for ip in ips:
                    logger.info(f"- {family}: http://{ip}:{args.port}/")
        
        logger.info("Press Ctrl+C to stop the server")
        
        # Run the server until interrupted
        server.serve_forever()
    except OSError as e:
        if e.errno == 98:
            logger.error(f"ERROR: Port {args.port} is already in use. Try another port.")
        else:
            logger.error(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
        if 'server' in locals():
            server.shutdown()
            server.server_close()

if __name__ == "__main__":
    main()