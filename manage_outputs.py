#!/usr/bin/env python3
"""
Output Management Utility Script

This script provides command-line utilities for organizing and managing
research outputs from the villanova_research project.

Usage:
  python manage_outputs.py dashboard      # Generate dashboard for existing outputs
  python manage_outputs.py open-dashboard # Open the dashboard in a browser
  python manage_outputs.py clean [days]   # Clean outputs older than specified days (default: 30)
  python manage_outputs.py list [days]    # List outputs from the last N days (default: all)
  python manage_outputs.py compare [dir1] [dir2] ... # Create a comparison directory
"""

import sys
import os
import argparse
import webbrowser
import subprocess
from output_manager import OutputManager

def open_dashboard(output_mgr):
    """Open the dashboard in a browser"""
    dashboard_path = os.path.join(output_mgr.dashboard_dir, "dashboard.html")
    
    if not os.path.exists(dashboard_path):
        output_mgr._update_dashboard_index()  # Ensure dashboard exists
        
    dashboard_url = f"file://{os.path.abspath(dashboard_path)}"
    print(f"Opening dashboard: {dashboard_url}")
    
    try:
        webbrowser.open(dashboard_url)
        return True
    except Exception as e:
        print(f"Failed to open browser: {e}")
        print(f"Please open {dashboard_url} manually.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Output Management Dashboard for Villanova Research Project"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Generate dashboard for existing outputs")
    
    # Open dashboard command
    open_parser = subparsers.add_parser("open-dashboard", help="Open dashboard in a browser")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean old outputs")
    clean_parser.add_argument(
        "days", 
        nargs="?", 
        type=int, 
        default=30,
        help="Number of days to keep (default: 30)"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available outputs")
    list_parser.add_argument(
        "days", 
        nargs="?", 
        type=int, 
        default=None,
        help="List outputs from the last N days (default: all)"
    )
    list_parser.add_argument(
        "--model",
        type=str,
        help="Filter by model name"
    )
    list_parser.add_argument(
        "--analysis",
        type=str,
        help="Filter by analysis type"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Create a comparison directory")
    compare_parser.add_argument(
        "dirs",
        nargs="+",
        type=str,
        help="Directories to compare (can be directory names or paths)"
    )
    compare_parser.add_argument(
        "--name",
        type=str,
        help="Name for the comparison"
    )
    compare_parser.add_argument(
        "--files",
        nargs="+",
        type=str,
        help="Key filenames or patterns to collect for comparison (e.g., '*.txt' or 'output_comparison.txt')"
    )
    
    args = parser.parse_args()
    
    # Initialize output manager
    output_mgr = OutputManager()
    
    if args.command == "dashboard":
        print("Generating dashboard...")
        output_mgr._update_dashboard_index()
        dashboard_path = os.path.join(output_mgr.dashboard_dir, "dashboard.html")
        print(f"✅ Dashboard generated: {dashboard_path}")
        
    elif args.command == "open-dashboard":
        if not open_dashboard(output_mgr):
            print("Could not open dashboard automatically. Try running:")
            print(f"  xdg-open {os.path.join(output_mgr.dashboard_dir, 'dashboard.html')}")
        
    elif args.command == "clean":
        print(f"Cleaning outputs older than {args.days} days...")
        output_mgr.clean_old_outputs(days_to_keep=args.days)
        print("✅ Cleanup complete!")
        
    elif args.command == "list":
        results = output_mgr.list_outputs(days=args.days, model=args.model, analysis_type=args.analysis)
        
        if not results:
            print("No outputs found matching the criteria.")
            return
        
        print("\n📊 AVAILABLE OUTPUTS:")
        print("=" * 80)
        
        # Group by date
        outputs_by_date = {}
        for output in results:
            date = output['date']
            if date not in outputs_by_date:
                outputs_by_date[date] = []
            outputs_by_date[date].append(output)
        
        # Print outputs grouped by date
        for date in sorted(outputs_by_date.keys(), reverse=True):
            year = date[:4]
            month = date[4:6]
            day = date[6:]
            formatted_date = f"{year}-{month}-{day}"
            
            print(f"\n📅 {formatted_date} ({len(outputs_by_date[date])} outputs)")
            print("-" * 80)
            
            for output in sorted(outputs_by_date[date], key=lambda x: x['time'], reverse=True):
                time_str = f"{output['time'][:2]}:{output['time'][2:4]}:{output['time'][4:]}"
                print(f"  [{time_str}] {output['analysis_type']} - {output['model_name']}")
                print(f"      {output['path']}")
        
        print("\nTip: Use the 'dashboard' command to view a more detailed dashboard.")
    
    elif args.command == "compare":
        # Resolve directory paths
        output_dirs = []
        base_dir = output_mgr.base_output_dir
        
        for dir_arg in args.dirs:
            # Check if it's a full path or just a directory name
            if os.path.isabs(dir_arg) and os.path.exists(dir_arg):
                # Absolute path
                output_dirs.append(dir_arg)
            elif os.path.exists(os.path.join(base_dir, dir_arg)):
                # Directory name in outputs folder
                output_dirs.append(os.path.join(base_dir, dir_arg))
            else:
                # Try to find a directory that contains this string
                possible_dirs = [d for d in os.listdir(base_dir) 
                               if os.path.isdir(os.path.join(base_dir, d)) and dir_arg in d]
                if possible_dirs:
                    output_dirs.append(os.path.join(base_dir, possible_dirs[0]))
                else:
                    print(f"Warning: Could not find directory matching '{dir_arg}', skipping.")
        
        if not output_dirs:
            print("Error: No valid directories found to compare.")
            return
        
        if len(output_dirs) < 2:
            print("Error: Need at least 2 directories to compare.")
            return
        
        print(f"Creating comparison of {len(output_dirs)} directories:")
        for dir_path in output_dirs:
            print(f"  - {os.path.basename(dir_path)}")
        
        comparison_dir = output_mgr.create_comparison(
            output_dirs,
            comparison_name=args.name,
            key_files=args.files
        )
        
        print(f"✅ Comparison created: {comparison_dir}")
        if args.files:
            print(f"   Collected files are in: {os.path.join(comparison_dir, 'collected_files')}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()