import os
import time
import shutil
import glob
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

class OutputManager:
    """
    Enhanced output manager for organizing and navigating research outputs
    """
    
    def __init__(self, base_output_dir="outputs"):
        """Initialize the output manager with the base output directory"""
        self.base_output_dir = base_output_dir
        
        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)
        
        # Define pattern for parsing output directory names
        self.dir_pattern = re.compile(r'(.+?)_(.+?)_(\d{8})_(\d{6})')
        
        # Create dashboard directory
        self.dashboard_dir = os.path.join(self.base_output_dir, "_dashboard")
        if not os.path.exists(self.dashboard_dir):
            os.makedirs(self.dashboard_dir)

    def get_output_dir(self, analysis_type, model_name, include_timestamp=True):
        """
        Generate an output directory with a consistent naming structure
        
        Args:
            analysis_type: Type of analysis being performed
            model_name: Name of the model being analyzed
            include_timestamp: Whether to include a timestamp in the directory name
            
        Returns:
            Full path to the output directory
        """
        # Create base output directory if it doesn't exist
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)
            
        # Generate directory name
        if include_timestamp:
            date_str = time.strftime("%Y%m%d")
            time_str = time.strftime("%H%M%S")
            dir_name = f"{analysis_type}_{model_name}_{date_str}_{time_str}"
        else:
            dir_name = f"{analysis_type}_{model_name}"
            
        # Create full path
        output_dir = os.path.join(self.base_output_dir, dir_name)
        
        # Create directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create symlink in daily directory for easy access
        self._create_daily_symlink(output_dir, analysis_type, model_name)
        
        # Create symlink in the latest directory
        self._update_latest_symlink(output_dir, analysis_type, model_name)
        
        # Update the index file
        self._update_dashboard_index()
        
        return output_dir
    
    def _create_daily_symlink(self, output_dir, analysis_type, model_name):
        """Create a symlink in the daily directory for quick access"""
        today = time.strftime("%Y%m%d")
        daily_dir = os.path.join(self.dashboard_dir, today)
        
        if not os.path.exists(daily_dir):
            os.makedirs(daily_dir)
            
        # Create a symlink with a friendly name
        symlink_name = f"{analysis_type}_{model_name}"
        symlink_path = os.path.join(daily_dir, symlink_name)
        
        # Remove existing symlink if it exists
        if os.path.islink(symlink_path):
            os.remove(symlink_path)
            
        # Create relative symlink
        rel_path = os.path.relpath(output_dir, daily_dir)
        os.symlink(rel_path, symlink_path)
    
    def _update_latest_symlink(self, output_dir, analysis_type, model_name):
        """Update the 'latest' symlink for this analysis type and model"""
        latest_dir = os.path.join(self.dashboard_dir, "latest")
        
        if not os.path.exists(latest_dir):
            os.makedirs(latest_dir)
            
        # Create a symlink with a friendly name
        symlink_name = f"{analysis_type}_{model_name}"
        symlink_path = os.path.join(latest_dir, symlink_name)
        
        # Remove existing symlink if it exists
        if os.path.islink(symlink_path):
            os.remove(symlink_path)
            
        # Create relative symlink
        rel_path = os.path.relpath(output_dir, latest_dir)
        os.symlink(rel_path, symlink_path)
    
    def get_output_path(self, filename, analysis_type, model_name, extension="txt", include_timestamp=False):
        """
        Generate an output file path within the structured directory
        
        Args:
            filename: Base name for the file
            analysis_type: Type of analysis being performed
            model_name: Name of the model being analyzed
            extension: File extension (without the dot)
            include_timestamp: Whether to include a timestamp in the filename
            
        Returns:
            Full path where the file should be saved
        """
        # Get the output directory
        output_dir = self.get_output_dir(analysis_type, model_name)
        
        # Generate filename with optional timestamp
        if include_timestamp:
            timestamp = time.strftime("%H%M%S")
            full_filename = f"{filename}_{timestamp}.{extension}"
        else:
            full_filename = f"{filename}.{extension}"
            
        return os.path.join(output_dir, full_filename)
    
    def _update_dashboard_index(self):
        """Update the dashboard index with the latest outputs"""
        # Create index structure
        index = {
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'by_date': {},
            'by_model': {},
            'by_analysis': {}
        }
        
        # Collect all output directories that match the pattern
        output_dirs = []
        for item in os.listdir(self.base_output_dir):
            item_path = os.path.join(self.base_output_dir, item)
            if os.path.isdir(item_path) and not item.startswith('_'):
                match = self.dir_pattern.match(item)
                if match:
                    analysis_type, model_name, date_str, time_str = match.groups()
                    output_dirs.append({
                        'path': item_path,
                        'name': item,
                        'analysis_type': analysis_type,
                        'model_name': model_name,
                        'date': date_str,
                        'time': time_str
                    })
        
        # Sort by date and time (newest first)
        output_dirs.sort(key=lambda x: (x['date'], x['time']), reverse=True)
        
        # Build the index structure
        for dir_info in output_dirs:
            date = dir_info['date']
            model = dir_info['model_name']
            analysis = dir_info['analysis_type']
            
            # Organize by date
            if date not in index['by_date']:
                index['by_date'][date] = []
            index['by_date'][date].append(dir_info)
            
            # Organize by model
            if model not in index['by_model']:
                index['by_model'][model] = []
            index['by_model'][model].append(dir_info)
            
            # Organize by analysis type
            if analysis not in index['by_analysis']:
                index['by_analysis'][analysis] = []
            index['by_analysis'][analysis].append(dir_info)
        
        # Save the index
        index_path = os.path.join(self.dashboard_dir, 'index.json')
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        # Generate HTML dashboard
        self._generate_html_dashboard(index)
    
    def _generate_html_dashboard(self, index):
        """Generate HTML dashboard from the index data"""
        html_path = os.path.join(self.dashboard_dir, 'dashboard.html')
        
        with open(html_path, 'w') as f:
            f.write(f'''<!DOCTYPE html>
<html>
<head>
    <title>Villanova Research Output Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }}
        .container {{
            width: 95%;
            margin: 20px auto;
        }}
        header {{
            background-color: #003366;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .tab {{
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
        }}
        .tab button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 17px;
        }}
        .tab button:hover {{
            background-color: #ddd;
        }}
        .tab button.active {{
            background-color: #003366;
            color: white;
        }}
        .tabcontent {{
            display: none;
            padding: 6px 12px;
            border: 1px solid #ccc;
            border-top: none;
            background-color: white;
        }}
        .date-group, .model-group, .analysis-group {{
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{background-color: #f5f5f5;}}
        .last-updated {{
            text-align: right;
            font-style: italic;
            color: #666;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Villanova Research Output Dashboard</h1>
    </header>
    
    <div class="container">
        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'ByDate')">By Date</button>
            <button class="tablinks" onclick="openTab(event, 'ByModel')">By Model</button>
            <button class="tablinks" onclick="openTab(event, 'ByAnalysis')">By Analysis Type</button>
        </div>
        
        <div id="ByDate" class="tabcontent" style="display: block;">
            <h2>Outputs By Date</h2>''')
            
            # Add date sections
            for date, dirs in index['by_date'].items():
                year = date[:4]
                month = date[4:6]
                day = date[6:]
                formatted_date = f"{year}-{month}-{day}"
                
                f.write(f'''
            <div class="date-group">
                <h3>{formatted_date}</h3>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Analysis Type</th>
                        <th>Model</th>
                        <th>Directory</th>
                    </tr>''')
                
                for dir_info in dirs:
                    time_str = f"{dir_info['time'][:2]}:{dir_info['time'][2:4]}:{dir_info['time'][4:]}"
                    rel_path = os.path.relpath(dir_info['path'], self.dashboard_dir)
                    
                    f.write(f'''
                    <tr>
                        <td>{time_str}</td>
                        <td>{dir_info['analysis_type']}</td>
                        <td>{dir_info['model_name']}</td>
                        <td><a href="../{rel_path}" target="_blank">{dir_info['name']}</a></td>
                    </tr>''')
                
                f.write('''
                </table>
            </div>''')
            
            f.write('''
        </div>
        
        <div id="ByModel" class="tabcontent">
            <h2>Outputs By Model</h2>''')
            
            # Add model sections
            for model, dirs in index['by_model'].items():
                f.write(f'''
            <div class="model-group">
                <h3>Model: {model}</h3>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Analysis Type</th>
                        <th>Directory</th>
                    </tr>''')
                
                for dir_info in dirs:
                    date_str = f"{dir_info['date'][:4]}-{dir_info['date'][4:6]}-{dir_info['date'][6:]}"
                    time_str = f"{dir_info['time'][:2]}:{dir_info['time'][2:4]}:{dir_info['time'][4:]}"
                    rel_path = os.path.relpath(dir_info['path'], self.dashboard_dir)
                    
                    f.write(f'''
                    <tr>
                        <td>{date_str}</td>
                        <td>{time_str}</td>
                        <td>{dir_info['analysis_type']}</td>
                        <td><a href="../{rel_path}" target="_blank">{dir_info['name']}</a></td>
                    </tr>''')
                
                f.write('''
                </table>
            </div>''')
            
            f.write('''
        </div>
        
        <div id="ByAnalysis" class="tabcontent">
            <h2>Outputs By Analysis Type</h2>''')
            
            # Add analysis type sections
            for analysis, dirs in index['by_analysis'].items():
                f.write(f'''
            <div class="analysis-group">
                <h3>Analysis: {analysis}</h3>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Model</th>
                        <th>Directory</th>
                    </tr>''')
                
                for dir_info in dirs:
                    date_str = f"{dir_info['date'][:4]}-{dir_info['date'][4:6]}-{dir_info['date'][6:]}"
                    time_str = f"{dir_info['time'][:2]}:{dir_info['time'][2:4]}:{dir_info['time'][4:]}"
                    rel_path = os.path.relpath(dir_info['path'], self.dashboard_dir)
                    
                    f.write(f'''
                    <tr>
                        <td>{date_str}</td>
                        <td>{time_str}</td>
                        <td>{dir_info['model_name']}</td>
                        <td><a href="../{rel_path}" target="_blank">{dir_info['name']}</a></td>
                    </tr>''')
                
                f.write('''
                </table>
            </div>''')
            
            f.write(f'''
        </div>
        
        <p class="last-updated">Last updated: {index['last_updated']}</p>
    </div>
    
    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
    </script>
</body>
</html>''')

    def clean_old_outputs(self, days_to_keep=30):
        """
        Clean up output directories that are older than the specified number of days
        
        Args:
            days_to_keep: Number of days to keep output directories (default: 30)
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_date_str = cutoff_date.strftime("%Y%m%d")
        
        # Iterate through all directories in the output folder
        for item in os.listdir(self.base_output_dir):
            item_path = os.path.join(self.base_output_dir, item)
            
            # Skip non-directories and dashboard directory
            if not os.path.isdir(item_path) or item.startswith('_'):
                continue
            
            # Check if the directory name matches our expected format
            match = self.dir_pattern.match(item)
            if match:
                _, _, date_str, _ = match.groups()
                
                # If the directory is older than the cutoff date, remove it
                if date_str < cutoff_date_str:
                    print(f"Removing old output directory: {item}")
                    shutil.rmtree(item_path)
        
        # Update the index after cleaning
        self._update_dashboard_index()
    
    def create_comparison(self, output_dirs, comparison_name=None, key_files=None):
        """
        Create a comparison directory that links to multiple output directories
        
        Args:
            output_dirs: List of output directories to compare
            comparison_name: Optional name for the comparison
            key_files: List of filenames to collect for comparison
            
        Returns:
            Path to the comparison directory
        """
        if comparison_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            comparison_name = f"comparison_{timestamp}"
        
        # Create comparison directory
        comparison_dir = os.path.join(self.base_output_dir, f"_comparison_{comparison_name}")
        if os.path.exists(comparison_dir):
            shutil.rmtree(comparison_dir)
        os.makedirs(comparison_dir)
        
        # Create symlinks to each output directory
        for i, dir_path in enumerate(output_dirs):
            dir_name = os.path.basename(dir_path)
            symlink_path = os.path.join(comparison_dir, f"run_{i+1}_{dir_name}")
            
            # Create relative symlink
            rel_path = os.path.relpath(dir_path, comparison_dir)
            os.symlink(rel_path, symlink_path)
        
        # Collect key files if specified
        if key_files:
            collected_dir = os.path.join(comparison_dir, "collected_files")
            os.makedirs(collected_dir)
            
            for file_pattern in key_files:
                for i, dir_path in enumerate(output_dirs):
                    dir_name = os.path.basename(dir_path)
                    matches = glob.glob(os.path.join(dir_path, file_pattern))
                    
                    for match in matches:
                        file_name = os.path.basename(match)
                        dest_name = f"run_{i+1}_{dir_name}_{file_name}"
                        dest_path = os.path.join(collected_dir, dest_name)
                        
                        # Copy the file
                        shutil.copy2(match, dest_path)
        
        return comparison_dir

    def list_outputs(self, days=None, model=None, analysis_type=None):
        """
        List available outputs, optionally filtered by days, model, and analysis type
        
        Args:
            days: Optional number of days to look back (if None, list all)
            model: Optional model name to filter by
            analysis_type: Optional analysis type to filter by
            
        Returns:
            List of matching output directories
        """
        # Get index data
        index_path = os.path.join(self.dashboard_dir, 'index.json')
        if not os.path.exists(index_path):
            self._update_dashboard_index()
            
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        # Start with all directories
        all_dirs = []
        for date_dirs in index['by_date'].values():
            all_dirs.extend(date_dirs)
        
        # Apply filters
        filtered_dirs = all_dirs
        
        if days is not None:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
            filtered_dirs = [d for d in filtered_dirs if d['date'] >= cutoff_date]
        
        if model is not None:
            filtered_dirs = [d for d in filtered_dirs if d['model_name'] == model]
        
        if analysis_type is not None:
            filtered_dirs = [d for d in filtered_dirs if d['analysis_type'] == analysis_type]
        
        return filtered_dirs
    
    def find_latest(self, model=None, analysis_type=None):
        """
        Find the most recent output directory matching the given criteria
        
        Args:
            model: Optional model name to filter by
            analysis_type: Optional analysis type to filter by
            
        Returns:
            Path to the most recent matching output directory, or None if not found
        """
        # List outputs with the given filters
        outputs = self.list_outputs(model=model, analysis_type=analysis_type)
        
        # Return the path of the first (most recent) output, or None if none found
        if outputs:
            return outputs[0]['path']
        else:
            return None