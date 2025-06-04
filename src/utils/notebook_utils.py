from IPython.display import display, HTML
import json
from pathlib import Path
import os
import sys

def create_notebook_cells(notebook_path: str | Path):
    """
    Programmatically create and populate cells in a Kaggle notebook from kaggle_notebook.py.
    
    Args:
        notebook_path: Path to the kaggle_notebook.py file
    """
    try:
        # Read the notebook content
        with open(notebook_path, 'r') as f:
            content = f.read()
        
        # Split content into cells (separated by # %%)
        cells = content.split('# %%')
        
        print(f"Found {len(cells)-1} cells to create")
        
        # Create a cell for each section
        for i, cell_content in enumerate(cells[1:], 1):  # Skip the first empty split
            print(f"Creating cell {i}...")
            
            # Create cell HTML
            cell_html = f"""
            <div class="cell" style="margin-bottom: 20px;">
                <div class="input_area">
                    <div class="CodeMirror cm-s-ipython">
                        <pre class="CodeMirror-line"><span class="cm-comment"># Cell {i}</span></pre>
                        <pre class="CodeMirror-line">{cell_content.strip()}</pre>
                    </div>
                </div>
            </div>
            """
            # Display the cell
            display(HTML(cell_html))
            
            # Execute the cell content
            get_ipython().run_cell(cell_content.strip())
            
        print("All cells created and executed successfully")
            
    except Exception as e:
        print(f"Error creating notebook cells: {str(e)}", file=sys.stderr)
        raise

def setup_kaggle_notebook():
    """
    Setup the Kaggle notebook environment and create cells from kaggle_notebook.py.
    This function assumes the repository is already cloned and we're in the correct directory.
    """
    try:
        # Verify we're in the correct directory
        if not Path('kaggle_notebook.py').exists():
            raise FileNotFoundError("kaggle_notebook.py not found. Make sure you're in the YaleGWI directory")
        
        # Check if competition data is available
        data_dir = Path('/kaggle/input/waveform-inversion')
        if not data_dir.exists():
            print("\n⚠️  Competition data not found!")
            print("\nPlease add the competition dataset to your notebook:")
            print("1. Click on the 'Data' tab")
            print("2. Click 'Add Data'")
            print("3. Search for 'Waveform Inversion'")
            print("4. Click 'Add' on the competition dataset")
            print("\nAfter adding the dataset, re-run this cell.")
            return
        
        # Set environment
        os.environ['GWI_ENV'] = 'kaggle'
        
        # Import and setup
        from src.core.config import CFG
        from src.core.setup import setup_environment
        setup_environment()
        
        # Create and populate cells
        create_notebook_cells('kaggle_notebook.py')
        
    except Exception as e:
        print(f"\n❌ Error setting up notebook: {str(e)}", file=sys.stderr)
        if "kaggle_notebook.py not found" in str(e):
            print("\nMake sure you've cloned the repository correctly:")
            print("1. Run the first cell to clone the repository")
            print("2. Wait for it to complete")
            print("3. Then run this cell again")
        raise 