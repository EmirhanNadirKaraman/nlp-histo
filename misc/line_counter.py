import pathlib

def count_lines_in_codebase(directory=".", extensions=(".py", ".sh")):
    total_lines = 0
    file_counts = {ext: 0 for ext in extensions}
    line_counts = {ext: 0 for ext in extensions}
    
    # List of folders to ignore
    ignore_dirs = {'.git', 'venv', '.venv', 'env', '__pycache__', 'node_modules'}

    base_path = pathlib.Path(directory)

    for file_path in base_path.rglob('*'):
        # Check if any part of the file path is in our ignore list
        if any(ignored in file_path.parts for ignored in ignore_dirs):
            continue
            
        if file_path.suffix in extensions:
            try:
                with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
                
                line_counts[file_path.suffix] += lines
                file_counts[file_path.suffix] += 1
                total_lines += lines
            except Exception:
                continue

    # Results Table
    print(f"{'Extension':<12} | {'Files':<8} | {'Lines':<10}")
    print("-" * 35)
    for ext in extensions:
        print(f"{ext:<12} | {file_counts[ext]:<8} | {line_counts[ext]:<10}")
    print("-" * 35)
    print(f"{'TOTAL':<12} | {sum(file_counts.values()):<8} | {total_lines:<10}")

if __name__ == "__main__":
    count_lines_in_codebase()