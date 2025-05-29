import os
import hashlib
import mimetypes
import json
from datetime import datetime

def get_file_hash(file_path, block_size=65536):
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                hasher.update(block)
        return hasher.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

def scan_directory(root_path):
    file_data = []
    directory_summary = {}

    for dirpath, dirnames, filenames in os.walk(root_path):
        relative_path = os.path.relpath(dirpath, root_path)
        # write dictionary for dirs
        directory_summary[relative_path] = {
            "full_path": dirpath,
            "file_count": len(filenames),
            "subdir_count": len(dirnames)
        }

        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            try:
                stat = os.stat(full_path)
                file_hash = get_file_hash(full_path)
                mime_type, _ = mimetypes.guess_type(full_path)
                # write dictionary for files
                file_info = {
                    "path": full_path,
                    "name": filename,
                    "size_bytes": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "mime_type": mime_type,
                    "hash_sha256": file_hash
                }
                file_data.append(file_info)
            except Exception as e:
                file_data.append({
                    "path": full_path,
                    "error": str(e)
                })

    return file_data, directory_summary

drive = input("Enter the full path to the drive: ")

# write dictionaries to json files
scanned_files, scanned_dirs = scan_directory(drive)
json_files = json.dumps(scanned_files, indent=2)
json_dirs = json.dumps(scanned_dirs, indent=2)
with open("scan_results_files.json", "w") as f:
    f.write(json_files)
with open("scan_results_dirs.json", "w") as d:
    d.write(json_dirs)
