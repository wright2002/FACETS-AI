import json
from collections import defaultdict

def find_duplicates(file_metadata_list):
    hash_map = defaultdict(list)

    for file_info in file_metadata_list:
        file_hash = file_info.get("hash_sha256")
        if file_hash and not file_hash.startswith("ERROR"):
            hash_map[file_hash].append(file_info["path"])

    # Only keep hashes with more than one file path (duplicates)
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    return duplicates

def generate_cleanup_report(file_metadata_path, report_output_path):
    with open(file_metadata_path, "r") as f:
        file_metadata = json.load(f)

    duplicates = find_duplicates(file_metadata)

    report = {
        "duplicate_files": [],
        "summary": {
            "total_files_scanned": len(file_metadata),
            "total_duplicate_sets": len(duplicates),
            "total_duplicate_files": sum(len(paths) for paths in duplicates.values())
        }
    }

    for file_hash, paths in duplicates.items():
        report["duplicate_files"].append({
            "hash_sha256": file_hash,
            "files": paths,
            "keep": paths[0],  # Keep the first, mark others for removal
            "remove": paths[1:]
        })

    with open(report_output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_output_path

def main():
    generate_cleanup_report("scan_results_files.json", "cleanup_report.json")

main()