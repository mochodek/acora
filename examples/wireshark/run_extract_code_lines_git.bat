python ../../scripts/extract_code_lines_git.py  ^
     --output_lines_path "data\\lines.json" ^
     --repo_path "..\\git\\wireshark" ^
     --file_extensions ".c" ".h" ^
     --last_commit_before_date "2015-02-23"
