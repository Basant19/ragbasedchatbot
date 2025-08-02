import os

directory = r"D:\ragbasedchatbot"  # Raw string to handle backslashes
output_file = "directory_structure.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for root, dirs, files in os.walk(directory):
        f.write(f"\n📁 Folder: {root}\n")
        for d in dirs:
            f.write(f"   📂 {d}\n")
        for file in files:
            f.write(f"   📄 {file}\n")

print(f"Directory structure saved to: {output_file}")
