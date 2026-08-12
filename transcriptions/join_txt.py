import glob

# Set the name of your output file
output_filename = "combined_output.txt"

# Find all text files in the current folder
txt_files = glob.glob("*.txt")

# Prevent the script from trying to combine the output file into itself
if output_filename in txt_files:
    txt_files.remove(output_filename)

# Open the new master file and append data line by line
with open(output_filename, "w", encoding="utf-8") as outfile:
    for filename in sorted(txt_files):

        # --- NEW: Write the divider and the filename ---
        outfile.write(f"=========== {filename} ===========\n")

        with open(filename, "r", encoding="utf-8") as infile:
            # Read and write line-by-line to prevent running out of RAM
            for line in infile:
                outfile.write(line)

            # Add a couple of newlines to cleanly separate the next divider
            outfile.write("\n\n")

print("Files combined successfully!")