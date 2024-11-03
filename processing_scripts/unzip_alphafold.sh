for file in data/raw_data/UP000005640_9606_HUMAN_v4/*.pdb.gz; do
    # Check if the file exists and is readable
    if [ -r "$file" ]; then
        echo "Unzipping $file..."
        gunzip -k "$file"
    else
        echo "Error: Cannot read $file"
    fi
done

echo "All .pdb.gz files have been unzipped"