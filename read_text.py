category = "cat"
text_path = f"top_1000_{category}_indices.txt"
with open(text_path, 'r') as f:
    print([int(line.rstrip()) for line in f.readlines()])