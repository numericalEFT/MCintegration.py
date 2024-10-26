import os

docs_source_dir = './source'

target_file = 'src.rst'

target_file_path = os.path.join(docs_source_dir, target_file)

if os.path.exists(target_file_path):
    with open(target_file_path, 'r') as f:
        content = f.read()
    content = content.replace('src.', '')
    with open(target_file_path, 'w') as f:
        f.write(content)