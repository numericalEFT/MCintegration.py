import re

def remove_prefix_and_horizontal_lines(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    cleaned_lines = []
    
    for i, line in enumerate(lines):
        if re.match(r"^[A-Za-z0-9._ ]+ module", line.strip()) and 'MCintegration.' in line:
            cleaned_lines.append(line.replace('MCintegration.', '').strip() + '\n')
        elif line.strip().startswith('.. automodule::'):
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    with open(filename, 'w') as file:
        file.writelines(cleaned_lines)

remove_prefix_and_horizontal_lines('./source/MCintegration.rst')
