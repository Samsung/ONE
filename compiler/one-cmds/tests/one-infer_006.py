# This script gets one argument and print it

import sys
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        filepath = Path(sys.argv[0])
        sys.exit("Usage: " + filepath.name + " [Word to print]")
    word = sys.argv[1]
    print(word)

if __name__ == '__main__':
    main()
