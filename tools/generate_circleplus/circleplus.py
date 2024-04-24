import argparse
from circle_plus_viewer import CirclePlusViewer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='circle+ tool', description='Manage circle+ file')
    parser.add_argument('input', help='circle+ file')
    parser.add_argument('-s', '--show', action='store_true', help='Show traininfo data')
    parser.add_argument('--show_meta', action='store_true', help='Show metadata')
    parser.add_argument('--show_details', action='store_true', help='Show details')
    args = parser.parse_args()

    if args.show or args.show_meta or args.show_details:
        builder = CirclePlusViewer(args.input)
        builder.show(meta=args.show_meta, details=args.show_details)
