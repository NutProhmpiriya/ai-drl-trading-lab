import argparse
import os
from utils.google_drive_manager import GoogleDriveManager

def main():
    parser = argparse.ArgumentParser(description='Google Drive File Manager for AI DRL Trading')
    parser.add_argument('action', choices=['upload', 'download', 'list'], help='Action to perform')
    parser.add_argument('--type', choices=['csv_files', 'models'], help='Type of file (required for upload/list)')
    parser.add_argument('--file', help='Local file path for upload or download')
    parser.add_argument('--file-id', help='Google Drive file ID for download')
    
    args = parser.parse_args()
    
    drive_manager = GoogleDriveManager()
    
    if args.action == 'upload':
        if not args.type or not args.file:
            parser.error("upload requires --type and --file arguments")
        if not os.path.exists(args.file):
            parser.error(f"File not found: {args.file}")
        drive_manager.upload_file(args.file, args.type)
    
    elif args.action == 'download':
        if not args.file_id or not args.file:
            parser.error("download requires --file-id and --file arguments")
        drive_manager.download_file(args.file_id, args.file)
    
    elif args.action == 'list':
        if not args.type:
            parser.error("list requires --type argument")
        drive_manager.list_files(args.type)

if __name__ == '__main__':
    main()
