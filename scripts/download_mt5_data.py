import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.google_drive import GoogleDriveConnector

def download_mt5_price_data():
    # Initialize Google Drive connector
    gdrive = GoogleDriveConnector()
    
    # Set credentials path
    credentials_path = os.path.join('src', 'credentials', 'credentials.json')
    token_path = os.path.join('src', 'credentials', 'token.pickle')
    
    # Authenticate with Google Drive
    gdrive.authenticate(credentials_path=credentials_path, token_path=token_path)
    
    # Create raw data directory if it doesn't exist
    raw_data_dir = os.path.join('src', 'data', 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # Find the AI-DRL folder
    folders = gdrive.list_files(file_type='application/vnd.google-apps.folder')
    ai_drl_folder_id = None
    for folder in folders:
        if folder['name'] == 'AI-DRL':
            ai_drl_folder_id = folder['id']
            break
    
    if not ai_drl_folder_id:
        raise Exception("AI-DRL folder not found in Google Drive")
    
    # Find the mt5_price_data folder inside AI-DRL
    mt5_folders = gdrive.list_files(folder_id=ai_drl_folder_id, 
                                  file_type='application/vnd.google-apps.folder')
    mt5_folder_id = None
    for folder in mt5_folders:
        if folder['name'] == 'mt5_price_data':
            mt5_folder_id = folder['id']
            break
    
    if not mt5_folder_id:
        raise Exception("mt5_price_data folder not found in AI-DRL folder")
    
    # List and download all CSV files
    csv_files = gdrive.list_files(folder_id=mt5_folder_id)
    for file in csv_files:
        if file['name'].endswith('.csv'):
            output_path = os.path.join(raw_data_dir, file['name'])
            print(f"Downloading {file['name']}...")
            gdrive.download_file(file['id'], output_path)
            print(f"Successfully downloaded to {output_path}")

if __name__ == "__main__":
    download_mt5_price_data()
